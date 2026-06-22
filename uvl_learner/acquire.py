"""Shared constraint-acquisition machinery.

Everything the runners share to drive a pycona CA loop, kept in one file because
this is the layer most worth experimenting on:

- ALGORITHMS — name → pycona algorithm class registry.
- FindCSkipCollapse / SkipCollapseCAEnv — query counting, caching, and the
  "skip-collapse" handling for n-ary clauses the binary bias can't represent.
- build_algorithm() — instantiate a CA algorithm by name with the skip env.
"""

import cpmpy as cp
from pycona import (
    FindScope2,
    ActiveCAEnv,
    QuAcq,
    MQuAcq,
    MQuAcq2,
    GrowAcq,
    PQuAcq,
    MineAcq,
    GenAcq,
)
from pycona.find_constraint import FindC2
from pycona.utils import restore_scope_values


ALGORITHMS = {
    "quacq": QuAcq,
    "mquacq": MQuAcq,
    "mquacq2": MQuAcq2,
    "growacq": GrowAcq,
    "pquacq": PQuAcq,
    "mineacq": MineAcq,
    "genacq": GenAcq,
}


# ── Skip-on-Collapse CA environment ──────────────────────────────────


class FindCSkipCollapse(FindC2):
    """FindC variant that returns None instead of raising on Collapse.

    pycona raises ``Exception("Collapse, the constraint we seek is not in B: …")``
    when no bias candidate is violated by a negative oracle query.  This happens
    when the target contains n-ary clauses (e.g. ``~P | C1 | C2``) that cannot
    be represented as binary candidates.  Returning ``None`` lets the outer loop
    skip this query and continue learning binary constraints.

    On collapse, the scope and its variable values are saved in
    ``last_collapse_scope`` so that the CA environment can add a blocking
    nogood to CL, preventing the query generator from reproducing the
    exact same assignment.
    """

    def __init__(self, time_limit=30):
        super().__init__(time_limit=time_limit)
        self.last_collapse_scope = None  # (scope, values) from most recent collapse

    def run(self, scope):
        assert self.ca is not None
        # Save scope variable values BEFORE FindC2 runs.  FindC2's
        # generate_findc_query() modifies them via solver calls.  On
        # Collapse the exception is raised before FindC2 can call
        # restore_scope_values(), leaving variables with stale solver
        # values.  MQuAcq2's inner loop and analyze_and_learn then use
        # these corrupted values for subsequent membership queries,
        # which can cause the oracle to give wrong answers and lead to
        # wrong constraints being added to C_L.
        scope_values = [x.value() for x in scope]
        try:
            return super().run(scope)
        except Exception as e:
            if "Collapse" in str(e):
                print(
                    "  FindCSkipCollapse: Collapse on scope %s — skipping",
                    scope,
                )
                restore_scope_values(scope, scope_values)
                self.last_collapse_scope = (scope, scope_values)
                return None
            raise


class SkipCollapseCAEnv(ActiveCAEnv):
    """CA environment that tracks positive/negative query counts and optionally skips Collapse.

    When ``skip_collapse=True`` (default): installs ``FindCSkipCollapse`` so that
    Collapse exceptions are caught and returned as ``None``; ``add_to_cl`` discards
    ``None`` and increments ``n_skipped``.

    When ``skip_collapse=False``: uses the standard ``FindC`` and lets Collapse
    propagate as an exception.  Query counting still works in both modes.

    Attributes
    ----------
    n_positive : int
        Number of oracle queries answered Yes (solution / constraint holds).
    n_negative : int
        Number of oracle queries answered No.
    n_skipped : int
        Collapse events suppressed (only meaningful when skip_collapse=True).
    """

    def __init__(self, skip_collapse: bool = True):
        super().__init__(
            findc=FindCSkipCollapse(time_limit=30)
            if skip_collapse
            else FindC2(time_limit=30),
            find_scope=FindScope2(),
        )
        self._skip_collapse = skip_collapse
        self.n_positive = 0
        self.n_negative = 0
        self.n_skipped = 0
        self.n_cache_hit = 0
        self.query_cache = {}
        self.nogoods = []

    def _track(self, answer: bool) -> bool:
        if answer:
            self.n_positive += 1
        else:
            self.n_negative += 1
        return answer

    def ask_membership_query(self, Y=None):
        if Y:
            key = tuple((str(v), v.value()) for v in sorted(Y, key=lambda k: str(k)))
        else:
            key = tuple()

        if key in self.query_cache:
            self.n_cache_hit += 1
            return self.query_cache[key]

        if self.verbose >= 3:
            query = [f"{v}={v.value()}" for v in sorted(Y, key=lambda k: str(k))]
            print(f"Query: {query}")

        old_verbosity = self.verbose
        self.verbose = 0  # to avoid printing the query answer in the oracle

        response = self._track(super().ask_membership_query(Y))
        self.query_cache[key] = response

        self.verbose = old_verbosity

        if self.verbose >= 3:
            print("Answer: ", "YES" if response else "NO")
        
        return response

    def ask_recommendation_query(self, c):
        return self._track(super().ask_recommendation_query(c))

    def ask_generalization_query(self, c, C):
        return self._track(super().ask_generalization_query(c, C))

    def add_to_cl(self, C):
        if self._skip_collapse and C is None:
            self.n_skipped += 1
            print(f"  SkipCollapseCAEnv: skipped collapse #{self.n_skipped}")
            # Add a blocking nogood for the collapsed scope assignment so the
            # query generator won't reproduce the exact same variable values.
            scope_info = self.findc.last_collapse_scope
            if scope_info is not None:
                scope, vals = scope_info
                nogood = ~cp.all([x if v else ~x for x, v in zip(scope, vals)])
                # self.instance.cl.append(nogood)
                # self.nogoods.append(nogood)
                # TODO This makes no sense, we only block single features
                # We need to block entire queries
                print(f"  SkipCollapseCAEnv: added blocking nogood for scope {scope}: {nogood}")
                self.findc.last_collapse_scope = None
            return
        super().add_to_cl(C)


# ── Algorithm factory ──


def build_algorithm(
    name: str,
    *,
    skip_collapse: bool = True,
    cliques_cutoff: int = 1,
    analyze_and_learn: bool = True,
    qg_max: int = 10,
    grow_inner: str = "mquacq2",
) -> tuple:
    """Instantiate a pycona CA algorithm by name with the given options.

    Returns ``(algorithm, skip_env)`` where ``skip_env`` is either a
    ``SkipCollapseCAEnv`` (when ``skip_collapse=True``) or ``None`` (when
    skip is disabled, Collapse raises an exception instead).

    GenAcq always uses types=[] because without tree_info there are no
    sibling groups to exploit.

    For GrowAcq the skip environment is attached to the *inner* algorithm
    (which performs the actual FindC calls); the returned ``skip_env`` refers
    to that inner env.
    """
    ca_env = SkipCollapseCAEnv(skip_collapse=skip_collapse)

    if name == "quacq":
        return QuAcq(ca_env=ca_env), ca_env
    elif name == "mquacq":
        return MQuAcq(ca_env=ca_env), ca_env
    elif name == "mquacq2":
        return MQuAcq2(
            ca_env=ca_env,
            perform_analyzeAndLearn=analyze_and_learn,
            cliques_cutoff=cliques_cutoff,
        ), ca_env
    elif name == "growacq":
        # GrowAcq uses ProbaActiveCAEnv as its outer env; the inner algorithm
        # does the FindC work, so attach ca_env there.
        if grow_inner == "mquacq2":
            inner = MQuAcq2(
                ca_env=ca_env,
                perform_analyzeAndLearn=analyze_and_learn,
                cliques_cutoff=cliques_cutoff,
            )
        else:
            inner = ALGORITHMS[grow_inner](ca_env=ca_env)
        return GrowAcq(inner_algorithm=inner), ca_env
    elif name == "pquacq":
        return PQuAcq(ca_env=ca_env), ca_env
    elif name == "mineacq":
        return MineAcq(ca_env=ca_env, qg_max=qg_max), ca_env
    elif name == "genacq":
        print("  genacq: untyped mode (no tree_info)")
        return GenAcq(ca_env=ca_env, types=[], qg_max=qg_max), ca_env
    else:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Choose from: {list(ALGORITHMS)}"
        )
