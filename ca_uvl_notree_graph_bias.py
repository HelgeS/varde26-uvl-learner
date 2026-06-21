import argparse
import itertools
import pickle
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import cpmpy as cp
import networkx as nx
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Expression
from pycona import (
    ActiveCAEnv,
    ConstraintOracle,
    Metrics,
    ProblemInstance,
    UserOracle,
    utils,
)
from pycona.active_algorithms.algorithm_core import AlgorithmCAInteractive
from pycona.answering_queries import Oracle
from pycona.ca_environment.acive_ca_proba import ProbaActiveCAEnv
from pycona.find_constraint.findc_core import FindCBase
from pycona.find_constraint.utils import (
    get_delta_p,
    get_max_conjunction_size,
    join_con_net,
    unravel_conjunctions,
)
from pycona.find_scope.findscope_core import FindScopeBase
from pycona.find_scope.findscope_obj import split_half, split_proba
from pycona.query_generation.qgen_core import QGenBase
from pycona.query_generation.qgen_obj import (
    obj_max_viol,
)
from pycona.utils import (
    Objectives,
    check_value,
    restore_scope_values,
)

from ca_common import (
    ALGORITHMS,
    TimeoutError,
    _timeout_handler,
    collect_uvl_paths,
    extract_feature_names,
    extract_target_constraints,
    save_result,
)
from ca_uvl_notree import build_bias
from tree_inference import (
    _fix_multi_parent_tree,
    _validate_tree,
    cleanup_dumb,
    constraints_from_tree,
    infer_and_refine_tree,
    infer_tree,
)
from uvl_export import export_learned_to_uvl, verify_learned

EXAMPLE_UVL = """\
features
    Sandwich
        mandatory
            Bread
        optional
            Sauce
                alternative
                    Ketchup
                    Mustard
            Cheese

constraints
    Ketchup => Cheese
"""


class GFindC2(FindCBase):
    """
    Implementation of the FindC algorithm from Bessiere et al., "Learning constraints through partial queries" (AIJ 2023).

    This function works also for non-normalised target networks!
    """

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=0.2, findscope=None):
        """
        Initialize the FindC2 class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findc query generation.
        :param findscope: The function to find the scope.
        """
        super().__init__(ca_env, time_limit)
        self._findscope = findscope

    @property
    def findscope(self):
        """
        Get the findscope function to be used.

        Returns:
            callable: The function used to determine constraint scopes
        """
        return self._findscope

    @findscope.setter
    def findscope(self, findscope):
        """
        Set the findscope function to be used.

        Args:
            findscope (callable): The function to be used for determining constraint scopes
        """
        self._findscope = findscope

    def run(self, scope):
        """
        Execute the FindC2 algorithm to learn constraints within a given scope.

        Args:
            scope (list): Variables defining the scope in which to search for constraints

        Returns:
            list: The constraint(s) found in the given scope.

        Raises:
            Exception: If the target constraint is not in the bias (search space).
        """
        assert self.ca is not None
        scope_values = [x.value() for x in scope]

        print(f"FindC2: running with scope={scope} and scope_values={scope_values}")

        # Initialize delta with constraints from bias that match the scope
        delta = get_con_subset(self.ca.instance.bias, scope, scope_size=len(scope))
        # delta = [c for c in delta if len(get_scope(c)) == len(scope)]

        # Join the constraints in delta with the violated constraints in kappaD
        kappaD = [c for c in delta if check_value(c) is False]
        delta = join_con_net(delta, kappaD)

        # Get subset of learned constraints in the current scope
        sub_cl = utils.get_con_subset(self.ca.instance.cl, scope)

        while True:
            # Generate a query to distinguish between candidate constraints
            if len(delta) == 0 or self.generate_findc_query(sub_cl, delta) is False:
                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    # Soft collapse: not a "constraint not in B" failure but
                    # an "FindC could not disambiguate this iteration"
                    # signal. The actual target constraint may well be in
                    # the bias (e.g. MySQL→Database) but is currently
                    # SATISFIED, so kappaD is empty and join_con_net
                    # collapses delta to []. Returning None lets the
                    # learner skip this iteration; a later QGen example
                    # that actually violates the target will surface it.
                    print(
                        f"FindC2: empty delta at scope={scope}; "
                        f"returning None (no disambiguation possible "
                        f"this iteration, will retry from next query)"
                    )
                    return None

                restore_scope_values(scope, scope_values)

                # Unravel nested AND constraints
                delta_unraveled = unravel_conjunctions(delta)

                # Return the smallest equivalent conjunction (if more than one, they are equivalent w.r.t. C_l)
                delta_unraveled = sorted(delta_unraveled, key=lambda x: len(x))
                return delta_unraveled[0]

            self.ca.metrics.increase_findc_queries()

            if self.ca.ask_membership_query(scope):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]
                [
                    self.ca.instance.bias.remove_from_bias(c)
                    for c in delta
                    if check_value(c) is False
                ]

            else:  # user says UNSAT
                # delta <- joint(delta,K_{delta}(e))

                kappaD = [c for c in delta if check_value(c) is False]
                scope2 = self.ca.run_find_scope(list(scope))
                if len(scope2) < len(scope):
                    # Recursively learn constraint in sub-scope
                    c = self.run(scope2)
                    self.ca.add_to_cl(c)
                    sub_cl.append(c)
                else:
                    delta = join_con_net(delta, kappaD)

    def generate_findc_query(self, L, delta):
        """
        Generate a query that helps distinguish between candidate constraints.

        Args:
            L (list): Currently learned constraints in the scope
            delta (list): Candidate constraints to distinguish between

        Returns:
            bool: True if a query was generated successfully, False otherwise

        Note:
            The method directly modifies variable values in the constraint network
        """
        tmp = cp.Model(L)

        satisfied_delta = sum(
            [c for c in delta]
        )  # get the amount of satisfied constraints from B

        scope = utils.get_scope(delta[0])

        # at least 1 violated and at least 1 satisfied
        # we want this to assure that each answer of the user will reduce
        # the set of candidates
        tmp += satisfied_delta < len(delta)
        tmp += satisfied_delta > 0

        max_conj_size = get_max_conjunction_size(delta)
        delta_p = get_delta_p(delta)

        for p in range(max_conj_size):
            s = cp.SolverLookup.get("ortools", tmp)

            kappa_delta_p = sum([c for c in delta_p[p]])
            s += kappa_delta_p < len(delta_p[p])

            # Solve without objective for start
            if not s.solve():  # if a solution is not found
                continue

            # Next solve will change the values of the variables in lY
            # so we need to return them to the original ones to continue if we don't find a solution next
            values = [x.value() for x in scope]

            p_soft_con = kappa_delta_p > 0

            # So a solution was found, try to find a better one now
            # set the objective
            s.maximize(p_soft_con)

            # Give hint with previous solution to the solver
            s.solution_hint(scope, values)

            # Solve with objective
            flag = s.solve(time_limit=self.time_limit, num_workers=8)
            if not flag:
                restore_scope_values(scope, values)
            return True

        return False


# We have to use FindScope2, because FindScope is not working for our models
# Even without the graph bias
class GFindScope2(FindScopeBase):
    """
    This is the version of the FindScope function that was presented in
    Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, split_func=None, time_limit=0.2):
        """
        Initialize the FindScope2 class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findscope query generation.
        """
        super().__init__(ca_env, time_limit, split_func=split_half)
        self._kappaB_pairwise = []
        # Set by the template-guided shortcut so the learner can skip
        # FindC and add the candidate directly. None when the run
        # finished via classical binary search.
        self.last_candidate: Optional[Tuple] = None
        # Set at run() entry: list of frozensets of variable NAMES that
        # should NOT be split across Y1/Y2 in _find_scope. Each cluster
        # corresponds to (the union of overlapping) violated higher-arity
        # candidate scope+parent. Variable values don't change during
        # FindScope recursion, so this is precomputed once.
        self._cluster_names: List[FrozenSet[str]] = []

    def run(self, Y, kappa=None):
        """
        Run the FindScope2 algorithm.

        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If the partial example is not a negative example.
        """
        assert self.ca is not None
        self.last_candidate = None

        # This is cached to avoid repeated calculations, but it's maybe not super necessary
        _kappaB_pairwise = (
            kappa
            if kappa is not None
            else self.ca.instance.bias.get_kappa(Y, extended=False)
        )
        print(f"FindScope2: pairwise kappaB for Y={Y} is {len(_kappaB_pairwise)}")
        self._kappaB_pairwise = _kappaB_pairwise
        # Note: pairwise kappa being empty doesn't preclude a higher-arity
        # violator (templates are not in `_kappaB_pairwise`). The shortcut
        # below handles that case.

        # ---- Template-guided shortcut (FindScope improvement #1+#2)
        # Only fires when pairwise kappa is empty — i.e., binary CA
        # genuinely cannot represent the violator (every pairwise
        # candidate evaluates fine, yet the example is negative). When
        # some pairwise constraint is violated, binary FindScope is
        # more reliable: pairwise scopes are atomic and don't risk the
        # spurious-clique-in-dense-initial-bias trap. The shortcut is
        # the *fallback for higher-arity-only violations*, exactly the
        # case that classically caused Collapse.
        if len(self._kappaB_pairwise) == 0:
            bias = self.ca.instance.bias
            for tmpl, cand in bias.iter_violated_higher_arity_in(Y):
                cand_var_names = set(cand.scope)
                parent = cand.params.get("parent")
                if parent is not None:
                    cand_var_names.add(parent)
                cand_vars = [v for v in Y if v.name in cand_var_names]
                if len(cand_vars) < 2:
                    continue
                self.ca.metrics.increase_findscope_queries()
                if self.ca.ask_membership_query(cand_vars):
                    continue
                self.last_candidate = (tmpl, cand)
                print(
                    f"FindScope2: higher-arity-only shortcut hit "
                    f"{tmpl.type_key}{tuple(sorted(cand_var_names))}"
                )
                return set(cand_vars)
            raise Exception(
                "The partial example e_Y, on the subset of variables Y given in FindScope, "
                "must be a negative example"
            )

        # ---- Precompute clusters for scope-aware splitting (#3).
        # Each cluster is a set of variable names the split must keep on
        # the same side of Y1/Y2. Built from currently-violated higher-
        # arity candidates whose full scope sits inside Y; overlapping
        # candidates merge (transitive closure) into a single cluster.
        # Variable values don't change during FindScope recursion, so
        # this is computed once per run().
        self._cluster_names = self._build_clusters(Y)
        if self._cluster_names:
            print(
                f"FindScope2: precomputed {len(self._cluster_names)} cluster(s) "
                f"to keep intact (sizes "
                f"{sorted([len(c) for c in self._cluster_names], reverse=True)})"
            )

        scope = self._find_scope(set(), Y)
        return scope

    def _build_clusters(self, Y) -> List[FrozenSet[str]]:
        """Build the keep-together clusters used by ``_split_aware``.

        For each currently-violated higher-arity candidate, *expand* its
        scope to the full structural scope (e.g. an AltGroup pair gets
        expanded to the full implies_not clique under the same parent).
        Without this expansion the cluster is the minimal violating
        sub-pattern (a pair + parent for AltGroup), which is smaller
        than the actual target alt-group — FindScope's split would
        still fragment the alt-group and FindC would confirm a wrong
        subset-sized candidate. With expansion the cluster matches the
        scope FindC needs to disambiguate.

        Heuristic: deduplicate by canonical id, take the K smallest by
        scope size. Don't union-merge overlapping clusters — in dense
        bias every True parent yields overlapping candidates and a
        merge collapses to one giant blob.
        """
        bias = self.ca.instance.bias
        seen_keys: Set[Tuple] = set()
        raw: List[FrozenSet[str]] = []
        for tmpl, cand in bias.iter_violated_higher_arity_in(Y):
            full_scope = self._expand_to_structural_scope(tmpl, cand, bias.G)
            parent = cand.params.get("parent")
            cluster_set = set(full_scope)
            if parent is not None:
                cluster_set.add(parent)
            if len(cluster_set) < 2 or len(cluster_set) > self.MAX_CLUSTER_SIZE:
                continue
            key = (tmpl.type_key, tuple(sorted(cluster_set)), parent)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            raw.append(frozenset(cluster_set))
        if not raw:
            return []
        raw.sort(key=len)
        return raw[: self.MAX_CLUSTERS]

    @staticmethod
    def _expand_to_structural_scope(tmpl, cand, G) -> Set[str]:
        """For AltGroup: greedily extend cand.scope by adding kids of the
        parent that are pairwise connected via `implies_not` to the
        current scope members — i.e. grow a clique containing the
        violating pair. For OrGroup the structural scope is already
        the full incoming-implies set of the parent (already returned
        by value-aware detect), no expansion needed.
        """
        if tmpl.type_key != "alternative_group":
            return set(cand.scope)
        parent = cand.params.get("parent")
        if parent is None:
            return set(cand.scope)
        # Candidate kids: parent's incoming-implies set (i.e. nodes
        # implying parent) minus the parent itself.
        kids: Set[str] = set()
        for u, _, k in G.in_edges(parent, keys=True):
            if k == "implies":
                kids.add(u)
        kids.discard(parent)
        full: Set[str] = set(cand.scope) & kids
        if not full:
            return set(cand.scope)

        def _connected(a, b):
            return (
                G.has_edge(a, b, key="implies_not")
                or G.has_edge(b, a, key="implies_not")
            )

        changed = True
        while changed:
            changed = False
            for kid in kids - full:
                if all(_connected(kid, e) for e in full):
                    full.add(kid)
                    changed = True
        return full

    # Cap on the number of clusters considered by the scope-aware split
    # and on each cluster's size. Tuned to favour small, plausibly-real
    # higher-arity scopes (UVL alt/OR groups are typically 2–5 children
    # plus a parent) and to keep _split_aware itself O(|Y|).
    MAX_CLUSTERS = 8
    MAX_CLUSTER_SIZE = 8

    # Scope-aware split is only worth its overhead when binary
    # split_half's recursion depth would be appreciable. For small Y
    # (e.g. <= ~10 vars) split_half converges in a couple of levels
    # and scope-aware can perturb its routing in ways that miss
    # binary constraints — see commit history. Stick with split_half
    # for the small case.
    SCOPE_AWARE_MIN_Y = 12

    def _split_aware(self, Y, R) -> Tuple[list, list]:
        """Scope-aware split: keep each precomputed cluster on the same
        side of the cut. Returns lists so pycona's ``split_half`` can
        slice them on the next recursion. Falls back to ``split_func``
        when no cluster fits inside Y at this recursion level OR when
        the largest applicable cluster is so big it would force a
        degenerate (or near-degenerate) split.
        """
        Y_list = list(Y)
        if len(Y_list) < self.SCOPE_AWARE_MIN_Y:
            return self.split_func(Y=Y_list, R=R)
        if not self._cluster_names:
            return self.split_func(Y=Y_list, R=R)
        Y_names = {v.name for v in Y_list}
        applicable = [c for c in self._cluster_names if c.issubset(Y_names)]
        if not applicable:
            return self.split_func(Y=Y_list, R=R)
        # Discard clusters too large to leave room on the other side.
        # Without this guard we'd produce splits like (Y, ∅) that send
        # _find_scope into infinite recursion.
        max_kept = max(1, len(Y_list) - 1)
        applicable = [c for c in applicable if len(c) <= max_kept]
        if not applicable:
            return self.split_func(Y=Y_list, R=R)
        name_to_var = {v.name: v for v in Y_list}
        applicable_sorted = sorted(applicable, key=len, reverse=True)
        Y1_names: Set[str] = set()
        Y2_names: Set[str] = set()
        for cluster in applicable_sorted:
            # Skip clusters that no longer fit (already partially
            # absorbed by the other side).
            if cluster & Y2_names and cluster & Y1_names:
                continue
            target = Y1_names if len(Y1_names) <= len(Y2_names) else Y2_names
            other = Y2_names if target is Y1_names else Y1_names
            if cluster & other:
                target = other
            target |= cluster
        # Distribute non-cluster vars to balance the two sides.
        remaining = sorted(Y_names - Y1_names - Y2_names)
        for n in remaining:
            target = Y1_names if len(Y1_names) <= len(Y2_names) else Y2_names
            target.add(n)
        # Preserve original Y ordering for downstream stability.
        Y1 = [v for v in Y_list if v.name in Y1_names]
        Y2 = [v for v in Y_list if v.name in Y2_names]
        if not Y1 or not Y2:
            return self.split_func(Y=Y_list, R=R)
        return Y1, Y2

    def _find_scope(self, R, Y):
        """
        Find the scope of the partial example.

        :param R: A set of variables.
        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If kappaB is not part of the bias.
        """
        # print(f"_find_scope({R}, {Y})")

        # TODO It's relatively easy to check if there are any n-arity constraints for a given scope
        # as long as we do not have to materialize all of them.
        # This should help for a more efficient findscope implementation.
        # We can do it with find_clique + nodes set to the scope
        # Especially in UVL we know, that the clique can at most contain |V|-1 variables,
        # because there must be a parent, too

        # Reqs.: includes nodes of edges that are violated and have correct type
        # 1. subgraph for scope
        # 2. foreach violated edge of correct type: find patterns including it (e.g. cliques)

        # So, what we need is the info whether there are any constraints violated
        # For the pairwise it could be easier to bookkeep them as they are, but maybe the minimal demo can just recreate
        # kappaB every time

        # UVL-specific: in our setup, a virtual constraint can only be violated
        # if at least one pairwise constraint is violated (true for sum?)
        pairwise_kappa = self.ca.instance.bias.get_kappa(R)
        
        if len(pairwise_kappa) > 0:
            self.ca.metrics.increase_findscope_queries()
            if self.ca.ask_membership_query(R):
                self.ca.instance.bias.remove_from_bias(pairwise_kappa)
                # Ignore kappa bookkeeping for now
            else:
                return set()
            
        if len(Y) == 1:
            return set(Y)

        # Create Y1, Y2 -------------------------
        Y1, Y2 = self._split_aware(Y=Y, R=R)

        # R U Y
        RY = R.union(Y)
        # R U Y1
        RY1 = R.union(Y1)

        S1 = set()
        S2 = set()

        if not self.ca.instance.bias.same_kappa(RY1, RY):
            S1 = self._find_scope(RY1, Y2)

        # R U S1
        RS1 = R.union(S1)

        if not self.ca.instance.bias.same_kappa(RS1, RY):
            S2 = self._find_scope(RS1, Y1)

        # print(f"union: {S1.union(S2)}")
        return S1.union(S2)


class GPQGen(QGenBase):
    """
    PQGen function for query generation.
    This class implements the query generator from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    """

    def __init__(
        self,
        ca_env: ActiveCAEnv = None,
        *,
        objective_function=None,
        time_limit=1,
        blimit=5000,
    ):
        """
        Initialize the PQGen with the given parameters.

        :param ca_env: The CA environment.
        :param objective_function: The objective function for PQGen.
        :param time_limit: The time limit for query generation.
        :param blimit: The bias limit to start optimization.
        """
        super().__init__(ca_env, time_limit)
        self.partial = False
        if objective_function is None:
            objective_function = obj_max_viol
        self.obj = objective_function
        self.blimit = blimit

    @property
    def obj(self):
        """
        Get the objective of PQGen.

        :return: The objective function.
        """
        return self._obj

    @obj.setter
    def obj(self, obj):
        """
        Set the objective of PQGen.

        :param obj: The objective function to set.
        """
        assert obj in Objectives.qgen_objectives()
        self._obj = obj

    @property
    def blimit(self):
        """
        Get the bias limit to start optimization in PQGen.

        :return: The bias limit.
        """
        return self._blimit

    @blimit.setter
    def blimit(self, blimit):
        """
        Set the bias limit to start optimization in PQGen.

        :param blimit: The bias limit.
        """
        self._blimit = blimit

    def reset_partial(self):
        """
        Reset the partial flag to False.
        """
        self.partial = False

    def generate(self, X=None):
        """
        Generate a query using PQGen.

        :return: A set of variables that form the query.
        """

        if X is None:
            X = self.env.instance.X

        # MODIFIED
        # TODO We generate queries on pairwise only. Okay?
        # B = get_con_subset(self.env.instance.bias, X)
        B = get_con_subset_pairwise(self.env.instance.bias, X)

        # Start time (for the cutoff t)
        t0 = time.time()

        # Project down to only vars in scope of B
        Y = frozenset(get_variables(B))

        lY = list(Y)

        Cl = utils.get_con_subset(self.env.instance.cl, Y)

        # If no constraints left in B, just return
        if len(B) == 0:
            return set()

        # sample from B using the probabilities -------------------
        # If no constraints learned yet, start by just generating an example in all the variables in Y
        if len(Cl) == 0:
            Cl = [cp.sum(Y) >= 1]

        m = cp.Model(Cl)
        s = cp.SolverLookup.get("ortools", m)

        if not self.partial and len(B) > self.blimit:
            flag = s.solve(num_workers=8)  # no time limit to ensure convergence

            if flag and not all([c.value() for c in B]):
                return lY
            else:
                self.partial = True

        # We want at least one constraint to be violated to assure that each answer of the user
        # will lead to new information
        s += ~cp.all(B)

        if self.env.verbose > 2:
            print("Solving first without objective (to find at least one solution)...")

        # Solve first without objective (to find at least one solution)
        flag = s.solve(num_workers=8)

        t1 = time.time() - t0
        if not flag or (t1 > self.time_limit):
            # UNSAT or already above time_limit, stop here --- cannot optimize
            return lY if flag else set()

        # Next solve will change the values of the variables in lY
        # so we need to return them to the original ones to continue if we don't find a solution next
        values = [x.value() for x in lY]

        # So a solution was found, try to find a better one now
        s.solution_hint(lY, values)
        try:
            objective = self.obj(B=B, ca_env=self.env)
        except:
            raise NotImplementedError(
                f"Objective given not implemented in PQGen: {self.obj} - Please report an issue"
            )

        # Run with the objective
        s.maximize(objective)

        if self.env.verbose > 2:
            print("Solving with objective...")

        flag2 = s.solve(time_limit=(self.time_limit - t1), num_workers=8)

        if flag2:
            return lY
        else:
            restore_scope_values(lY, values)
            return lY


def find_cliques_with_common_star(
    G, clique_type="implies_not", star_type="implies", scope_size=None, nodes=None
):
    """Backward-compat shim around ``AlternativeGroupTemplate``.

    The historical contract: enumerate cliques on ``clique_type`` whose
    members all share an outgoing ``star_type`` edge to the same center,
    materialize ``center.implies(sum(clique) == 1)`` for each match.

    The template covers exactly this case (clique_type='implies_not',
    star_type='implies'). Other type combinations are still supported via
    a one-shot ad-hoc enumeration here so the debug scratchpad
    (``nxdbg.py``) keeps working.
    """
    if clique_type == "implies_not" and star_type == "implies":
        tmpl = AlternativeGroupTemplate()
        name_to_var = {n: d["var"] for n, d in G.nodes(data=True)}
        out = []
        for cand in tmpl.detect_scopes(G, nodes=nodes, scope_size=scope_size):
            out.append(tmpl.materialize(cand, name_to_var))
        return out

    # Fallback: replicate the legacy generic enumeration verbatim.
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(G.nodes(data=True))
    for u, v, k in G.edges(keys=True):
        if k == clique_type:
            clique_graph.add_edge(u, v)

    star_targets: dict = {}
    for u, v, k in G.edges(keys=True):
        if k == star_type:
            star_targets.setdefault(u, set()).add(v)

    results = []
    iterator = nx.enumerate_all_cliques(clique_graph)
    if nodes is not None:
        iterator = (
            clique for clique in iterator if any(n in clique for n in nodes)
        )
    for clique in iterator:
        if len(clique) < 2:
            continue
        if scope_size is not None and len(clique) != scope_size - 1:
            continue
        clique_set = set(clique)
        common: Optional[set] = None
        for node in clique:
            targets = star_targets.get(node, set()) - clique_set
            if common is None:
                common = set(targets)
            else:
                common &= targets
            if not common:
                break
        if not common:
            continue
        for center in common:
            cvar = G._node[center]["var"]
            clique_vars = [G._node[n]["var"] for n in clique]
            results.append(cvar.implies(cp.sum(clique_vars) == 1))
    return results


KEYMAP = {
    "->": "implies",
    "->!": "implies_not",
    "!->!": "not_implies_not",
    "!->": "not_implies",
    "==": "equals",
}


# ---------------------------------------------------------------------------
# Layered higher-arity bias: RelationTemplate + per-template index
# ---------------------------------------------------------------------------
#
# Adapted from ../ca_higher_arity/. The split is:
#
#   - The pairwise layer remains nx.MultiDiGraph with edge keys taken from
#     KEYMAP. Edges carry `learned: bool`. Removal is still destructive (other
#     code reads bias.G.edges directly), so we model "eliminated" implicitly
#     by absence-from-graph and "confirmed" by `learned=True`.
#
#   - The virtual layer is per-template. Each template owns a CandidateScope
#     index keyed by canonical_id. Candidates are detected lazily from the
#     pairwise graph; on edge elimination/learning, the template's reverse
#     edge -> candidates index is walked synchronously to mark dependents
#     dead. This replaces the coarse `bias.removed` blocklist.
#
# The public GraphBias surface is unchanged: get_kappa, same_kappa,
# mark_as_learned, remove_from_bias, copy, len, plus the .G and .removed
# attributes. `bias.removed` is still populated (for downstream code that
# reads it) but is no longer the primary cascade mechanism.

class ViolationStatus:
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EdgeKey:
    """Lightweight identifier for an edge in the pairwise multigraph.

    Mirrors what nx stores: (u_name, v_name, rel_type) where rel_type is one
    of KEYMAP's values. Used as the canonical id in template support indices.
    """

    u: str
    v: str
    rel_type: str  # one of KEYMAP.values()

    def in_graph(self, G) -> bool:
        return G.has_edge(self.u, self.v, key=self.rel_type)

    def get_data(self, G):
        if not self.in_graph(G):
            return None
        return G.edges[self.u, self.v, self.rel_type]


@dataclass
class CandidateScope:
    """A higher-arity virtual candidate.

    `params` carries the semantic extras that change the constraint's meaning
    (e.g. the parent variable for parent→group). `meta` is for diagnostics
    only and does not contribute to identity.
    """

    type_key: str
    scope: Tuple[str, ...]  # variable NAMES (canonical sorted order)
    params: dict = field(default_factory=dict)
    alive: bool = True
    meta: dict = field(default_factory=dict)

    def canonical_id(self) -> Tuple:
        return (self.type_key, tuple(sorted(self.scope)),
                tuple(sorted(self.params.items())))


class RelationTemplate(ABC):
    """ABC every higher-arity relation in the bias implements.

    A template knows how to:
      - detect candidate scopes from the pairwise graph (pattern matching),
      - materialize a candidate to a CPMpy expression (so PyCONA can use it),
      - report which pairwise EdgeKeys "support" a candidate (cascade kill).
    """

    type_key: str
    min_arity: int = 3

    @abstractmethod
    def detect_scopes(
        self,
        G,
        nodes: Optional[Iterable[str]] = None,
        scope_size: Optional[int] = None,
    ) -> Iterable[CandidateScope]:
        ...

    @abstractmethod
    def materialize(self, scope: CandidateScope, name_to_var: dict):
        """Return the CPMpy expression for this candidate."""
        ...

    @abstractmethod
    def supporting_edges(self, scope: CandidateScope) -> List[EdgeKey]:
        ...

    def evaluate(self, scope: CandidateScope, name_to_var: dict) -> str:
        """Direct-Python violation check against current ``.value()``s.

        Returns one of ViolationStatus values. The default fallback builds
        the CPMpy expression and runs ``check_value``; concrete templates
        SHOULD override with a cheap direct check, since this lives in the
        FindScope hot path (``same_kappa`` /
        ``has_any_violated_higher_arity``) and avoids constructing
        potentially huge CPMpy sums.
        """
        expr = self.materialize(scope, name_to_var)
        v = check_value(expr)
        if v is False:
            return ViolationStatus.VIOLATED
        if v is True:
            return ViolationStatus.SATISFIED
        return ViolationStatus.UNKNOWN

    def detect_violated(
        self,
        G,
        name_to_var: dict,
        Y_names: Optional[FrozenSet[str]] = None,
    ) -> Iterable[CandidateScope]:
        """Yield candidates that are CURRENTLY VIOLATED under .value()s.

        Default fallback: ``detect_scopes`` then filter by ``evaluate``.
        Concrete templates SHOULD override with a value-aware enumeration
        that skips structurally-fine but currently-satisfied patterns.
        ``same_kappa`` and ``has_any_violated_higher_arity`` use this.
        """
        for cand in self.detect_scopes(G):
            if Y_names is not None and not set(cand.scope).issubset(Y_names):
                continue
            parent = cand.params.get("parent")
            if parent is not None and Y_names is not None and parent not in Y_names:
                continue
            if self.evaluate(cand, name_to_var) == ViolationStatus.VIOLATED:
                yield cand

    def detect_scopes_anchored(
        self, G, anchor: str
    ) -> Iterable[CandidateScope]:
        """Yield STRUCTURAL candidates whose full scope (including any
        params-borne parent) includes ``anchor``.

        Default fallback filters ``detect_scopes(G)`` by membership;
        concrete templates SHOULD override with anchor-bounded
        enumeration. Used as a building block by ``detect_violated_anchored``.
        """
        for cand in self.detect_scopes(G):
            if anchor in cand.scope:
                yield cand
                continue
            parent = cand.params.get("parent")
            if parent == anchor:
                yield cand

    def detect_violated_anchored(
        self, G, name_to_var: dict, anchor: str
    ) -> Iterable[CandidateScope]:
        """Yield candidates that include ``anchor`` AND are currently
        violated under variable ``.value()``s.

        This is the primitive ``Bias.kappa_delta`` actually consumes:
        existence-style enumeration that bypasses the (potentially
        exponential) structural enumeration when the value-side prunes
        most candidates anyway. The default fallback wraps
        ``detect_scopes_anchored`` + ``evaluate``; templates with dense
        structural patterns SHOULD override with value-aware enumeration.
        """
        for cand in self.detect_scopes_anchored(G, anchor):
            if self.evaluate(cand, name_to_var) == ViolationStatus.VIOLATED:
                yield cand


class _TemplateIndex:
    """Per-template virtual-candidate cache + reverse support index.

    Cache invalidation is coarse: on any edge mutation we mark the cache
    stale; the next refresh re-runs detect_scopes and reconciles by
    canonical_id, preserving meta on survivors.
    """

    def __init__(self, template: RelationTemplate, graph):
        self.template = template
        self.G = graph
        # canonical_id -> CandidateScope
        self._candidates: dict = {}
        # canonical_id -> List[EdgeKey]
        self._support: dict = {}
        # EdgeKey -> set of canonical_ids relying on it
        self._edge_to_cids: dict = {}
        self._stale = True

    def mark_stale(self) -> None:
        self._stale = True

    def kill_dependent(self, edge: EdgeKey) -> List[CandidateScope]:
        killed: List[CandidateScope] = []
        for cid in list(self._edge_to_cids.get(edge, ())):
            cand = self._candidates.get(cid)
            if cand is not None and cand.alive:
                cand.alive = False
                killed.append(cand)
        return killed

    def refresh(self) -> None:
        if not self._stale:
            return
        seen_ids: Set[Tuple] = set()
        for scope in self.template.detect_scopes(self.G):
            cid = scope.canonical_id()
            seen_ids.add(cid)
            if cid in self._candidates:
                existing = self._candidates[cid]
                if not existing.alive:
                    existing.alive = True
                continue
            self._candidates[cid] = scope
            edges = self.template.supporting_edges(scope)
            self._support[cid] = edges
            for e in edges:
                self._edge_to_cids.setdefault(e, set()).add(cid)
        # Anything no longer detected: mark dead (don't drop -- keep history).
        for cid, cand in self._candidates.items():
            if cid not in seen_ids and cand.alive:
                cand.alive = False
        self._stale = False

    def alive_in_scope(
        self, Y_names: Set[str], scope_size: Optional[int] = None
    ) -> Iterator[CandidateScope]:
        self.refresh()
        for cand in self._candidates.values():
            if not cand.alive:
                continue
            if scope_size is not None and len(cand.scope) != scope_size:
                continue
            # full scope (children + any params-borne parent) must be ⊆ Y
            if not set(cand.scope).issubset(Y_names):
                continue
            parent = cand.params.get("parent")
            if parent is not None and parent not in Y_names:
                continue
            yield cand

    def num_alive(self) -> int:
        self.refresh()
        return sum(1 for c in self._candidates.values() if c.alive)

    def copy(self, new_graph) -> "_TemplateIndex":
        new = _TemplateIndex(self.template, new_graph)
        # Preserve cached candidates / reverse index but force a re-validate
        # against the new graph on next refresh.
        for cid, cand in self._candidates.items():
            new._candidates[cid] = CandidateScope(
                type_key=cand.type_key,
                scope=cand.scope,
                params=dict(cand.params),
                alive=cand.alive,
                meta=dict(cand.meta),
            )
        for cid, edges in self._support.items():
            new._support[cid] = list(edges)
        for e, cids in self._edge_to_cids.items():
            new._edge_to_cids[e] = set(cids)
        new._stale = True
        return new


# ---------------------------------------------------------------------------
# Concrete templates
# ---------------------------------------------------------------------------


class AlternativeGroupTemplate(RelationTemplate):
    """parent -> sum(children) == 1.

    Pattern: clique on `implies_not` between children + every child has an
    `implies` edge to a common parent. (This is the ALT-group structure
    encoded by the existing `find_cliques_with_common_star` helper.)
    """

    type_key = "alternative_group"
    min_arity = 3  # parent + at least 2 children

    def detect_scopes(
        self,
        G,
        nodes: Optional[Iterable[str]] = None,
        scope_size: Optional[int] = None,
    ) -> Iterable[CandidateScope]:
        # Build the clique-side graph (implies_not) and the star-side index
        # in one pass over edges. Restricting to alive (non-learned) edges
        # would prune more, but matches existing behaviour to keep parity.
        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(G.nodes(data=True))
        star_targets: dict = {}
        for u, v, k in G.edges(keys=True):
            if k == "implies_not":
                clique_graph.add_edge(u, v)
            elif k == "implies":
                star_targets.setdefault(u, set()).add(v)

        iterator = nx.enumerate_all_cliques(clique_graph)
        if nodes is not None:
            node_set = set(nodes)
            iterator = (c for c in iterator if any(n in node_set for n in c))

        for clique in iterator:
            if len(clique) < 2:  # need at least 2 children
                continue
            if scope_size is not None and len(clique) != scope_size - 1:
                continue
            clique_set = set(clique)
            common_targets: Optional[Set[str]] = None
            for node in clique:
                targets = star_targets.get(node, set()) - clique_set
                if common_targets is None:
                    common_targets = set(targets)
                else:
                    common_targets &= targets
                if not common_targets:
                    break
            if not common_targets:
                continue
            scope_tuple = tuple(sorted(clique))
            for parent in common_targets:
                yield CandidateScope(
                    type_key=self.type_key,
                    scope=scope_tuple,
                    params={"parent": parent},
                    meta={"variant": "alt-clique"},
                )

    def materialize(self, scope: CandidateScope, name_to_var: dict):
        parent = scope.params["parent"]
        cvar = name_to_var[parent]
        children = [name_to_var[n] for n in scope.scope]
        return cvar.implies(cp.sum(children) == 1)

    def evaluate(self, scope: CandidateScope, name_to_var: dict) -> str:
        # parent.implies(sum(children) == 1)
        # violated iff parent is True and sum(children) != 1
        pval = name_to_var[scope.params["parent"]].value()
        if pval is None:
            return ViolationStatus.UNKNOWN
        if pval is False:
            return ViolationStatus.SATISFIED
        ones = 0
        for n in scope.scope:
            cv = name_to_var[n].value()
            if cv is None:
                return ViolationStatus.UNKNOWN
            if cv:
                ones += 1
                if ones > 1:
                    return ViolationStatus.VIOLATED
        return ViolationStatus.SATISFIED if ones == 1 else ViolationStatus.VIOLATED

    def detect_violated(
        self,
        G,
        name_to_var: dict,
        Y_names: Optional[FrozenSet[str]] = None,
    ) -> Iterable[CandidateScope]:
        """Cheap value-aware skip then default fallback.

        An alt-group `parent → sum(children) == 1` can only be violated
        if its parent is currently True. If no node in scope is True at
        all, no alt-group candidate can be violated; we skip the (heavy)
        clique enumeration entirely.
        """
        if Y_names is None:
            iter_nodes = G.nodes()
        else:
            iter_nodes = Y_names
        any_true = any(name_to_var[n].value() is True for n in iter_nodes
                       if n in name_to_var)
        if not any_true:
            return
        yield from super().detect_violated(G, name_to_var, Y_names)

    def detect_violated_anchored(
        self, G, name_to_var: dict, anchor: str
    ) -> Iterable[CandidateScope]:
        """Value-aware: only yield currently-violated alt-group
        candidates that include ``anchor``.

        An alt-group is violated iff parent True AND `sum(children true)
        != 1`. This narrows the structural search dramatically:

          (A) Anchor as parent: requires anchor True.  Then look at the
              anchor's incoming-implies set — if ≥2 of those children
              are True and at least one pair is connected by
              `implies_not`, that pair forms a minimal violated scope.
              If 0 are True, any pair of False children with
              `implies_not` is violated.
          (B) Anchor in scope: requires SOME parent V (reachable via
              `anchor implies V`) to be True. For each such V, apply
              case (A)'s logic restricted to V's incoming-implies set
              that contains anchor.

        We yield the SMALLEST violating candidate per (parent, pair) and
        rely on caller short-circuit; redundant supersets aren't
        enumerated.
        """
        anchor_val = name_to_var[anchor].value()

        def _yield_pair_for_parent(parent: str, kids_set: set):
            if not kids_set or len(kids_set) < 2:
                return
            true_kids = []
            false_kids = []
            for k in kids_set:
                v = name_to_var[k].value()
                if v is True:
                    true_kids.append(k)
                elif v is False:
                    false_kids.append(k)
            # Case ≥2-True
            if len(true_kids) >= 2:
                for i in range(len(true_kids)):
                    a = true_kids[i]
                    for b in true_kids[i + 1 :]:
                        if (G.has_edge(a, b, key="implies_not")
                                or G.has_edge(b, a, key="implies_not")):
                            yield CandidateScope(
                                type_key=self.type_key,
                                scope=tuple(sorted([a, b])),
                                params={"parent": parent},
                                meta={"variant": "alt-violated-2true"},
                            )
            # Case 0-True (every kid False)
            if not true_kids and len(false_kids) >= 2:
                for i in range(len(false_kids)):
                    a = false_kids[i]
                    for b in false_kids[i + 1 :]:
                        if (G.has_edge(a, b, key="implies_not")
                                or G.has_edge(b, a, key="implies_not")):
                            yield CandidateScope(
                                type_key=self.type_key,
                                scope=tuple(sorted([a, b])),
                                params={"parent": parent},
                                meta={"variant": "alt-violated-0true"},
                            )

        # Case (A): anchor as parent.
        if anchor_val is True:
            kids: set = set()
            for u, _, k in G.in_edges(anchor, keys=True):
                if k == "implies":
                    kids.add(u)
            kids.discard(anchor)
            yield from _yield_pair_for_parent(anchor, kids)

        # Case (B): anchor as child.  Iterate parents V via implies, V True.
        seen_parents: set = set()
        for _, v, k in G.edges(anchor, keys=True):
            if k != "implies":
                continue
            if v == anchor or v in seen_parents:
                continue
            seen_parents.add(v)
            if name_to_var[v].value() is not True:
                continue
            sibs: set = set()
            for u, _, k2 in G.in_edges(v, keys=True):
                if k2 == "implies":
                    sibs.add(u)
            sibs.discard(v)
            if anchor not in sibs:
                continue
            yield from _yield_pair_for_parent(v, sibs)

    def detect_scopes_anchored(
        self, G, anchor: str
    ) -> Iterable[CandidateScope]:
        """Cliques on `implies_not` containing ``anchor`` + a common
        `implies` parent reached by every clique member.

        Implementation: restrict to `anchor` + its `implies_not`
        neighbours, then enumerate cliques in that induced subgraph.
        Cost is bounded by the size of the anchor's neighbourhood, not
        the full graph. Yields candidates anchored either as a clique
        member OR as the parent.
        """
        # Build clique-side adjacency restricted to anchor's implies_not nbrs.
        nbrs: set = set()
        for u, v, k in G.edges(anchor, keys=True):
            if k == "implies_not":
                nbrs.add(v)
        for u, v, k in G.in_edges(anchor, keys=True):
            if k == "implies_not":
                nbrs.add(u)

        if nbrs:
            sub = nx.Graph()
            sub.add_node(anchor)
            for n in nbrs:
                sub.add_node(n)
            anchor_set = nbrs | {anchor}
            for u, v, k in G.edges(anchor_set, keys=True):
                if k == "implies_not" and v in anchor_set:
                    sub.add_edge(u, v)
            for u, v, k in G.in_edges(anchor_set, keys=True):
                if k == "implies_not" and u in anchor_set:
                    sub.add_edge(u, v)

            # Star-side index restricted to nodes that may show up in cliques.
            star_targets: dict = {}
            for u, v, k in G.edges(anchor_set, keys=True):
                if k == "implies":
                    star_targets.setdefault(u, set()).add(v)

            for clique in nx.enumerate_all_cliques(sub):
                if anchor not in clique:
                    continue
                if len(clique) < 2:
                    continue
                clique_set = set(clique)
                common: Optional[set] = None
                for node in clique:
                    targets = star_targets.get(node, set()) - clique_set
                    if common is None:
                        common = set(targets)
                    else:
                        common &= targets
                    if not common:
                        break
                if not common:
                    continue
                scope_tuple = tuple(sorted(clique))
                for parent in common:
                    yield CandidateScope(
                        type_key=self.type_key,
                        scope=scope_tuple,
                        params={"parent": parent},
                        meta={"variant": "alt-anchored"},
                    )

        # Anchor as parent: cliques among its incoming-implies set.
        kids: set = set()
        for u, _, k in G.in_edges(anchor, keys=True):
            if k == "implies":
                kids.add(u)
        kids.discard(anchor)
        if len(kids) >= 2:
            sub2 = nx.Graph()
            for n in kids:
                sub2.add_node(n)
            for u, v, k in G.edges(kids, keys=True):
                if k == "implies_not" and v in kids:
                    sub2.add_edge(u, v)
            for clique in nx.enumerate_all_cliques(sub2):
                if len(clique) < 2:
                    continue
                # Every clique member must `implies` the anchor (already
                # ensured by membership in `kids`); no extra star check
                # needed because `anchor` is the chosen parent.
                yield CandidateScope(
                    type_key=self.type_key,
                    scope=tuple(sorted(clique)),
                    params={"parent": anchor},
                    meta={"variant": "alt-anchored-parent"},
                )

    def supporting_edges(self, scope: CandidateScope) -> List[EdgeKey]:
        edges: List[EdgeKey] = []
        verts = list(scope.scope)
        # Mutual-exclusion clique (implies_not, both directions stored).
        for i in range(len(verts)):
            for j in range(len(verts)):
                if i == j:
                    continue
                edges.append(EdgeKey(verts[i], verts[j], "implies_not"))
        # Children -> parent (the "common star").
        parent = scope.params["parent"]
        for v in verts:
            edges.append(EdgeKey(v, parent, "implies"))
        return edges


class OrGroupTemplate(RelationTemplate):
    """parent -> sum(children) >= 1.

    Pattern: every child has an `implies` edge to the parent. No clique
    structure required (children may be independently selectable).

    Combinatorial blow-up is bounded by `scope_size`: detect_scopes only
    enumerates sub-bunches of the requested arity. When called without
    scope_size, each parent yields one MAXIMAL candidate (full child set).
    """

    type_key = "or_group"
    min_arity = 3  # parent + at least 2 children

    def detect_scopes(
        self,
        G,
        nodes: Optional[Iterable[str]] = None,
        scope_size: Optional[int] = None,
    ) -> Iterable[CandidateScope]:
        # parent -> set of children that imply it
        children_of: dict = {}
        for u, v, k in G.edges(keys=True):
            if k == "implies":
                children_of.setdefault(v, set()).add(u)

        node_set = set(nodes) if nodes is not None else None
        for parent, kids in children_of.items():
            if node_set is not None and parent not in node_set:
                # also accept parents reached through node_set membership
                if not (kids & node_set):
                    continue
            kids = kids - {parent}
            if len(kids) < 2:
                continue
            if scope_size is None:
                # Maximal candidate per parent.
                scope_tuple = tuple(sorted(kids))
                yield CandidateScope(
                    type_key=self.type_key,
                    scope=scope_tuple,
                    params={"parent": parent},
                    meta={"variant": "or-maximal"},
                )
            else:
                target_arity = scope_size - 1
                if target_arity < 2 or target_arity > len(kids):
                    continue
                # Enumerate sub-bunches of the right size.
                for combo in itertools.combinations(sorted(kids), target_arity):
                    yield CandidateScope(
                        type_key=self.type_key,
                        scope=tuple(combo),
                        params={"parent": parent},
                        meta={"variant": f"or-{target_arity}"},
                    )

    def materialize(self, scope: CandidateScope, name_to_var: dict):
        parent = scope.params["parent"]
        cvar = name_to_var[parent]
        children = [name_to_var[n] for n in scope.scope]
        return cvar.implies(cp.sum(children) >= 1)

    def evaluate(self, scope: CandidateScope, name_to_var: dict) -> str:
        # parent.implies(sum(children) >= 1) — violated iff parent True
        # and every child False.
        pval = name_to_var[scope.params["parent"]].value()
        if pval is None:
            return ViolationStatus.UNKNOWN
        if pval is False:
            return ViolationStatus.SATISFIED
        any_unknown = False
        for n in scope.scope:
            cv = name_to_var[n].value()
            if cv is None:
                any_unknown = True
                continue
            if cv:
                return ViolationStatus.SATISFIED
        return ViolationStatus.UNKNOWN if any_unknown else ViolationStatus.VIOLATED

    def detect_violated(
        self,
        G,
        name_to_var: dict,
        Y_names: Optional[FrozenSet[str]] = None,
    ) -> Iterable[CandidateScope]:
        """Value-aware: a maximal OR-group `parent -> any(children)` is
        violated iff the parent is currently True AND every child currently
        False. We only consider parents that are True and whose
        incoming-implies neighbours are *all* False.
        """
        # parent -> set of children that imply it (restricted to Y if given)
        children_of: dict = {}
        for u, v, k in G.edges(keys=True):
            if k != "implies":
                continue
            if Y_names is not None and (u not in Y_names or v not in Y_names):
                continue
            children_of.setdefault(v, set()).add(u)

        for parent, kids in children_of.items():
            kids = kids - {parent}
            if len(kids) < 2:
                continue
            pvar = name_to_var.get(parent)
            if pvar is None or pvar.value() is not True:
                continue
            # Cheap False-only check, short-circuit on first True child.
            any_unknown = False
            all_false = True
            for c in kids:
                cv = name_to_var[c].value()
                if cv is True:
                    all_false = False
                    break
                if cv is None:
                    any_unknown = True
            if not all_false or any_unknown:
                continue
            yield CandidateScope(
                type_key=self.type_key,
                scope=tuple(sorted(kids)),
                params={"parent": parent},
                meta={"variant": "or-violated"},
            )

    def detect_violated_anchored(
        self, G, name_to_var: dict, anchor: str
    ) -> Iterable[CandidateScope]:
        """Value-aware: only yield currently-violated OR-group candidates
        that include ``anchor``.

        OR-group `parent → any(children)` is violated iff parent True AND
        every child False. So:
          (A) anchor as parent: requires anchor True and every
              incoming-implies child False (with ≥2 such children).
          (B) anchor as child: requires anchor False and the candidate
              parent V (anchor implies V) to be True with every other
              child of V also False.
        Cost: O(degree(anchor)).
        """
        anchor_val = name_to_var[anchor].value()

        def _violated_at(parent: str, kids: set):
            if len(kids) < 2:
                return None
            for k in kids:
                v = name_to_var[k].value()
                if v is None or v is True:
                    return None
            return CandidateScope(
                type_key=self.type_key,
                scope=tuple(sorted(kids)),
                params={"parent": parent},
                meta={"variant": "or-violated-anchored"},
            )

        # (A) anchor as parent
        if anchor_val is True:
            kids: set = set()
            for u, _, k in G.in_edges(anchor, keys=True):
                if k == "implies":
                    kids.add(u)
            kids.discard(anchor)
            cand = _violated_at(anchor, kids)
            if cand is not None:
                yield cand

        # (B) anchor as child (must be False)
        if anchor_val is False:
            seen_parents: set = set()
            for _, v, k in G.edges(anchor, keys=True):
                if k != "implies":
                    continue
                if v == anchor or v in seen_parents:
                    continue
                seen_parents.add(v)
                if name_to_var[v].value() is not True:
                    continue
                sibs: set = set()
                for u, _, k2 in G.in_edges(v, keys=True):
                    if k2 == "implies":
                        sibs.add(u)
                sibs.discard(v)
                if anchor not in sibs:
                    continue
                cand = _violated_at(v, sibs)
                if cand is not None:
                    yield cand

    def detect_scopes_anchored(
        self, G, anchor: str
    ) -> Iterable[CandidateScope]:
        """Two cases:
          - anchor as parent: incoming-implies set (children) of size >= 2.
          - anchor as child: each parent v reachable via `anchor implies v`
            yields the OR-group at v (incoming-implies set of v including
            anchor).
        Cost: O(degree(anchor)).
        """
        # Case A: anchor as parent.
        kids_at_anchor: set = set()
        for u, _, k in G.in_edges(anchor, keys=True):
            if k == "implies":
                kids_at_anchor.add(u)
        kids_at_anchor.discard(anchor)
        if len(kids_at_anchor) >= 2:
            yield CandidateScope(
                type_key=self.type_key,
                scope=tuple(sorted(kids_at_anchor)),
                params={"parent": anchor},
                meta={"variant": "or-anchored-parent"},
            )

        # Case B: anchor as child. Each parent v with anchor->v is a
        # candidate parent; emit its full incoming-implies group.
        seen_parents: set = set()
        for _, v, k in G.edges(anchor, keys=True):
            if k != "implies":
                continue
            if v == anchor or v in seen_parents:
                continue
            seen_parents.add(v)
            siblings: set = set()
            for u, _, k2 in G.in_edges(v, keys=True):
                if k2 == "implies":
                    siblings.add(u)
            siblings.discard(v)
            if len(siblings) >= 2 and anchor in siblings:
                yield CandidateScope(
                    type_key=self.type_key,
                    scope=tuple(sorted(siblings)),
                    params={"parent": v},
                    meta={"variant": "or-anchored-child"},
                )

    def supporting_edges(self, scope: CandidateScope) -> List[EdgeKey]:
        parent = scope.params["parent"]
        return [EdgeKey(c, parent, "implies") for c in scope.scope]


class GraphBias:
    """Layered pairwise + higher-arity bias.

    Pairwise edges live in ``self.G`` (an ``nx.MultiDiGraph``) with
    ``learned: bool`` per edge and standard KEYMAP relation keys. Higher-arity
    candidates live in per-template indices (``self._templates``) — see the
    RelationTemplate docstrings above.

    Public surface (do NOT change):
        len(bias), bias.initial_size, bias.G, bias.removed,
        bias.get_kappa(Y, extended=False), bias.same_kappa(Y1, Y2),
        bias.mark_as_learned(C), bias.remove_from_bias(C), bias.copy()
    """

    def __init__(self, bias, verbose=3):
        self.G = self._build_graph_from_bias(bias)
        self.verbose = verbose
        self.initial_size = self.__len__()
        # `removed` is kept populated for backward compat (downstream readers).
        # The new template indices are now the primary cascade mechanism.
        self.removed = set()
        self._templates: dict = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        for tmpl in (AlternativeGroupTemplate(), OrGroupTemplate()):
            self._templates[tmpl.type_key] = _TemplateIndex(tmpl, self.G)

    def _build_graph_from_bias(self, bias):
        G = nx.MultiDiGraph()

        for c in bias:
            assert len(c.args) == 2 and all(
                isinstance(arg, (NegBoolView, _BoolVarImpl)) for arg in c.args
            ), "GraphBias can only be constructed from pairwise constraints"

            expr_name = c.name
            a0, a1 = c.args
            a0_name = a0.name
            a1_name = a1.name

            if isinstance(a0, NegBoolView):
                a0_name = a0_name[1:]
                expr_name = "!" + expr_name
                a0 = a0._bv

            if isinstance(a1, NegBoolView):
                expr_name += "!"
                a1_name = a1_name[1:]  # remove the leading "~" from the variable name
                a1 = a1._bv

            G.add_node(a0_name, var=a0)
            G.add_node(a1_name, var=a1)

            G.add_edge(a0_name, a1_name, key=KEYMAP[expr_name], expr=c, learned=False)

        return G

    # ---------------- name <-> CPMpy var bridge ----------------

    def _name_to_var(self) -> dict:
        return {n: d["var"] for n, d in self.G.nodes(data=True)}

    # ---------------- edge identification ----------------

    def _pairwise_edge_key(self, c) -> Optional[EdgeKey]:
        """Return the EdgeKey for a pairwise CPMpy constraint, or None."""
        a0, a1 = c.args
        if not isinstance(a1, (NegBoolView, _BoolVarImpl)):
            return None
        expr_name = c.name
        a0_name = a0.name
        a1_name = a1.name
        if isinstance(a0, NegBoolView):
            expr_name = "!" + expr_name
            a0_name = a0_name[1:]
        if isinstance(a1, NegBoolView):
            expr_name += "!"
            a1_name = a1_name[1:]
        return EdgeKey(a0_name, a1_name, KEYMAP[expr_name])

    def _identify_higher_arity(self, c) -> Optional[Tuple[str, Tuple]]:
        """Pattern-match a higher-arity expression back to (type_key, canonical_id).

        Recognizes the materialization shapes produced by the registered
        templates. Returns None if the constraint isn't one we materialized.
        """
        try:
            if c.name != "->":
                return None
            parent_var = c.args[0]
            if not isinstance(parent_var, _BoolVarImpl) or isinstance(
                parent_var, NegBoolView
            ):
                return None
            body = c.args[1]
            body_name = getattr(body, "name", None)
            if body_name not in ("==", ">="):
                return None
            sum_expr, k = body.args
            if not hasattr(sum_expr, "args"):
                return None
            child_vars = list(sum_expr.args)
            if not all(
                isinstance(v, _BoolVarImpl) and not isinstance(v, NegBoolView)
                for v in child_vars
            ):
                return None
            if int(k) != 1:
                return None
            type_key = (
                "alternative_group" if body_name == "==" else "or_group"
            )
            scope = tuple(sorted(v.name for v in child_vars))
            params = (("parent", parent_var.name),)
            cid = (type_key, scope, params)
            return type_key, cid
        except Exception:
            return None

    # ---------------- mutation cascade ----------------

    def _mark_all_stale(self) -> None:
        for idx in self._templates.values():
            idx.mark_stale()

    def _cascade_kill(self, edge_key: EdgeKey) -> None:
        """Notify every template index that ``edge_key`` is gone, then bump
        the staleness flag so detect_scopes re-runs on next refresh."""
        for idx in self._templates.values():
            idx.kill_dependent(edge_key)
        self._mark_all_stale()

    # ---------------- standard interface ----------------

    def __len__(self):
        return sum(1 for _, _, e in self.G.edges(data=True) if not e["learned"])

    def kappa_delta(self, small_Y, large_Y):
        """Iterate constraints in ``kappa(large) \\ kappa(small)``.

        REQUIRES ``small_Y ⊆ large_Y``. Iteration is bounded by
        ``|large \\ small|`` because every yielded constraint must touch
        a delta node (otherwise it would already be in ``kappa(small)``).

        Yields CPMpy expressions for both pairwise and higher-arity
        violators. Caller short-circuits with ``next(it, None)`` for an
        existence check (the canonical use site is ``same_kappa``).
        """
        small_names = frozenset(v.name for v in small_Y)
        large_names = frozenset(v.name for v in large_Y)
        delta_names = large_names - small_names
        if not delta_names:
            return

        # ---- Pairwise: edges with at least one endpoint in delta and
        # the other endpoint in large_names. Walk in/out edges of delta
        # nodes; dedupe across multi-key edges.
        seen_edges: set = set()
        for n in delta_names:
            for u, v, k, e in self.G.edges(n, keys=True, data=True):
                if v not in large_names:
                    continue
                if e["learned"]:
                    continue
                eid = (u, v, k)
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                if check_value(e["expr"]) is False:
                    yield e["expr"]
            for u, v, k, e in self.G.in_edges(n, keys=True, data=True):
                if u not in large_names:
                    continue
                if e["learned"]:
                    continue
                eid = (u, v, k)
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                if check_value(e["expr"]) is False:
                    yield e["expr"]

        # ---- Higher-arity: value-aware anchored detection per template
        # per delta node. A candidate is in delta-kappa iff its full
        # scope ⊆ large AND touches a delta node (otherwise it's already
        # in kappa(small)). detect_violated_anchored bakes the violation
        # filter in, so we don't pay for structurally-fine candidates.
        n2v = self._name_to_var()
        seen_cids: set = set()
        for idx in self._templates.values():
            tmpl = idx.template
            for n in delta_names:
                for cand in tmpl.detect_violated_anchored(self.G, n2v, n):
                    cid = cand.canonical_id()
                    if cid in seen_cids:
                        continue
                    seen_cids.add(cid)
                    if not set(cand.scope).issubset(large_names):
                        continue
                    parent = cand.params.get("parent")
                    if parent is not None and parent not in large_names:
                        continue
                    tracked = idx._candidates.get(cid)
                    if tracked is not None and not tracked.alive:
                        continue
                    yield tmpl.materialize(cand, n2v)

    def same_kappa(self, Y1: set, Y2):
        """Whether ``kappa(Y1) == kappa(Y2)``.

        When ``Y1 ⊆ Y2`` (or vice versa) — which is always the case
        coming from FindScope — the work is bounded by the set
        difference: we existence-check ``kappa_delta(small, large)`` and
        return on the first hit. No need to materialize either kappa set.

        For the rare unrelated case (callers other than FindScope), we
        fall back to the scope-local two-sided comparison.
        """
        if Y1 == Y2:
            return True
        s1 = set(Y1)
        s2 = set(Y2)
        if s1.issubset(s2):
            small, large = s1, s2
        elif s2.issubset(s1):
            small, large = s2, s1
        else:
            return self._same_kappa_general(s1, s2)
        return next(iter(self.kappa_delta(small, large)), None) is None

    def _same_kappa_general(self, Y1, Y2) -> bool:
        """Fallback when Y1 / Y2 are not subset-related.

        Pairwise count check then scope-local virtual comparison. Kept
        for correctness; FindScope2 doesn't actually hit this path.
        """
        kappaY1 = self.get_kappa(Y1)
        kappaY2 = self.get_kappa(Y2)
        if len(kappaY1) != len(kappaY2):
            return False
        Y1_names = frozenset(v.name for v in Y1)
        Y2_names = frozenset(v.name for v in Y2)
        sg1 = self.G.subgraph(Y1_names)
        sg2 = self.G.subgraph(Y2_names)
        n2v = self._name_to_var()
        for idx in self._templates.values():
            tmpl = idx.template
            v1 = self._violated_cids_local(tmpl, sg1, Y1_names, n2v, idx)
            v2 = self._violated_cids_local(tmpl, sg2, Y2_names, n2v, idx)
            if v1 != v2:
                return False
        return True

    @staticmethod
    def _violated_cids_local(tmpl, subgraph, Y_names, n2v, idx) -> set:
        out: set = set()
        for cand in tmpl.detect_scopes(subgraph):
            if not set(cand.scope).issubset(Y_names):
                continue
            parent = cand.params.get("parent")
            if parent is not None and parent not in Y_names:
                continue
            cid = cand.canonical_id()
            tracked = idx._candidates.get(cid)
            if tracked is not None and not tracked.alive:
                continue
            if tmpl.evaluate(cand, n2v) == ViolationStatus.VIOLATED:
                out.add(cid)
        return out

    def has_any_violated_higher_arity(self, Y) -> bool:
        """Short-circuit existence: True iff any alive higher-arity
        candidate whose full scope sits inside Y is currently violated.
        """
        return next(self.iter_violated_higher_arity_in(Y), None) is not None

    def iter_violated_higher_arity_in(self, Y):
        """Iterator yielding ``(template, CandidateScope)`` for every
        currently-violated higher-arity candidate whose full scope
        (including any params-borne parent) sits inside Y.

        Used by FindScope's template-guided shortcut so the candidate
        identity propagates back to the learner — no FindC round-trip
        needed when the bias already pinpoints the violator.
        """
        Y_names = frozenset(v.name for v in Y)
        if not Y_names:
            return
        n2v = self._name_to_var()
        seen_cids: set = set()
        for idx in self._templates.values():
            tmpl = idx.template
            for n in Y_names:
                for cand in tmpl.detect_violated_anchored(self.G, n2v, n):
                    cid = cand.canonical_id()
                    if cid in seen_cids:
                        continue
                    seen_cids.add(cid)
                    if not set(cand.scope).issubset(Y_names):
                        continue
                    parent = cand.params.get("parent")
                    if parent is not None and parent not in Y_names:
                        continue
                    tracked = idx._candidates.get(cid)
                    if tracked is not None and not tracked.alive:
                        continue
                    yield tmpl, cand

    def get_kappa(self, Y, extended=False):
        """Constraints whose scope is a subset of Y and are violated."""
        Y_names = frozenset(v.name for v in Y)
        subgraph = self.G.subgraph(Y_names)

        kappa = [
            e["expr"]
            for _, _, e in subgraph.edges(data=True, keys=False)
            if check_value(e["expr"]) is False and not e["learned"]
        ]

        if extended:
            n2v = self._name_to_var()
            for idx in self._templates.values():
                for cand in idx.alive_in_scope(Y_names):
                    expr = idx.template.materialize(cand, n2v)
                    if check_value(expr) is False:
                        kappa.append(expr)
        return kappa

    def mark_as_learned(self, C):
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), (
            "mark_as_learned accepts as input a list of constraints or a constraint"
        )

        if self.verbose >= 3:
            print(f"marking the following constraints as learned: {C}")

        for c in C:
            edge_key = self._pairwise_edge_key(c)
            if edge_key is None:
                # Higher-arity: mark candidate dead in its template index, AND
                # mark every supporting edge as learned (the constraint logically
                # entails them).
                self.removed.add(c)
                ident = self._identify_higher_arity(c)
                if ident is None:
                    continue
                type_key, cid = ident
                idx = self._templates.get(type_key)
                if idx is None:
                    continue
                idx.refresh()
                cand = idx._candidates.get(cid)
                if cand is not None:
                    cand.alive = False
                    for e in idx._support.get(cid, ()):
                        if e.in_graph(self.G):
                            self.G.edges[e.u, e.v, e.rel_type]["learned"] = True
                continue

            # Pairwise: mark learned in graph; downstream cascade lets templates
            # re-evaluate (detect_scopes typically still finds them since they
            # remain in G — we don't auto-kill virtual candidates on learning).
            if edge_key.in_graph(self.G):
                self.G.edges[edge_key.u, edge_key.v, edge_key.rel_type][
                    "learned"
                ] = True
            self._mark_all_stale()

    def remove_from_bias(self, C):
        """Remove given constraints from the bias (candidates)."""
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), (
            "remove_from_bias accepts as input a list of constraints or a constraint"
        )

        if self.verbose >= 3:
            print(f"removing the following constraints from bias: {C}")

        for c in C:
            edge_key = self._pairwise_edge_key(c)
            if edge_key is None:
                # Higher-arity: mark dead in template index. Backwards compat:
                # also keep the blocklist set populated.
                self.removed.add(c)
                ident = self._identify_higher_arity(c)
                if ident is not None:
                    type_key, cid = ident
                    idx = self._templates.get(type_key)
                    if idx is not None:
                        idx.refresh()
                        cand = idx._candidates.get(cid)
                        if cand is not None:
                            cand.alive = False
                continue

            # Pairwise: remove edge then cascade-kill virtual candidates
            # whose support included this edge.
            len_before = len(self.G.edges)
            try:
                self.G.remove_edge(edge_key.u, edge_key.v, key=edge_key.rel_type)
            except nx.NetworkXError:
                pass
            len_after = len(self.G.edges)
            if len_before == len_after:
                print(f"Warning: edge not found for constraint {c} - no edge removed")
                continue
            self._cascade_kill(edge_key)

    def copy(self):
        new_bias = object.__new__(GraphBias)
        new_bias.G = self.G.copy()
        new_bias.verbose = self.verbose
        new_bias.initial_size = self.initial_size
        new_bias.removed = self.removed.copy()
        new_bias._templates = {
            k: idx.copy(new_bias.G) for k, idx in self._templates.items()
        }
        return new_bias

    # ---------------- diagnostics ----------------

    def template_stats(self) -> dict:
        return {
            k: idx.num_alive() for k, idx in self._templates.items()
        }

    def export_html(self, filename="debug_bias.html", show=True):
        """Export the graph bias to an interactive HTML file for debugging.

        Edges are color-coded by constraint type and styled by learned status.
        Hover over edges to see the cpmpy expression.
        """
        from pyvis.network import Network

        EDGE_COLORS = {
            "implies": "#4285F4",        # blue
            "implies_not": "#EA4335",    # red
            "not_implies_not": "#FBBC04", # yellow
            "not_implies": "#9334E6",    # purple
            "equals": "#34A853",         # green
        }

        net = Network(
            height="95vh",
            width="100%",
            directed=True,
            notebook=False,
            cdn_resources="in_line",
        )
        net.barnes_hut(gravity=-3000, spring_length=150)
        net.set_options("""{
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
            },
            "physics": {
                "barnesHut": {"gravitationalConstant": -3000, "springLength": 150}
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "tooltipDelay": 50
            }
        }""")

        # Count unlearned edges per node for sizing
        degree = {}
        for u, v, data in self.G.edges(data=True):
            if not data["learned"]:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1

        for node in self.G.nodes():
            d = degree.get(node, 0)
            net.add_node(node, label=node, size=10 + min(d, 40), title=f"{node} ({d} unlearned edges)")

        # Group edges by undirected node pair to assign distinct curvatures
        from collections import defaultdict
        pair_counts = defaultdict(int)  # (min, max) -> count so far
        for u, v, key, data in self.G.edges(data=True, keys=True):
            pair_key = (min(u, v), max(u, v))
            pair_counts[pair_key] += 1
        # Total edges per pair, for centering the fan
        pair_totals = dict(pair_counts)
        pair_counts.clear()

        for u, v, key, data in self.G.edges(data=True, keys=True):
            pair_key = (min(u, v), max(u, v))
            idx = pair_counts.get(pair_key, 0)
            pair_counts[pair_key] = idx + 1
            total = pair_totals[pair_key]

            color = EDGE_COLORS.get(key, "#999999")
            learned = data.get("learned", False)
            expr_str = str(data.get("expr", ""))

            # Fan edges out: center around 0, step by 0.12
            if total == 1:
                smooth = {"type": "dynamic"}
            else:
                offset = (idx - (total - 1) / 2) * 0.12
                curve_type = "curvedCW" if offset >= 0 else "curvedCCW"
                smooth = {"type": curve_type, "roundness": abs(offset) + 0.05}

            style = {
                "color": {"color": color, "opacity": 0.3 if learned else 0.9},
                "dashes": learned,
                "width": 1 if learned else 2,
                "title": f"[{key}] {expr_str}" + (" (learned)" if learned else ""),
                "smooth": smooth,
            }
            net.add_edge(u, v, **style)

        # Inject a legend and filter controls into the HTML
        legend_html = "<div id='legend' style='position:fixed;top:10px;right:10px;background:white;padding:12px;border:1px solid #ccc;border-radius:6px;font-family:sans-serif;font-size:13px;z-index:999;'>"
        legend_html += "<b>Edge types</b> (click to toggle)<br>"
        for key, color in EDGE_COLORS.items():
            legend_html += (
                f"<label style='display:block;cursor:pointer;margin:2px 0;'>"
                f"<input type='checkbox' checked data-edge-type='{key}' onchange='toggleEdgeType(this)'> "
                f"<span style='color:{color};font-weight:bold;'>&#9644;</span> {key}</label>"
            )
        legend_html += "<hr style='margin:6px 0;'>"
        legend_html += "<label style='cursor:pointer;'><input type='checkbox' id='showLearned' checked onchange='toggleLearned(this)'> Show learned</label>"
        legend_html += f"<hr style='margin:6px 0;'><small>{len(self.G.nodes())} nodes, {self.G.number_of_edges()} edges ({len(self)} unlearned)</small>"
        legend_html += "</div>"

        filter_script = """
        <script>
        var allEdges = null;
        var network = null;
        // pyvis stores the network object; grab it after load
        document.addEventListener('DOMContentLoaded', function() {
            // pyvis exposes edges/nodes as DataSets
            setTimeout(function() {
                allEdges = edges;
                network = container.network || window.network;
            }, 500);
        });
        function toggleEdgeType(cb) {
            var edgeType = cb.dataset.edgeType;
            var show = cb.checked;
            var showLearned = document.getElementById('showLearned').checked;
            allEdges.forEach(function(e) {
                if (e.title && e.title.startsWith('[' + edgeType + ']')) {
                    var isLearned = e.title.indexOf('(learned)') !== -1;
                    var visible = show && (showLearned || !isLearned);
                    allEdges.update({id: e.id, hidden: !visible});
                }
            });
        }
        function toggleLearned(cb) {
            var showLearned = cb.checked;
            var checkboxes = document.querySelectorAll('[data-edge-type]');
            var enabledTypes = new Set();
            checkboxes.forEach(function(c) { if (c.checked) enabledTypes.add(c.dataset.edgeType); });
            allEdges.forEach(function(e) {
                if (e.title) {
                    var match = e.title.match(/^\\[([^\\]]+)\\]/);
                    if (match) {
                        var etype = match[1];
                        var isLearned = e.title.indexOf('(learned)') !== -1;
                        var visible = enabledTypes.has(etype) && (showLearned || !isLearned);
                        allEdges.update({id: e.id, hidden: !visible});
                    }
                }
            });
        }
        </script>
        """

        net.save_graph(filename)
        # Inject legend and script before </body>
        with open(filename, "r") as f:
            html = f.read()
        html = html.replace("</body>", legend_html + filter_script + "</body>")
        with open(filename, "w") as f:
            f.write(html)

        print(f"Graph bias exported to {filename}")
        if show:
            import webbrowser
            webbrowser.open(filename)


def get_con_subset_pairwise(GBias: GraphBias, Y):
    """Pairwise (binary) candidate constraints whose scope is a subset of Y."""
    Y_names = frozenset(v.name for v in Y)
    subgraph = GBias.G.subgraph(Y_names)
    cons = [
        e["expr"]
        for _, _, e in subgraph.edges(data=True, keys=False)
        if not e["learned"]
    ]
    print(f"get_con_subset_pairwise: base constraints: {len(cons)}")
    return cons


def get_con_subset(GBias: GraphBias, Y, scope_size=None):
    """Pairwise + higher-arity candidate constraints scoped to Y.

    Detection runs against ``G.subgraph(Y)`` directly so cost stays
    bounded by |Y|, not |V|. We do *not* go through the global
    ``_TemplateIndex.refresh`` here, because that would force a global
    re-enumeration of cliques on every call from FindC. The cascade-kill
    set carried by each template index is still consulted so candidates
    whose pairwise support has been eliminated are excluded.
    """
    Y_names = frozenset(v.name for v in Y)
    subgraph = GBias.G.subgraph(Y_names)
    cons = []

    if scope_size is None or scope_size == 2:
        cons.extend(
            e["expr"]
            for _, _, e in subgraph.edges(data=True, keys=False)
            if not e["learned"]
        )

    if scope_size is None or scope_size > 2:
        n2v = GBias._name_to_var()
        target_arity = None if scope_size is None else scope_size - 1
        for idx in GBias._templates.values():
            tmpl = idx.template
            for cand in tmpl.detect_scopes(subgraph, scope_size=scope_size):
                if target_arity is not None and len(cand.scope) != target_arity:
                    continue
                cid = cand.canonical_id()
                tracked = idx._candidates.get(cid)
                if tracked is not None and not tracked.alive:
                    continue
                expr = tmpl.materialize(cand, n2v)
                if expr in GBias.removed:
                    continue
                cons.append(expr)

    return cons


class GQuAcq(AlgorithmCAInteractive):
    """
    QuAcq is an implementation of the ICA_Algorithm that uses the QuAcq algorithm to learn constraints.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        """
        Initialize the QuAcq algorithm with an optional constraint acquisition environment.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        """
        super().__init__(ca_env)

    def learn(
        self,
        instance: ProblemInstance,
        oracle: Oracle = UserOracle(),
        verbose=0,
        X=None,
        metrics: Metrics = None,
    ):
        """
        Learn constraints using the QuAcq algorithm by generating queries and analyzing the results.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :param X: The set of variables to consider, default is None.
        :return: the learned instance
        """
        if X is None:
            X = instance.X
        assert isinstance(X, list), (
            "When using .learn(), set parameter X must be a list of variables. Instead got: {}".format(
                X
            )
        )
        assert set(X).issubset(set(instance.X)), (
            "When using .learn(), set parameter X must be a subset of the problem instance variables. Instead got: {}".format(
                X
            )
        )

        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(X)

        first_query = True

        while True:
            if self.env.verbose > 2:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B (pairwise): ", len(self.env.instance.bias))
                print("Number of Queries: ", self.env.metrics.membership_queries_count)

            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()

            if len(Y) == 0:
                # if no query can be generated it means we have (prematurely) converged to the target network -----
                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(
                        f"\nLearned {self.env.metrics.cl} constraints in "
                        f"{self.env.metrics.membership_queries_count} queries."
                    )
                return self.env.instance

            self.env.metrics.increase_generation_time(gen_end - gen_start)
            self.env.metrics.increase_generated_queries()
            self.env.metrics.increase_top_queries()

            def debug_query(q):
                vars = []
                varval = []

                for qi in q:
                    vn, vv = qi.split("=")
                    vars.append(next(v for v in Y if v.name == vn))
                    varval.append(vv == "True")

                restore_scope_values(vars, varval)
                return vars
                # resp = oracle.answer_membership_query(vars)
                # return resp

            # if first_query:
            #     q = ['Sandwich=True', 'Bread=True', 'Sauce=True', 'Ketchup=False', 'Mustard=False', 'Cheese=False']
            #     Y = debug_query(q)
            #     first_query = False

            # MODIFIED
            kappaB = self.env.instance.bias.get_kappa(Y)

            answer = self.env.ask_membership_query(Y)
            if answer:
                # it is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                # MODIFIED
                self.env.instance.bias.remove_from_bias(kappaB)
                # self.env.remove_from_bias(kappaB)

            else:  # user says UNSAT
                print(f"FindScope({Y})")
                scope = self.env.run_find_scope(Y)

                # FindC2 returns the constraint, OR None when delta is
                # empty at this scope (the bias may well contain the
                # target constraint, but it is currently SATISFIED, so
                # FindC has nothing to refine in this iteration). In
                # the None case we just skip add_to_cl; the next QGen
                # example that violates the target will surface it.
                #
                # If FindC raises (genuinely) we fall back to the
                # template shortcut's pinned candidate when one exists.
                last_cand = getattr(self.env.find_scope, "last_candidate", None)
                try:
                    print(f"FindC: {scope}")
                    c = self.env.run_findc(scope)
                except Exception as findc_err:
                    if last_cand is None or "Collapse" not in str(findc_err):
                        raise
                    tmpl, cand = last_cand
                    n2v = self.env.instance.bias._name_to_var()
                    c = tmpl.materialize(cand, n2v)
                    print(
                        f"FindC raised; using template candidate "
                        f"{tmpl.type_key}{cand.scope} parent={cand.params.get('parent')}"
                    )

                if c is None:
                    print(
                        "FindC: skipping iteration (no disambiguation "
                        "possible at this scope)"
                    )
                    continue

                self.env.add_to_cl(c)


class TrackAndCacheCAEnv(ActiveCAEnv):
    def __init__(self):
        super().__init__(
            findc=GFindC2(time_limit=300),
            qgen=GPQGen(time_limit=300),
            find_scope=GFindScope2(time_limit=300),
        )
        self.n_positive = 0
        self.n_negative = 0
        self.n_cache_hit = 0
        self.query_cache = {}

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
        """
        Add the given constraints to the list of learned constraints

        :param C: Constraints to add to CL
        """
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), (
            "add_to_cl accepts as input a list of constraints or a constraint"
        )

        if self.verbose >= 3:
            print(f"adding the following constraints to C_L: {C}")

        # Add constraint(s) c to the learned network and remove them from the bias
        self.instance.cl.extend(C)
        # self.instance.bias = list(set(self.instance.bias) - set(C))
        self.instance.bias.mark_as_learned(C)

        self.metrics.cl += len(C)
        if self.verbose == 1:
            for c in C:
                print("L", end="")


def run_experiment(
    uvl_path: str,
    timeout: int = 0,
    verify: bool = False,
    cleanup: bool = False,
    export_uvl: str | None = None,
    algorithm: str = "quacq",
    skip_collapse: bool = True,
    group_bias_max: int = 1,
    cliques_cutoff: int = 1,
    analyze_and_learn: bool = False,
) -> dict:
    """Run CA on a single UVL model (tree-unknown scenario) and return a results dict.

    Returns a dict with keys:
        model, features, cnf_clauses, bias_size, learned, queries_*,
        time_*, constraints, converged, error, inferred_tree,
        n_skipped_collapses (only when skip_collapse=True)
    """
    uvl_path = str(uvl_path)
    result = {"model": uvl_path, "algorithm": algorithm, "error": None}

    if timeout > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        t0 = time.monotonic()

        # 1. Extract feature names only (no tree)
        feature_names = extract_feature_names(uvl_path)
        n_features = len(feature_names)
        result["features"] = n_features
        result["feature_names"] = feature_names
        print(f"  features: {n_features}")

        # 2. Variables
        variables = [cp.boolvar(name=f) for f in feature_names]

        # 3. Target constraints (oracle)
        target = extract_target_constraints(uvl_path, variables, feature_names)
        result["cnf_clauses"] = len(target)
        print(f"  CNF clauses: {len(target)}")

        # SETUP STOP

        # 4. Flat binary bias (no tree) + optional group bias
        t_bias = time.monotonic()
        bias = GraphBias(build_bias(variables))
        result["bias_size"] = len(bias)
        result["group_bias_max"] = group_bias_max
        result["time_bias"] = round(time.monotonic() - t_bias, 4)
        print(
            f"  bias: {len(bias)} constraints (group_bias_max={group_bias_max}, {result['time_bias']:.2f}s)"
        )

        # 5. Run CA
        t_ca = time.monotonic()
        problem = ProblemInstance(variables=variables, bias=bias)
        oracle = ConstraintOracle(target)
        metrics = Metrics()

        def debug_query(q):
            vars = []
            varval = []

            for qi in q:
                vn, vv = qi.split("=")
                vars.append(next(v for v in variables if v.name == vn))
                varval.append(vv == "True")

            restore_scope_values(vars, varval)
            resp = oracle.answer_membership_query(vars)
            return resp

        # q = ['Core=True', 'Diesel=True', 'Electric=False', 'Metrics=False', 'System=True', 'Tracing=False']
        # debug_query(q)
        # q = ['Core=True', 'Diesel=True', 'Electric=False', 'Logging=False', 'Metrics=False', 'System=True', 'Tracing=False']
        # debug_query(q)

        ca_env = TrackAndCacheCAEnv()
        ca = GQuAcq(ca_env=ca_env)
        # ca = GMQuAcq2(
        #     ca_env=skip_env,
        #     perform_analyzeAndLearn=True,  # analyze_and_learn,
        #     cliques_cutoff=cliques_cutoff,
        # )

        try:
            learned_instance = ca.learn(
                instance=problem,
                oracle=oracle,
                verbose=3,
                metrics=metrics,
            )
        except Exception as e:
            print(f"Exception occurred: {e}")
            print("EXCEPTION == Learned CL")
            for c in ca_env.instance.cl:
                print(f"  {c}")

            raise

        print("+++ CA run complete +++")
        pickle.dump(ca_env.query_cache, open("query_cache.p", "wb"))

        print(f"  Learned from CA: {len(learned_instance.cl)}")

        if len(learned_instance.cl) <= 100:
            for c in learned_instance.cl:
                print(f"    {c}")

        # TODO Remaining postprocessing for graph bias

        metrics.finalize_statistics()
        result["time_ca"] = round(time.monotonic() - t_ca, 4)

        result["queries_positive"] = ca_env.n_positive
        result["queries_negative"] = ca_env.n_negative
        if skip_collapse:
            result["n_skipped_collapses"] = ca_env.n_skipped
            if ca_env.n_skipped:
                print(f"  skipped collapses: {ca_env.n_skipped}")

        # 6. Collect results
        learned = learned_instance.cl
        result["converged"] = bool(metrics.converged)
        result["queries_total"] = metrics.total_queries
        result["queries_membership"] = metrics.membership_queries_count
        result["queries_recommendation"] = metrics.recommendation_queries_count
        result["queries_generalization"] = metrics.generalization_queries_count
        result["time_ca_internal"] = round(metrics.total_time, 4)
        result["constraints_learned"] = [str(c) for c in learned]

        print(
            f"  learned {len(learned)} constraints in {result['queries_total']} queries ({time.monotonic() - t_ca:.2f}s)"
        )
        # 7-9. Unified tree inference + refinement
        n_learned_pre_tree = len(learned)
        learned, inferred, n_tree_queries = infer_and_refine_tree(
            feature_names,
            variables,
            learned,
            oracle,
        )
        n_refined = len(learned) - n_learned_pre_tree
        result["learned"] = len(learned)
        result["refined"] = n_refined
        result["completeness_added"] = 0  # subsumed by unified pipeline
        result["tree_queries"] = n_tree_queries
        result["queries_total"] = metrics.total_queries + n_tree_queries
        result["time_total"] = round(time.monotonic() - t0, 4)

        print(f"  refined tree: {len(inferred)} parent nodes")

        # 10. Validate tree structure
        validation_errors = _validate_tree(feature_names, inferred)
        result["tree_validation"] = {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
        }

        # Apply fix for multiple parent violations if needed
        if validation_errors:
            print("  Tree validation failed, applying fixes...")
            fixed_tree, cross_tree_cl = _fix_multi_parent_tree(feature_names, inferred)
            result["tree_fixed"] = {
                "applied_fix": True,
                "cross_tree_constraints": cross_tree_cl,
                "errors_after_fix": _validate_tree(feature_names, fixed_tree),
            }
            inferred = fixed_tree
            # Re-serialize fixed tree
            result["inferred_tree"] = {
                parent: [[gtype, children] for gtype, children in groups]
                for parent, groups in inferred.items()
            }
            print(f"  Fixed tree: {len(inferred)} parent nodes")
        # Until here no duplicate constraints
        # 11. Reconstruct enhanced constraint model from tree + cross-tree learned
        enhanced_cl, cross_tree_cl = constraints_from_tree(
            feature_names, variables, inferred, learned
        )

        # 12. Cleanup: remove spurious cross-tree constraints
        if cleanup:
            t_cleanup = time.monotonic()
            enhanced_cl, removed_cl, n_cleanup_queries = cleanup_dumb(  # TODO
                feature_names, variables, enhanced_cl, inferred, oracle
            )
            result["cleanup_removed"] = len(removed_cl)
            result["cleanup_removed_constraints"] = [str(c) for c in removed_cl]
            result["queries_cleanup"] = n_cleanup_queries
            result["queries_total"] += n_cleanup_queries
            result["time_cleanup"] = round(time.monotonic() - t_cleanup, 4)
            # Re-derive cross-tree from the cleaned enhanced set.
            # cleanup_constraints returns structural + kept_cross, so
            # the cross-tree portion is everything beyond structural.
            structural_only, _ = constraints_from_tree(
                feature_names, variables, inferred, []
            )
            n_structural = len(structural_only)
            cross_tree_cl = enhanced_cl[n_structural:]
        else:
            result["cleanup_removed"] = 0
            result["queries_cleanup"] = 0

        result["constraints"] = [str(c) for c in enhanced_cl]
        result["inferred_tree"] = infer_tree(feature_names, variables, enhanced_cl)
        result["query_cache_hits"] = ca_env.n_cache_hit

        # # Serialize to JSON-safe format: {parent: [[gtype, [children]], ...]}
        # result["inferred_tree"] = {
        #     parent: [[gtype, children] for gtype, children in groups]
        #     for parent, groups in inferred.items()
        # }

        if verify:
            result["verification"] = verify_learned(learned, target, variables)
            eq_flat = result["verification"]["equivalent"]
            print(f"  verification (flat): equivalent={eq_flat}")

            result["verification_tree"] = verify_learned(enhanced_cl, target, variables)
            eq_tree = result["verification_tree"]["equivalent"]
            print(f"  verification (tree): equivalent={eq_tree}")

        if export_uvl:
            out_path = Path(export_uvl)
            if out_path.is_dir():
                stem = Path(uvl_path).stem
                out_path = out_path / f"{stem}_learned.uvl"
            exported, skipped_ex = export_learned_to_uvl(
                feature_names, inferred, cross_tree_cl, str(out_path)
            )
            result["exported_uvl"] = str(out_path)
            print(
                f"  exported to {out_path} ({exported} constraints, {skipped_ex} skipped)"
            )

    except TimeoutError:
        result["error"] = f"timeout ({timeout}s)"
        print(f"  TIMEOUT after {timeout}s")
    except Exception as e:
        import traceback

        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR: {result['error']}")
        print(f"  Traceback:\n{traceback.format_exc()}")
    finally:
        if timeout > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return result


# ── Output ───────────────────────────────────────────────────────────


def print_result(r: dict):
    """Print a human-readable summary of one experiment result."""
    print(f"\n{'=' * 65}")
    print(f"Model: {r['model']}")
    print(f"{'=' * 65}")

    if r.get("error"):
        print(f"  ERROR: {r['error']}")
        if "features" in r:
            print(f"  (features: {r['features']})")
        return

    print(f"  Algorithm:       {r.get('algorithm', 'quacq')}")
    print(f"  Features:        {r['features']}")
    print(f"  CNF clauses:     {r['cnf_clauses']}")
    print(f"  Bias size:       {r['bias_size']}")
    print(
        f"  Learned:         {r['learned']} constraints"
        f" (+{r.get('completeness_added', 0)} completeness in {r.get('queries_completeness', '?')} queries,"
        f" +{r.get('refined', 0)} tree-refined in {r.get('tree_queries', '?')} queries)"
    )
    if "n_skipped_collapses" in r:
        print(f"  Skipped collapses: {r['n_skipped_collapses']}")
    print(f"  Converged:       {r['converged']}")
    print(f"  Queries total:   {r['queries_total']}")
    print(f"  Cache Hits:      {r['query_cache_hits']}")
    print(
        f"    CA:            {r['queries_membership'] + r['queries_recommendation'] + r['queries_generalization']}"
    )
    print(f"      positive:    {r.get('queries_positive', '?')}")
    print(f"      negative:    {r.get('queries_negative', '?')}")
    print(f"      membership:  {r['queries_membership']}")
    print(f"      recommend.:  {r['queries_recommendation']}")
    print(f"      generaliz.:  {r['queries_generalization']}")
    print(f"    completeness:  {r.get('queries_completeness', '?')}")
    print(f"    tree refine:   {r.get('tree_queries', '?')}")
    print(f"    cleanup:       {r.get('queries_cleanup', '?')}")
    if r.get("cleanup_removed", 0) > 0:
        print(f"  Cleanup removed: {r['cleanup_removed']} spurious constraint(s)")
        for c in r.get("cleanup_removed_constraints", []):
            print(f"    - {c}")
    print(f"  Time total:      {r['time_total']:.2f}s")
    print(f"    bias build:    {r['time_bias']:.2f}s")
    print(f"    CA solver:     {r['time_ca']:.2f}s")
    if "time_cleanup" in r:
        print(f"    cleanup:       {r['time_cleanup']:.2f}s")
    print("  Learned constraints:")
    for c in r.get("constraints", []):
        print(f"    {c}")
    if "inferred_tree" in r:
        print(f"  Inferred tree ({len(r['inferred_tree'])} parent nodes):")
        for parent, groups in r["inferred_tree"].items():
            for gtype, children in groups:
                print(f"    {parent} --[{gtype}]--> {children}")
    if "tree_validation" in r:
        tv = r["tree_validation"]
        print(f"  Tree validation: {'PASS' if tv['valid'] else 'FAIL'}")
        if tv["errors"]:
            print("    Errors:")
            for err in tv["errors"]:
                print(f"      - {err}")
        if "tree_fixed" in r:
            tf = r["tree_fixed"]
            print("  Tree fix applied: YES")
            print(f"    Cross-tree constraints: {len(tf['cross_tree_constraints'])}")
            print(
                f"    Errors after fix: {'None' if tf['errors_after_fix'] == [] else tf['errors_after_fix']}"
            )
    for label, key in [
        ("Verification (flat)", "verification"),
        ("Verification (tree)", "verification_tree"),
    ]:
        if key not in r:
            continue
        v = r[key]
        print(f"  {label}:")
        print(f"    Equivalent:        {v['equivalent']}")
        fp = (
            "No"
            if not v["has_false_positives"]
            else f"Yes (example: {v['fp_example']})"
        )
        fn = (
            "No"
            if not v["has_false_negatives"]
            else f"Yes (example: {v['fn_example']})"
        )
        print(f"    False positives:   {fp}")
        print(f"    False negatives:   {fn}")
    if "exported_uvl" in r:
        print(f"  Exported UVL:      {r['exported_uvl']}")


# ── CLI ──────────────────────────────────────────────────────────────


def generate_example(path: str = "sandwich.uvl"):
    Path(path).write_text(EXAMPLE_UVL)
    print(f"Generated example UVL model: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run constraint acquisition on UVL feature models "
            "(tree-unknown variant — only feature names used during learning)"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="UVL file(s) or directories to scan for .uvl files",
    )
    parser.add_argument(
        "--generate-example",
        action="store_true",
        help="Generate sandwich.uvl and add it to the input list",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Skip models with more features than this (0 = no limit)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-model timeout in seconds (0 = no limit)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Write one JSON file per model into this directory",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-model console output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run SAT equivalence check after learning",
    )
    parser.add_argument(
        "--export-uvl",
        type=str,
        default=None,
        metavar="PATH",
        help="Export learned model to UVL file (dir → {stem}_learned.uvl)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Disable the post-learning cleanup pass (enabled by default). "
            "The cleanup tests each cross-tree constraint by generating a "
            "counter-example and asking the oracle, removing spurious "
            "constraints that the oracle does not enforce."
        ),
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep SAT-based constraint diagnosis after learning (uses tree-enhanced constraints)",
    )
    parser.add_argument(
        "--no-skip-collapse",
        action="store_true",
        help=(
            "Disable Collapse skipping — let pycona raise an exception on Collapse "
            "instead of silently continuing (default: skip is enabled)"
        ),
    )
    parser.add_argument(
        "--group-bias-max",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum group size for n-ary group bias. 1 = no group bias (default), "
            "3 = add group bias for sizes 2 and 3. Completeness clauses are recovered "
            "post-CA by refine_completeness regardless of this setting."
        ),
    )

    alg_group = parser.add_argument_group(
        "algorithm",
        "CA algorithm and its options (see --algorithm choices for details)",
    )
    alg_group.add_argument(
        "--algorithm",
        "-a",
        choices=list(ALGORITHMS),
        default="quacq",
        metavar="ALG",
        help=(
            "CA algorithm to use. Choices: "
            + ", ".join(ALGORITHMS)
            + " (default: mquacq2)"
        ),
    )
    alg_group.add_argument(
        "--cliques-cutoff",
        type=int,
        default=1,
        metavar="K",
        help=("[mquacq2, growacq] Quasi-clique detection cutoff (default: 1)"),
    )
    alg_group.add_argument(
        "--no-analyze-learn",
        action="store_true",
        help="[mquacq2, growacq] Disable the analyzeAndLearn step",
    )
    alg_group.add_argument(
        "--qg-max",
        type=int,
        default=10,
        metavar="N",
        help="[mineacq, genacq] Max generalization queries per learned constraint (default: 10)",
    )
    alg_group.add_argument(
        "--grow-inner",
        choices=["quacq", "mquacq", "mquacq2"],
        default="mquacq2",
        help="[growacq] Inner algorithm used at each growth step (default: mquacq2)",
    )

    args = parser.parse_args()

    input_paths = list(args.paths)
    if args.generate_example:
        input_paths.append(generate_example())

    if not input_paths:
        parser.error("No input files specified. Use paths or --generate-example.")

    uvl_files = collect_uvl_paths(input_paths)
    if not uvl_files:
        parser.error("No .uvl files found")

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(uvl_files)} UVL model(s)")

    skip_collapse = not args.no_skip_collapse
    results = []
    skipped = 0
    already_done = 0
    for i, uvl_path in enumerate(uvl_files, 1):
        print(f"[{i}/{len(uvl_files)}] {uvl_path}")

        if args.max_features > 0:
            try:
                names = extract_feature_names(str(uvl_path))
                if len(names) > args.max_features:
                    print(f"  SKIP: {len(names)} features > {args.max_features} limit")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"  SKIP: cannot parse ({e})")
                skipped += 1
                continue

        r = run_experiment(
            str(uvl_path),
            timeout=args.timeout,
            verify=args.verify,
            cleanup=not args.no_cleanup,
            export_uvl=args.export_uvl,
            algorithm=args.algorithm,
            skip_collapse=skip_collapse,
            group_bias_max=args.group_bias_max,
            cliques_cutoff=args.cliques_cutoff,
            analyze_and_learn=not args.no_analyze_learn,
        )
        results.append(r)

        if not args.quiet:
            print_result(r)

        if out_dir is not None:
            save_result(r, out_dir / f"{uvl_path.stem}.json")

    if args.deep:
        from report_results import deep_analysis

        # Add synthetic keys expected by deep_analysis
        for r in results:
            r.setdefault("_name", Path(r["model"]).stem)
            r.setdefault("_file", r["model"])
        deep_subset = [r for r in results if r.get("converged") and not r.get("error")]
        if deep_subset:
            deep_analysis(deep_subset)

    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    print(f"\n{'=' * 65}")
    print(
        f"Summary: {len(ok)} succeeded, {len(failed)} failed, {skipped} skipped, {already_done} already done"
    )
    if ok:
        total_q = sum(r["queries_total"] for r in ok)
        total_t = sum(r["time_total"] for r in ok)
        total_learned = sum(r["learned"] for r in ok)
        print(f"  Total learned:  {total_learned} constraints")
        print(f"  Total queries:  {total_q}")
        print(f"  Total time:     {total_t:.2f}s")
        if skip_collapse:
            total_skipped = sum(r.get("n_skipped_collapses", 0) for r in ok)
            print(f"  Total skipped collapses: {total_skipped}")
    if failed:
        print("  Failed models:")
        for r in failed:
            print(f"    {r['model']}: {r['error']}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
