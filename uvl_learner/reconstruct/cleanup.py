"""Remove spurious cross-tree constraints via targeted oracle queries."""

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison
from pycona import ConstraintOracle

from ._common import _mentioned_features
from .extract import constraints_from_tree


def cleanup_dumb(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    tree_info: dict,
    oracle: ConstraintOracle,
):
    def _negate(c):
        negation = [~c]

        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                # Force the antecedent True
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: force A=T, B=F
                    negation = [lhs, ~rhs]
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: force A=T, B=T
                    negation = [lhs, rhs._bv]
                else:
                    # A => complex_rhs: force A=T, negate rhs
                    negation = [lhs, ~rhs]

        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: try A=T, B=F
                # negation = [a_expr, ~b_expr]
                negation = [sum([a_expr, b_expr]) == 1]

        return negation

    removed: list = []
    n_queries = 0
    remaining = list(learned_cl)
    changed = True

    while changed:
        changed = False
        kept_cross: list = []

        for i, c in enumerate(remaining):
            negation = _negate(c)
            all_cl = [c2 for c2 in learned_cl if c2 not in [c] and c2 not in removed]
            # TODO Not sure we actually need reachability
            m = cp.Model(all_cl + negation)
            if not m.solve():
                # Can't violate c while satisfying everything else → redundant, keep
                kept_cross.append(c)
                print(f"  cleanup: KEEP (implied) {c}")
                continue

            # Ask the oracle
            n_queries += 1
            try:
                # TODO Some form of caching of queries would be nice,
                # in this loop it can happen that we query the same thing multiple times
                oracle_accepts = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    kept_cross.append(c)
                    print(f"  cleanup: KEEP (collapse) {c}")
                    continue
                raise

            if oracle_accepts:
                # Oracle accepts the c-violating assignment → c is spurious
                removed.append(c)
                print(f"  cleanup: REMOVE {c}")
                changed = True
            else:
                kept_cross.append(c)
                print(f"  cleanup: KEEP (oracle rejected) {c}")

        remaining = kept_cross

    print(f"  cleanup: removed {len(removed)} / {len(learned_cl)} constraint(s) in {n_queries} queries")
    # cleaned_cl = list(learned_cl)  + kept_cross
    cleaned_cl = [c for c in learned_cl if c not in removed]
    return cleaned_cl, removed, n_queries


def cleanup_constraints(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    tree_info: dict,
    oracle: ConstraintOracle,
) -> tuple[list, list, int]:
    """Remove spurious cross-tree constraints by generating per-constraint counter-examples.

    For each cross-tree constraint C (i.e. constraints not implied by the tree
    structure), we try to find an assignment that:

    1. Satisfies all tree-structural constraints (child→parent, group
       completeness, alternative exclusions, mandatory links).
    2. Satisfies all other cross-tree constraints (everything except C).
    3. **Violates** C.
    4. Keeps the features mentioned in C **reachable** — their ancestor path
       in the tree is forced active so the test is non-vacuous.

    If such an assignment exists and the oracle **accepts** it, then C is
    spurious (the ground truth does not require it) and we remove it.

    If the solver says UNSAT, C is implied by the remaining constraints and
    is therefore redundant but not wrong — we keep it.

    If the oracle rejects the assignment, C is genuinely needed — we keep it.

    Parameters
    ----------
    feature_names : list[str]
    variables : list of CPMpy BoolVar
    learned_cl : list of CPMpy constraints (flat CA + completeness + tree-refined)
    tree_info : dict  {parent: [(gtype, [children]), ...]}
    oracle : ConstraintOracle

    Returns
    -------
    (cleaned_cl, removed_cl, n_queries)
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}

    # ── Build tree-structural constraints and classify learned ────────
    structural_cl, cross_tree_cl = constraints_from_tree(
        feature_names, variables, tree_info, learned_cl
    )

    # Build ancestor map from tree_info for reachability forcing
    tree_edges: set[tuple[str, str]] = set()  # (child, parent)
    for parent, groups in tree_info.items():
        for _, children in groups:
            for child in children:
                tree_edges.add((child, parent))

    ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child, parent in tree_edges:
        ancestors[child].add(parent)
    changed = True
    while changed:
        changed = False
        for child in feature_names:
            for parent in list(ancestors[child]):
                new_anc = ancestors[parent] - ancestors[child]
                if new_anc:
                    ancestors[child].update(new_anc)
                    changed = True

    # ── Helper: negate a constraint ───────────────────────────────────
    def _negate_with_reachability(c) -> tuple[list, list]:
        """Return (negation_assumptions, reachability_assumptions).

        negation_assumptions: list of CPMpy expressions that together negate C
            while keeping the relevant features active.
        reachability_assumptions: list of BoolVars forced True (ancestors).
        """
        feats = _mentioned_features(c)
        # Reachability: all ancestors of mentioned features must be True
        reach = set()
        for f in feats:
            reach.update(ancestors.get(f, set()))
        reachability = [var_of[a] for a in reach if a in var_of]

        # Also force the features themselves to be active where sensible.
        # For A => B: force A=True (makes the implication non-vacuous),
        #   the negation then requires B=False.
        # For A => ~B: force A=True, negation requires B=True.
        # For A == B: try A=True, B=False first.
        # Generic fallback: just negate the whole constraint.
        negation = [~c]

        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                # Force the antecedent True
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: force A=T, B=F
                    negation = [lhs, ~rhs]
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: force A=T, B=T
                    negation = [lhs, rhs._bv]
                else:
                    # A => complex_rhs: force A=T, negate rhs
                    negation = [lhs, ~rhs]

        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: try A=T, B=F
                negation = [a_expr, ~b_expr]

        return negation, reachability

    # ── Main loop: test each cross-tree constraint ────────────────────
    removed: list = []
    n_queries = 0
    remaining = list(cross_tree_cl)
    changed = True

    while changed:
        changed = False
        kept_cross: list = []

        for i, c in enumerate(remaining):
            negation, reachability = _negate_with_reachability(c)

            # Build the constraint set: structural + already-kept cross-tree + all
            # remaining untested cross-tree (conservative: don't remove a constraint
            # that's only needed because another spurious one was kept).
            other_cross = kept_cross + remaining[i + 1 :]
            all_cl = list(structural_cl) + other_cross
            all_cl = [c2 for c2 in all_cl if c2 not in [c] and c2 not in removed]
            # TODO Not sure we actually need reachability
            m = cp.Model(all_cl + reachability + negation)
            if not m.solve():
                # Can't violate c while satisfying everything else → redundant, keep
                kept_cross.append(c)
                print(f"  cleanup: KEEP (implied) {c}")
                continue

            # Ask the oracle
            n_queries += 1
            try:
                # TODO Some form of caching of queries would be nice,
                # in this loop it can happen that we query the same thing multiple times
                oracle_accepts = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    kept_cross.append(c)
                    print(f"  cleanup: KEEP (collapse) {c}")
                    continue
                raise

            if oracle_accepts:
                # Oracle accepts the c-violating assignment → c is spurious
                removed.append(c)
                print(f"  cleanup: REMOVE {c}")
                changed = True
            else:
                kept_cross.append(c)
                print(f"  cleanup: KEEP (oracle rejected) {c}")

        remaining = kept_cross

    print(f"  cleanup: removed {len(removed)} / {len(cross_tree_cl)} cross-tree constraint(s) in {n_queries} queries")

    # Return: structural + surviving cross-tree
    cleaned_cl = list(structural_cl) + kept_cross
    cleaned_cl = [c for c in cleaned_cl if c not in removed]
    return cleaned_cl, removed, n_queries
