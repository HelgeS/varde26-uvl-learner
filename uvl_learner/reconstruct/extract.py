"""Reconstruct CPMpy constraints from an inferred feature tree."""

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison


def constraints_from_tree(
    feature_names: list[str],
    variables: list,
    tree_info: dict,
    learned_cl: list,
) -> list:
    """Reconstruct a CPMpy constraint model from the inferred tree structure.

    Generates structural constraints directly from ``tree_info`` (child→parent
    edges, group-completeness for or/alternative groups, mutual exclusions for
    alternative groups, mandatory equivalences) and then appends any constraints
    from ``learned_cl`` that are *not* already encoded by the tree (cross-tree
    requires/excludes, equivalences not captured as mandatory pairs, unary
    core/dead).

    The result can be passed to ``verify_learned`` to test whether the
    tree reconstruction — together with residual learned constraints —
    is semantically equivalent to the oracle target.

    Notes
    -----
    For alternative groups the tree encoding adds *both* directions of each
    mutual exclusion (``Ci => ~Cj`` and ``Cj => ~Ci``) even if only one was
    learned; this is fine because both directions are the same SAT clause.
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}

    # ── 1. Structural constraints from tree ────────────────────────────
    structural_cl: list = []
    tree_edges: set[tuple[str, str]] = set()  # (child, parent) direct edges
    alt_pairs: set[frozenset] = set()  # {c1, c2} alternative siblings
    mandatory_pairs: set[frozenset] = set()  # {parent, child} mandatory groups
    or_alt_parents: set[str] = set()  # parents with completeness clauses

    # Root feature is always selected in a feature model
    all_children: set[str] = set()
    for groups in tree_info.values():
        for _, children in groups:
            all_children.update(children)
    for name in feature_names:
        if name not in all_children:
            structural_cl.append(var_of[name])
            break

    for parent, groups in tree_info.items():
        pv = var_of[parent]
        for gtype, children in groups:
            cvs = [var_of[c] for c in children]
            # child → parent holds for every group type
            for c, cv in zip(children, cvs):
                tree_edges.add((c, parent))
                structural_cl.append(cv.implies(pv))

            if gtype == "mandatory":
                # parent ↔ child (single mandatory child)
                structural_cl.append(pv.implies(cvs[0]))
                mandatory_pairs.add(frozenset({parent, children[0]}))
            elif gtype in ("or", "alternative"):
                # group completeness: parent → at least one child
                structural_cl.append(pv.implies(cp.any(cvs)))
                or_alt_parents.add(parent)
                if gtype == "alternative":
                    # mutual exclusion between every pair of children
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            structural_cl.append(cvs[i].implies(~cvs[j]))
                            alt_pairs.add(frozenset({children[i], children[j]}))
            # "optional": only child → parent (already added above)

    # ── 2. Cross-tree constraints from learned ─────────────────────────
    # Retain learned constraints not already encoded by the tree structure.
    # Compute tree-reachable ancestors for each feature so we can skip
    # any A => B where B is an ancestor of A in the tree.
    tree_ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child, parent in tree_edges:
        tree_ancestors[child].add(parent)
    # Propagate: ancestors of my parent are also my ancestors
    changed = True
    while changed:
        changed = False
        for child in feature_names:
            for parent in list(tree_ancestors[child]):
                new_anc = tree_ancestors[parent] - tree_ancestors[child]
                if new_anc:
                    tree_ancestors[child].update(new_anc)
                    changed = True

    cross_tree_cl: list = []
    for c in learned_cl:
        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: skip if B is a tree ancestor of A (direct or transitive)
                    if rhs.name in tree_ancestors.get(lhs.name, set()):
                        continue
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: skip if it's an alternative-sibling exclusion
                    if frozenset({lhs.name, rhs._bv.name}) in alt_pairs:
                        continue
                else:
                    # A => any([...]): skip if A is an or/alt parent (already have completeness)
                    if lhs.name in or_alt_parents:
                        continue
            cross_tree_cl.append(c)
        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: skip if it's encoded as a mandatory pair
                if frozenset({a_expr.name, b_expr.name}) in mandatory_pairs:
                    continue
            cross_tree_cl.append(c)
        else:
            # Unary constraints (core/dead): include if not already part of the tree
            if c not in structural_cl:
                cross_tree_cl.append(c)

    return structural_cl + cross_tree_cl, cross_tree_cl


# ── Constraint cleanup (post-tree, pre-verification) ────────────────


