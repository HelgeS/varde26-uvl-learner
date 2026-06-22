"""Grow-bias construction.

- build_bias() — flat binary candidate set.
- add_group_bias() — widen an existing candidate set with n-ary group
  candidates of a given size; called repeatedly by the grow runner, which
  enlarges the group size whenever CA collapses.
"""

import cpmpy as cp


def build_bias(variables):
    """Build flat pairwise candidate constraint set — binary only.

    For each pair (i < j):
        vi.implies(vj), vj.implies(vi)     # requires (both directions)
        vi.implies(~vj), vj.implies(~vi)   # excludes (both directions)
        vi == vj                           # equivalence

    For each variable vi:
        vi, ~vi                            # core / dead (unary)

    Size: 5·C(n,2) + 2·n.  For n=20 → 990 candidates.

    N-ary group-completeness clauses are not representable here.  With
    skip-collapse enabled (default), Collapse events from them are suppressed
    and the missing clauses are recovered post-CA by ``refine_completeness``.
    """
    bias = []
    n = len(variables)

    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = variables[i], variables[j]
            bias.append(vi.implies(vj))
            bias.append(vj.implies(vi))
            bias.append(vi.implies(~vj))
            bias.append(vj.implies(~vi))
            bias.append(vi == vj)

    # TODO mquacq2 does not support unary constraints
    # for vi in variables:
    #     bias.append(vi)
    #     bias.append(~vi)

    return bias


def add_group_bias(variables, group_size: int, constraint_base: list) -> list:
    from itertools import combinations

    bias = []
    n = len(variables)

    for parent_idx in range(n):
        parent = variables[parent_idx]
        others = [variables[j] for j in range(n) if j != parent_idx]

        valid_children = [c for c in others if c.implies(parent) in constraint_base]
        valid_exclusions = {
            (c1, c2)
            for c1, c2 in combinations(valid_children, 2)
            if c1.implies(~c2) in constraint_base or c2.implies(~c1) in constraint_base
        }

        for children in combinations(valid_children, group_size):
            bias.append(parent.implies(cp.any(children)))

            # Fast alternative group check:
            # Are all pairs within this specific combination in our exclusions set?
            # This is an instant O(1) hash lookup per pair.
            if all(
                (c1, c2) in valid_exclusions for c1, c2 in combinations(children, 2)
            ):
                bias.append(parent.implies(sum(children) == 1))

    return bias


# ── Algorithm factory ─────────────────────────────────────────────────


