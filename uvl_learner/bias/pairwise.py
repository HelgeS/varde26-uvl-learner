"""Static pairwise bias construction.

- build_bias() — flat binary candidate set (requires/excludes/equivalence).
- build_group_bias() — optional n-ary group-semantics candidates.
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
            # bias.append((~vi).implies(~vj))
            # bias.append((~vj).implies(~vi))
            # bias.append(vi == vj)
            # TODO Is it non-normalized if we have a->b and b->a instead of a==b; I think so, because a scope is normally unordered

    # TODO mquacq2 does not support unary constraints
    # for vi in variables:
    #     bias.append(vi)
    #     bias.append(~vi)

    return bias


def build_group_bias(variables, group_size: int) -> list:
    """Build group-semantics bias constraints for all combinations of one parent
    and ``group_size`` children drawn from ``variables``.

    For every ordered choice of one parent variable and every ``group_size``-
    element subset of the remaining variables as children, the following
    candidate constraints are added:

      - OR group:      parent => (c1 | c2 | ... | cn)
      - Alternative:   parent => (sum(children) == 1)
      - Cardinality:   parent => (sum(children) >= k)  for k in 1..group_size
                       parent => (sum(children) <= k)  for k in 1..group_size
      - Child-parent:  ci => parent                    for each child ci

    Parameters
    ----------
    variables : list
        CPMpy BoolVar list (one per feature).
    group_size : int
        Number of children in each candidate group (must be >= 1).

    Returns
    -------
    list
        Flat list of CPMpy constraint expressions.
    """
    from itertools import combinations

    bias = []
    n = len(variables)

    for parent_idx in range(n):
        parent = variables[parent_idx]
        others = [variables[j] for j in range(n) if j != parent_idx]

        for children in combinations(others, group_size):
            children = list(children)

            # OR group: parent implies at least one child selected
            bias.append(parent.implies(cp.any(children)))

            # Alternative group: parent implies exactly one child selected
            bias.append(parent.implies(sum(children) == 1))

            # Cardinality bounds
            # for k in range(1, group_size + 1):
            #     bias.append(parent.implies(sum(children) >= k))
            #     bias.append(parent.implies(sum(children) <= k))

            # Every child implies the parent
            # for c in children:
            #     bias.append(c.implies(parent))

    return bias
