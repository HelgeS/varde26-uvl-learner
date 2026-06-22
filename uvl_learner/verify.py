"""SAT-based equivalence verification for learned constraint sets.

This is the "validation" pillar: given a learned constraint set and the
ground-truth target clauses (from the UVL oracle), decide whether they are
logically equivalent and, if not, produce a counterexample.
"""

import cpmpy as cp


def verify_learned(learned_cl: list, target_cl: list, variables: list) -> dict:
    """Check whether *learned_cl* is logically equivalent to *target_cl*.

    Uses O(n) sequential SAT checks:

    - False positive: a solution that satisfies all learned constraints but
      violates at least one target clause.
    - False negative: a solution that satisfies all target clauses but violates
      at least one learned constraint.

    Parameters
    ----------
    learned_cl  : learned CPMpy constraint list
    target_cl   : ground-truth CPMpy constraint list (CNF from the UVL model)
    variables   : list of CPMpy BoolVar — used to read back the counterexample

    Returns
    -------
    dict with keys:
        equivalent          bool
        has_false_positives bool
        fp_example          dict[str, bool] | None
        has_false_negatives bool
        fn_example          dict[str, bool] | None
    """
    has_fp = False
    fp_example = None
    has_fn = False
    fn_example = None

    # False positives: satisfies learned but violates some target clause
    for t in target_cl:
        if cp.Model(learned_cl + [~t]).solve():
            has_fp = True
            fp_example = {v.name: bool(v.value()) for v in variables}
            break

    # False negatives: satisfies target but violates some learned constraint
    for ell in learned_cl:
        if cp.Model(target_cl + [~ell]).solve():
            has_fn = True
            fn_example = {v.name: bool(v.value()) for v in variables}
            break

    return {
        "equivalent": not has_fp and not has_fn,
        "has_false_positives": has_fp,
        "fp_example": fp_example,
        "has_false_negatives": has_fn,
        "fn_example": fn_example,
    }
