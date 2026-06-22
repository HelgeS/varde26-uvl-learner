"""Shared helpers for the reconstruct pipeline."""

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison


def _mentioned_features(c) -> set[str]:
    """Return set of feature names that appear in constraint c."""
    names = set()

    def _walk(expr):
        if isinstance(expr, NegBoolView):
            names.add(expr._bv.name)
        elif isinstance(expr, _BoolVarImpl):
            names.add(expr.name)
        elif hasattr(expr, "args"):
            for arg in expr.args:
                _walk(arg)

    _walk(c)
    return names


def merge_shared_sets(sets):
    merged = []

    for s in sets:
        # Convert to a standard, mutable set
        current_set = set(s)
        unmerged = []

        # Compare the current set with the ones we've already processed
        for m in merged:
            if not current_set.isdisjoint(m):
                # If they share an item, combine them
                current_set.update(m)
            else:
                # If they don't share an item, keep the processed set as is
                unmerged.append(m)

        # Add the newly merged set back into the list
        unmerged.append(current_set)
        merged = unmerged

    return merged


# ── Unified tree inference + refinement ──────────────────────────────


