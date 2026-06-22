"""Diagnose why CA under-constrains: what target clauses are missing and
whether they could have been learned given the bias.

For each FN-only result (flat constraints are a subset of target, but miss
some target clauses), this script:

1. Rebuilds target CNF clauses and learned constraints from JSON
2. Finds which target clauses are NOT implied by learned constraints
3. Classifies each missing clause by shape (unary, binary implication,
   binary exclusion, equivalence, n-ary disjunction, etc.)
4. Checks whether each missing clause (or an equivalent) exists in the
   bias that was used during CA

Usage:
    python -m diagnostics.underconstraining results/ --summary
    python -m diagnostics.underconstraining results/model_foo.json -v
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from uvl_learner.oracle import (
    extract_feature_names,
    extract_target_constraints,
)


# ── Clause classification ──────────────────────────────────────────────


def _classify_clause(clause) -> dict:
    """Classify a single CNF clause by its shape.

    Returns a dict with keys:
        type: str — one of 'core', 'dead', 'implies', 'excludes',
              'equivalence', 'completeness', 'disjunction', 'other'
        arity: int — number of literals
        literals: list of (name, positive) tuples
        description: str — human-readable
    """
    # Unary: single literal
    if isinstance(clause, _BoolVarImpl) and not isinstance(clause, NegBoolView):
        return {
            "type": "core",
            "arity": 1,
            "literals": [(clause.name, True)],
            "description": f"core({clause.name})",
        }
    if isinstance(clause, NegBoolView):
        return {
            "type": "dead",
            "arity": 1,
            "literals": [(clause._bv.name, False)],
            "description": f"dead({clause._bv.name})",
        }

    # Multi-literal disjunction (from cp.any([...]))
    lits = _extract_literals(clause)
    if lits is None:
        return {
            "type": "other",
            "arity": 0,
            "literals": [],
            "description": str(clause),
        }

    n = len(lits)

    if n == 1:
        name, pos = lits[0]
        return {
            "type": "core" if pos else "dead",
            "arity": 1,
            "literals": lits,
            "description": f"{'core' if pos else 'dead'}({name})",
        }

    if n == 2:
        (a, a_pos), (b, b_pos) = lits
        # Binary clause: (lit_a OR lit_b)
        # Equivalent to: ~lit_a => lit_b
        if not a_pos and b_pos:
            return {
                "type": "implies",
                "arity": 2,
                "literals": lits,
                "description": f"{a} => {b}",
            }
        if a_pos and not b_pos:
            return {
                "type": "implies",
                "arity": 2,
                "literals": lits,
                "description": f"{b} => {a}",
            }
        if a_pos and b_pos:
            return {
                "type": "at_least_one",
                "arity": 2,
                "literals": lits,
                "description": f"({a} | {b})",
            }
        # Both negative: ~a | ~b => not(a & b) => a => ~b
        return {
            "type": "excludes",
            "arity": 2,
            "literals": lits,
            "description": f"{a} => ~{b}",
        }

    # N-ary clause
    neg_lits = [(name, pos) for name, pos in lits if not pos]
    pos_lits = [(name, pos) for name, pos in lits if pos]

    if len(neg_lits) == 1 and len(pos_lits) >= 2:
        parent = neg_lits[0][0]
        children = [name for name, _ in pos_lits]
        return {
            "type": "completeness",
            "arity": n,
            "literals": lits,
            "description": f"{parent} => or({children})",
        }

    return {
        "type": "disjunction",
        "arity": n,
        "literals": lits,
        "description": f"or({[f'{'~' if not p else ''}{n}' for n, p in lits]})",
    }


def _extract_literals(clause) -> list | None:
    """Extract literals from a CNF clause (disjunction).

    Returns list of (name: str, positive: bool) or None if not parseable.
    """
    if isinstance(clause, _BoolVarImpl) and not isinstance(clause, NegBoolView):
        return [(clause.name, True)]
    if isinstance(clause, NegBoolView):
        return [(clause._bv.name, False)]

    if isinstance(clause, Operator) and clause.name == "or":
        lits = []
        for arg in clause.args:
            sub = _extract_literals(arg)
            if sub is None:
                return None
            lits.extend(sub)
        return lits

    return None


# ── Bias coverage check ────────────────────────────────────────────────


def _is_in_binary_bias(clause_info: dict, feature_names: list) -> bool:
    """Check if a classified clause could be learned from the binary bias.

    The binary bias contains for each pair (i, j):
        vi => vj, vj => vi, vi => ~vj, vj => ~vi, vi == vj
    Plus unary: vi, ~vi (commented out but could be enabled).
    """
    ctype = clause_info["type"]

    if ctype in ("core", "dead"):
        return False  # Unary constraints are commented out in bias

    if ctype in ("implies", "excludes"):
        return True  # All binary implications/exclusions are in the bias

    if ctype == "at_least_one":
        # (a | b) is equivalent to (~a => b) which IS in the bias
        return True

    return False


def _is_in_group_bias(clause_info: dict, feature_names: list, max_group_size: int = 3) -> bool:
    """Check if a classified clause could be learned from the group bias.

    The group bias contains for group_size in {2, 3}:
        parent => or(children)        — completeness
        parent => sum(children) == 1  — alternative
    """
    if clause_info["type"] != "completeness":
        return False

    lits = clause_info["literals"]
    neg_lits = [name for name, pos in lits if not pos]
    pos_lits = [name for name, pos in lits if pos]

    if len(neg_lits) != 1:
        return False

    # The completeness clause has form: parent => or(child1, child2, ..., childN)
    # The group bias generates this for group_size in {2, 3}
    # But the target clause may have more children than max_group_size
    n_children = len(pos_lits)
    return n_children <= max_group_size


def check_bias_coverage(clause_info: dict, feature_names: list) -> dict:
    """Check whether a missing clause is covered by the bias.

    Returns dict with:
        in_binary_bias: bool
        in_group_bias: bool
        in_any_bias: bool
        reason: str — why it's missing from the bias (if not covered)
    """
    in_binary = _is_in_binary_bias(clause_info, feature_names)
    in_group = False #_is_in_group_bias(clause_info, feature_names)  #TODO Read group size from result json
    in_any = in_binary or in_group

    if in_any:
        reason = "in bias — CA should have learned it"
    elif clause_info["type"] in ("core", "dead"):
        reason = "unary constraints not in bias (mquacq2 limitation)"
    elif clause_info["type"] == "completeness":
        n_children = sum(1 for _, pos in clause_info["literals"] if pos)
        reason = f"completeness with {n_children} children > max group_size 3"
    elif clause_info["type"] == "disjunction":
        reason = f"general {clause_info['arity']}-ary disjunction not in bias"
    else:
        reason = "unknown constraint shape"

    return {
        "in_binary_bias": in_binary,
        "in_group_bias": in_group,
        "in_any_bias": in_any,
        "reason": reason,
    }


# ── Missing clause analysis ───────────────────────────────────────────


def _is_implied_by(clause, learned_cl, variables) -> bool:
    """Check if a single clause is logically implied by the learned constraints.

    Returns True if learned_cl |= clause (no assignment satisfies learned but violates clause).
    """
    m = cp.Model(learned_cl + [~clause])
    return not m.solve()


def analyze_missing(result_path: str, model_override: str = None) -> dict:
    """Analyze which target clauses are missing and whether they're in the bias."""
    result_path = Path(result_path)
    with open(result_path) as f:
        result = json.load(f)

    if result.get("error"):
        return {"file": str(result_path), "status": "error", "error": result["error"]}

    # Check verification status
    v = result.get("verification", {})
    flat_eq = v.get("equivalent", False)
    has_fp = v.get("has_false_positives", False)
    has_fn = v.get("has_false_negatives", False)

    model_path = model_override or result["model"]
    if not Path(model_path).exists():
        return {"file": str(result_path), "status": "model_not_found"}

    feature_names = result.get("feature_names") or extract_feature_names(model_path)
    variables = [cp.boolvar(name=f) for f in feature_names]

    # Rebuild target and learned
    target_cl = extract_target_constraints(model_path, variables, feature_names)

    from diagnostics.refine_from_json import parse_constraints
    learned_cl = parse_constraints(result.get("constraints", []), variables)

    # Find missing target clauses
    missing = []
    for i, t in enumerate(target_cl):
        if not _is_implied_by(t, learned_cl, variables):
            info = _classify_clause(t)
            coverage = check_bias_coverage(info, feature_names)
            missing.append({
                "index": i,
                "clause": str(t),
                **info,
                **coverage,
            })

    # Classify missing by type and bias coverage
    by_type = Counter(m["type"] for m in missing)
    in_bias = sum(1 for m in missing if m["in_any_bias"])
    not_in_bias = sum(1 for m in missing if not m["in_any_bias"])

    # Also check: which learned constraints are NOT implied by target (extra)?
    extra = []
    for i, c in enumerate(learned_cl):
        try:
            neg = ~c
        except Exception:
            continue
        m = cp.Model(target_cl + [neg])
        if m.solve():
            extra.append({"index": i, "constraint": str(c)})

    return {
        "file": str(result_path),
        "model": model_path,
        "status": "ok",
        "features": len(feature_names),
        "n_target": len(target_cl),
        "n_learned": len(learned_cl),
        "flat_equivalent": flat_eq,
        "has_fp": has_fp,
        "has_fn": has_fn,
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_by_type": dict(by_type),
        "n_in_bias": in_bias,
        "n_not_in_bias": not_in_bias,
        "missing": missing,
        "extra": extra,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────


def collect_json(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.json")))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Result JSON(s) or directories")
    parser.add_argument("--model", default=None, help="Override UVL model path")
    parser.add_argument("--summary", action="store_true", help="Compact summary")
    parser.add_argument("--only-fn", action="store_true",
                        help="Only analyze FN-only models (no FP)")
    parser.add_argument("--only-non-eq", action="store_true",
                        help="Only analyze non-equivalent models")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of models to analyze")
    args = parser.parse_args()

    json_files = collect_json(args.paths)
    if not json_files:
        parser.error("No JSON files found")

    # Pre-filter by verification status if requested
    filtered = []
    for jf in json_files:
        with open(jf) as f:
            r = json.load(f)
        if r.get("error"):
            continue
        v = r.get("verification", {})
        if args.only_fn:
            if not (not v.get("has_false_positives") and v.get("has_false_negatives")):
                continue
        elif args.only_non_eq:
            if v.get("equivalent"):
                continue
        filtered.append(jf)

    if args.limit:
        filtered = filtered[:args.limit]

    print(f"Analyzing {len(filtered)} result(s)...\n")

    # Aggregate stats
    all_results = []
    total_missing_by_type = Counter()
    total_in_bias = 0
    total_not_in_bias = 0
    total_missing = 0
    completeness_sizes = []  # track children count for completeness clauses

    for i, jf in enumerate(filtered):
        if not args.summary:
            print(f"[{i+1}/{len(filtered)}] {jf.name}")

        diag = analyze_missing(str(jf), model_override=args.model)
        all_results.append(diag)

        if diag["status"] != "ok":
            if not args.summary:
                print(f"  status: {diag['status']}")
            continue

        total_missing += diag["n_missing"]
        total_in_bias += diag["n_in_bias"]
        total_not_in_bias += diag["n_not_in_bias"]

        for t, c in diag["missing_by_type"].items():
            total_missing_by_type[t] += c

        # Track completeness clause sizes
        for m in diag["missing"]:
            if m["type"] == "completeness":
                n_children = sum(1 for _, pos in m["literals"] if pos)
                completeness_sizes.append(n_children)

        if args.summary:
            in_b = diag["n_in_bias"]
            not_b = diag["n_not_in_bias"]
            types = diag["missing_by_type"]
            type_str = ", ".join(f"{t}={c}" for t, c in sorted(types.items()))
            print(
                f"  {jf.stem:50s}  miss={diag['n_missing']:3d}  "
                f"in_bias={in_b:3d}  not_in_bias={not_b:3d}  "
                f"FP={'Y' if diag['has_fp'] else 'N'} FN={'Y' if diag['has_fn'] else 'N'}  "
                f"[{type_str}]"
            )
        elif args.verbose:
            print(f"  features={diag['features']}, target={diag['n_target']}, "
                  f"learned={diag['n_learned']}")
            print(f"  missing={diag['n_missing']} (in_bias={diag['n_in_bias']}, "
                  f"not_in_bias={diag['n_not_in_bias']})")
            print(f"  extra={diag['n_extra']}")
            print(f"  types: {dict(diag['missing_by_type'])}")
            for m in diag["missing"]:
                bias_mark = "BIAS" if m["in_any_bias"] else "MISS"
                print(f"    [{bias_mark}] {m['type']:15s}  {m['clause']}")
                if not m["in_any_bias"]:
                    print(f"           reason: {m['reason']}")
            if diag["extra"]:
                print(f"  extra learned (not implied by target):")
                for e in diag["extra"]:
                    print(f"    [{e['index']:3d}] {e['constraint']}")
            print()

    # ── Deep analysis: are missing completeness parents in the inferred tree? ──
    parent_in_tree = 0
    parent_not_in_tree = 0
    children_match = 0
    children_partial = 0
    children_mismatch = 0

    # Track what group type the tree already assigned
    group_type_of_missing = Counter()  # what gtype did the tree give this group?
    missing_multi_group = 0  # parent has children spread across multiple groups

    for r in all_results:
        if r["status"] != "ok":
            continue
        result_json = json.load(open(r["file"]))
        tree = result_json.get("inferred_tree", {})

        # Per-group detail
        tree_groups = {}  # parent -> [(gtype, set(children))]
        tree_children_of = {}
        for parent, groups in tree.items():
            tree_groups[parent] = [(g[0], set(g[1])) for g in groups]
            all_ch = set()
            for g in groups:
                all_ch.update(g[1])
            tree_children_of[parent] = all_ch

        for m in r.get("missing", []):
            if m["type"] != "completeness":
                continue
            neg_lits = [name for name, pos in m["literals"] if not pos]
            pos_lits = [name for name, pos in m["literals"] if pos]
            if len(neg_lits) != 1:
                continue
            parent = neg_lits[0]
            expected_children = set(pos_lits)

            if parent in tree_children_of:
                parent_in_tree += 1
                actual = tree_children_of[parent]
                if expected_children <= actual:
                    children_match += 1
                elif expected_children & actual:
                    children_partial += 1
                else:
                    children_mismatch += 1

                # What group type covers these children?
                for gtype, gchildren in tree_groups.get(parent, []):
                    if expected_children <= gchildren or (expected_children & gchildren):
                        group_type_of_missing[gtype] += 1
                        break
                else:
                    # Children spread across multiple groups
                    missing_multi_group += 1
            else:
                parent_not_in_tree += 1

    print(f"\n  Deep analysis of missing completeness clauses:")
    print(f"    Parent IS in inferred tree:    {parent_in_tree}")
    print(f"      Children fully match:        {children_match}")
    print(f"      Children partial match:      {children_partial}")
    print(f"      Children no match:           {children_mismatch}")
    print(f"    Parent NOT in inferred tree:   {parent_not_in_tree}")
    print(f"    Group type assigned to missing completeness:")
    for gt, c in group_type_of_missing.most_common():
        print(f"      {gt:15s}: {c}")
    print(f"    Children across multiple groups: {missing_multi_group}")

    # ── Aggregate summary ──────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "ok"]
    print(f"\n{'=' * 70}")
    print(f"AGGREGATE SUMMARY ({len(ok)} models analyzed)")
    print(f"{'=' * 70}")
    print(f"Total missing target clauses: {total_missing}")
    print(f"  In bias (CA should learn):  {total_in_bias}")
    print(f"  NOT in bias (can't learn):  {total_not_in_bias}")
    print(f"\nMissing by clause type:")
    for t, c in total_missing_by_type.most_common():
        print(f"  {t:20s}: {c:5d}")

    if completeness_sizes:
        from collections import Counter as C
        size_dist = C(completeness_sizes)
        print(f"\nCompleteness clause children count distribution:")
        for size, count in sorted(size_dist.items()):
            in_bias = "IN BIAS" if size <= 3 else "NOT IN BIAS"
            print(f"  {size} children: {count:4d}  ({in_bias})")

    # How many models have ALL missing in bias vs some not?
    all_in = sum(1 for r in ok if r["n_missing"] > 0 and r["n_not_in_bias"] == 0)
    some_out = sum(1 for r in ok if r["n_not_in_bias"] > 0)
    no_missing = sum(1 for r in ok if r["n_missing"] == 0)
    print(f"\nPer-model breakdown:")
    print(f"  No missing clauses:             {no_missing}")
    print(f"  All missing in bias:            {all_in}")
    print(f"  Some missing NOT in bias:       {some_out}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
