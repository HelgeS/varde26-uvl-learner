"""Generate a comprehensive diagnosis report for all CA results.

Combines the analysis from diagnose_missing.py, diagnose_underconstraining.py,
and analyze_results.py into a single report. Reads only the fast JSON metadata
(no SAT solving) for the overview, then optionally runs deeper SAT-based
diagnosis on selected models.

Usage:
    # Fast overview from JSON metadata only (no SAT solving)
    python report_results.py results/

    # Full SAT-based diagnosis (slow — checks every constraint)
    python report_results.py results/ --deep

    # Deep diagnosis on named models only
    python report_results.py results/ --deep --named-only

    # Write report to file
    python report_results.py results/ --deep -o report.txt
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# ── Lazy imports for deep mode ──
# These pull in cpmpy + flamapy and are slow to import, so defer them.
_deep_imports_done = False


def _ensure_deep_imports():
    global _deep_imports_done
    if _deep_imports_done:
        return
    global cp, extract_feature_names, extract_target_constraints
    global rebuild_learned_constraints, find_missing_constraints
    global _classify_clause, check_bias_coverage
    global constraints_from_tree, _rebuild_tree_info
    import cpmpy as cp
    from ca_common import extract_feature_names, extract_target_constraints
    from tree_inference import constraints_from_tree
    from diagnose_missing import rebuild_learned_constraints, find_missing_constraints, _rebuild_tree_info
    from diagnose_underconstraining import _classify_clause, check_bias_coverage
    _deep_imports_done = True


# ── JSON-only fast analysis ───────────────────────────────────────────


def load_results(paths: list[str]) -> list[dict]:
    """Load all JSON result files from paths (files or directories)."""
    json_files = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json":
            json_files.append(p)
        elif p.is_dir():
            json_files.extend(sorted(p.glob("*.json")))
    results = []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        d["_file"] = str(jf)
        d["_name"] = jf.stem
        results.append(d)
    return results


NAMED_PREFIXES = ("REAL-FM-", "DELL", "aircraft", "connector", "fame", "smart_home", "stack", "movies")


def is_named(r: dict) -> bool:
    return any(r["_name"].startswith(p) for p in NAMED_PREFIXES)


def fast_report(results: list[dict], out=sys.stdout):
    """Print overview report using only JSON metadata (no SAT solving)."""
    p = lambda *a, **kw: print(*a, **kw, file=out)

    n = len(results)
    errors = [r for r in results if r.get("error")]
    converged = [r for r in results if r.get("converged") and not r.get("error")]

    p(f"{'=' * 78}")
    p(f"  CA Results Report — {n} models")
    p(f"{'=' * 78}")
    p()

    # ── Overview ──
    p(f"Total results:    {n}")
    p(f"Converged:        {len(converged)}")
    p(f"Errors:           {len(errors)}")
    if errors:
        p()
        for r in errors[:10]:
            p(f"  {r['_name']}: {r['error']}")
        if len(errors) > 10:
            p(f"  ... and {len(errors) - 10} more")
    p()

    # ── Verification overview ──
    v_flat = {"eq": 0, "neq": 0}
    v_tree = {"eq": 0, "neq": 0, "none": 0}
    for r in converged:
        v = r.get("verification", {})
        vt = r.get("verification_tree", {})
        if v.get("equivalent"):
            v_flat["eq"] += 1
        else:
            v_flat["neq"] += 1
        if vt:
            if vt.get("equivalent"):
                v_tree["eq"] += 1
            else:
                v_tree["neq"] += 1
        else:
            v_tree["none"] += 1

    p(f"{'─' * 78}")
    p(f"  Equivalence Verification")
    p(f"{'─' * 78}")
    p(f"{'':30s}  {'Flat (CA only)':>14s}  {'Tree-enhanced':>14s}")
    p(f"  {'Equivalent (EQ)':28s}  {v_flat['eq']:>6d} ({100*v_flat['eq']/max(len(converged),1):.1f}%)  "
      f"{v_tree['eq']:>6d} ({100*v_tree['eq']/max(len(converged),1):.1f}%)")
    p(f"  {'Not equivalent (NEQ)':28s}  {v_flat['neq']:>6d} ({100*v_flat['neq']/max(len(converged),1):.1f}%)  "
      f"{v_tree['neq']:>6d} ({100*v_tree['neq']/max(len(converged),1):.1f}%)")
    tree_recovered = v_tree["eq"] - v_flat["eq"]
    if tree_recovered > 0:
        p(f"\n  Tree refinement recovers {tree_recovered} additional models to equivalence.")
    p()

    # ── FP/FN breakdown ──
    fp_fn = {"both": 0, "fp_only": 0, "fn_only": 0, "neither": 0}
    for r in converged:
        v = r.get("verification", {})
        has_fp = v.get("has_false_positives", False)
        has_fn = v.get("has_false_negatives", False)
        if has_fp and has_fn:
            fp_fn["both"] += 1
        elif has_fp:
            fp_fn["fp_only"] += 1
        elif has_fn:
            fp_fn["fn_only"] += 1
        else:
            fp_fn["neither"] += 1

    p(f"{'─' * 78}")
    p(f"  False Positive / False Negative Breakdown (flat)")
    p(f"{'─' * 78}")
    p(f"  Perfect (no FP, no FN):         {fp_fn['neither']:>6d}")
    p(f"  FP only (extra, nothing miss):  {fp_fn['fp_only']:>6d}")
    p(f"  FN only (missing, nothing xtra):{fp_fn['fn_only']:>6d}")
    p(f"  Both FP + FN:                   {fp_fn['both']:>6d}")
    p()

    # ── Size statistics ──
    def _stats(vals):
        if not vals:
            return {}
        return {
            "n": len(vals), "mean": statistics.mean(vals),
            "median": statistics.median(vals), "std": statistics.pstdev(vals),
            "min": min(vals), "max": max(vals),
            "p25": sorted(vals)[len(vals)//4],
            "p75": sorted(vals)[3*len(vals)//4],
            "p95": sorted(vals)[int(0.95*len(vals))],
        }

    features = [r["features"] for r in converged]
    cnf = [r["cnf_clauses"] for r in converged]
    bias = [r["bias_size"] for r in converged]
    time_ca = [r["time_ca"] for r in converged]
    queries = [r["queries_total"] for r in converged]

    p(f"{'─' * 78}")
    p(f"  Aggregate Statistics")
    p(f"{'─' * 78}")
    cols = [("Features", features), ("CNF Clauses", cnf), ("Bias Size", bias),
            ("Time CA (s)", time_ca), ("Queries", queries)]
    header = f"  {'Stat':<8s}" + "".join(f"{name:>14s}" for name, _ in cols)
    p(header)
    for stat_name in ["mean", "median", "std", "min", "p25", "p75", "p95", "max"]:
        row = f"  {stat_name:<8s}"
        for _, vals in cols:
            s = _stats(vals)
            row += f"{s.get(stat_name, 0):>14.1f}"
        p(row)
    p()

    # ── Size vs equivalence ──
    eq_feats = [r["features"] for r in converged if r.get("verification_tree", {}).get("equivalent")]
    neq_feats = [r["features"] for r in converged if not r.get("verification_tree", {}).get("equivalent")]
    if eq_feats and neq_feats:
        p(f"{'─' * 78}")
        p(f"  Size vs Equivalence (tree)")
        p(f"{'─' * 78}")
        p(f"  EQ models:  n={len(eq_feats)}, mean={statistics.mean(eq_feats):.1f}, "
          f"median={statistics.median(eq_feats):.0f}, max={max(eq_feats)}")
        p(f"  NEQ models: n={len(neq_feats)}, mean={statistics.mean(neq_feats):.1f}, "
          f"median={statistics.median(neq_feats):.0f}, max={max(neq_feats)}")
        p()

    # ── Named model results ──
    named = [r for r in converged if is_named(r)]
    if named:
        p(f"{'─' * 78}")
        p(f"  Named Model Results")
        p(f"{'─' * 78}")
        named.sort(key=lambda r: r["_name"])
        for r in named:
            v = r.get("verification", {})
            vt = r.get("verification_tree", {})
            flat_eq = "EQ" if v.get("equivalent") else "NEQ"
            tree_eq = "EQ" if vt.get("equivalent") else "NEQ"
            fp = "Y" if v.get("has_false_positives") else "N"
            fn = "Y" if v.get("has_false_negatives") else "N"
            p(f"  {r['_name']:35s}  flat:{flat_eq:3s}  tree:{tree_eq:3s}  "
              f"FP={fp} FN={fn}  feat={r['features']:3d}  "
              f"cnf={r['cnf_clauses']:4d}  t={r['time_ca']:.1f}s")
        p()


# ── Deep SAT-based analysis ──────────────────────────────────────────


def deep_analysis(results: list[dict], out=sys.stdout):
    """Run SAT-based diagnosis on each model: find missing/extra constraints
    and classify them by type and bias coverage."""
    _ensure_deep_imports()
    p = lambda *a, **kw: print(*a, **kw, file=out)

    p(f"{'─' * 78}")
    p(f"  Deep Constraint Diagnosis (SAT-based)")
    p(f"{'─' * 78}")
    p()

    total_missing_by_type = Counter()
    total_in_bias = 0
    total_not_in_bias = 0
    total_missing = 0
    total_extra = 0
    completeness_sizes = []
    per_model = []  # (name, n_miss, n_extra, n_target, n_learned, by_type, in_bias, not_in_bias)
    n_eq = 0
    n_neq = 0
    n_err = 0

    for i, r in enumerate(results):
        name = r["_name"]
        fpath = r["_file"]
        sys.stderr.write(f"\r  [{i+1}/{len(results)}] {name:50s}")
        sys.stderr.flush()

        if r.get("error") or "constraints" not in r:
            n_err += 1
            continue

        model_path = r["model"]
        if not Path(model_path).exists():
            alt = Path(fpath).parent / model_path
            if alt.exists():
                model_path = str(alt)
            else:
                print(f"Model file {alt} does not exist... skip")
                n_err += 1
                continue

        feature_names = r.get("feature_names") or extract_feature_names(model_path)
        variables = [cp.boolvar(name=f) for f in feature_names]
        target_cl = extract_target_constraints(model_path, variables, feature_names)
        learned_cl, unparsed = rebuild_learned_constraints(r, variables)

        # Use tree-enhanced constraints when an inferred tree is available
        tree_info = _rebuild_tree_info(r)
        if tree_info:
            tree_cl, _ = constraints_from_tree(feature_names, variables, tree_info, learned_cl)
            effective_cl = tree_cl
        else:
            effective_cl = learned_cl

        # Find missing and extra
        diag = find_missing_constraints(effective_cl, target_cl, variables)
        n_miss = len(diag["missing_from_learned"])
        n_ext = len(diag["extra_in_learned"])

        if n_miss == 0 and n_ext == 0:
            n_eq += 1
        else:
            n_neq += 1

        total_missing += n_miss
        total_extra += n_ext

        # Classify missing
        by_type = Counter()
        in_bias = 0
        not_in_bias = 0
        print("Missing constraints:")
        for m in diag["missing_from_learned"]:
            # Re-find the target constraint for classification
            idx = m["index"]
            clause = target_cl[idx]
            info = _classify_clause(clause)
            cov = check_bias_coverage(info, feature_names)
            by_type[info["type"]] += 1
            total_missing_by_type[info["type"]] += 1
            if cov["in_any_bias"]:
                in_bias += 1
                total_in_bias += 1
            else:
                not_in_bias += 1
                total_not_in_bias += 1
            if info["type"] == "completeness":
                n_children = sum(1 for _, pos in info["literals"] if pos)
                completeness_sizes.append(n_children)

            print(f"  {m['constraint']}\t{info['type']}\tb:{cov['in_any_bias']}")

        print("Extra constraints:")
        for m in diag["extra_in_learned"]:
            print(f"  {m['constraint']}")

        per_model.append((name, n_miss, n_ext, len(target_cl), len(learned_cl),
                          dict(by_type), in_bias, not_in_bias))

    sys.stderr.write("\r" + " " * 70 + "\r")
    sys.stderr.flush()

    # ── Per-model table (NEQ only, sorted by missing desc) ──
    neq_models = [m for m in per_model if m[1] > 0 or m[2] > 0]
    neq_models.sort(key=lambda m: m[1], reverse=True)

    p(f"  {'Model':<40s}  {'miss':>5s}  {'extra':>5s}  {'target':>6s}  {'learned':>7s}  "
      f"{'in_bias':>7s}  {'!bias':>5s}  types")

    for name, n_miss, n_ext, n_target, n_learned, by_type, ib, nib in neq_models[:30]:
        type_str = ", ".join(f"{t}={c}" for t, c in sorted(by_type.items()))
        p(f"  {name:<40s}  {n_miss:>5d}  {n_ext:>5d}  {n_target:>6d}  {n_learned:>7d}  "
          f"{ib:>7d}  {nib:>5d}  {type_str}")
    if len(neq_models) > 30:
        p(f"  ... and {len(neq_models) - 30} more NEQ models")
    p()

    # ── Aggregate summary ──
    p(f"{'─' * 78}")
    p(f"  Deep Diagnosis Aggregate")
    p(f"{'─' * 78}")
    p(f"  Models analyzed:     {len(per_model)}")
    p(f"  Equivalent:          {n_eq}")
    p(f"  Not equivalent:      {n_neq}")
    p(f"  Errors/skipped:      {n_err}")
    p()
    p(f"  Total missing target clauses: {total_missing}")
    p(f"    In bias (CA should learn):  {total_in_bias}")
    p(f"    NOT in bias (can't learn):  {total_not_in_bias}")
    p(f"  Total extra learned:          {total_extra}")
    p()
    p(f"  Missing by constraint type:")
    for t, c in total_missing_by_type.most_common():
        pct = 100 * c / max(total_missing, 1)
        p(f"    {t:<20s}  {c:>6d}  ({pct:.1f}%)")
    p()

    if completeness_sizes:
        size_dist = Counter(completeness_sizes)
        p(f"  Completeness clause children count distribution:")
        for size, count in sorted(size_dist.items()):
            in_b = "IN BIAS" if size <= 3 else "NOT IN BIAS"
            p(f"    {size} children: {count:>4d}  ({in_b})")
        p()

    # ── Distribution of missing count ──
    miss_counts = [m[1] for m in per_model]
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 50), (51, float("inf"))]
    bucket_names = ["0", "1-2", "3-5", "6-10", "11-50", ">50"]
    p(f"  Distribution of missing constraints per model:")
    for (lo, hi), bname in zip(buckets, bucket_names):
        cnt = sum(1 for m in miss_counts if lo <= m <= hi)
        p(f"    miss={bname:<6s}  {cnt:>5d} models")
    p()


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Result JSON file(s) or directory")
    parser.add_argument("--deep", action="store_true",
                        help="Run SAT-based per-constraint diagnosis (slow)")
    parser.add_argument("--named-only", action="store_true",
                        help="With --deep, only diagnose named (non-synthetic) models")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Write report to file instead of stdout")
    args = parser.parse_args() #["dbg/debug_challenge.json", "--deep"])
    
    results = load_results(args.paths)
    if not results:
        parser.error("No JSON result files found")

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        fast_report(results, out=out)

        if args.deep:
            subset = [r for r in results if is_named(r)] if args.named_only else results
            # Filter to converged only
            subset = [r for r in subset if r.get("converged") and not r.get("error")]
            deep_analysis(subset, out=out)
    finally:
        if args.output:
            out.close()
            print(f"Report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
