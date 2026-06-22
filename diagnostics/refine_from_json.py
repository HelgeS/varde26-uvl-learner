"""
Re-run tree inference and refinement from a saved JSON result file.

Reconstructs CPMpy constraints from the JSON's string representations,
rebuilds the oracle from the ground-truth UVL file, and re-runs the
post-CA pipeline (infer_tree → refine_tree_groups → verify → export).

This avoids the costly CA process and lets you iterate on the heuristics.

Usage:
    # Re-run refinement on a single result
    python -m diagnostics.refine_from_json results/REAL-FM-5.json --verify --export-uvl dbg

    # Re-run on all results in a directory
    python -m diagnostics.refine_from_json results/ --verify --out-dir results_v2/

    # Dry run: just show verification without writing anything
    python -m diagnostics.refine_from_json results/REAL-FM-5.json --verify
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import cpmpy as cp
from pycona import ConstraintOracle

from uvl_learner.oracle import extract_feature_names, extract_target_constraints
from uvl_learner.io import save_result, export_learned_to_uvl
from runners.pairwise import print_result
from uvl_learner.reconstruct.tree import (
    infer_tree,
    _validate_tree,
    _fix_multi_parent_tree,
)
from uvl_learner.reconstruct.cleanup import cleanup_constraints, cleanup_dumb
from uvl_learner.reconstruct.refine import infer_and_refine_tree
from uvl_learner.reconstruct.extract import constraints_from_tree
from diagnostics.report import deep_analysis
from uvl_learner.verify import verify_learned

log = logging.getLogger("refine_from_json")


# ── Constraint string parser ──────────────────────────────────────────────


def parse_constraints(
    constraint_strs: list[str],
    variables: list,
    *,
    count: int | None = None,
) -> list:
    """Parse constraint strings (from JSON) back into CPMpy expressions.

    Parameters
    ----------
    constraint_strs : list of constraint string representations
    variables : list of CPMpy BoolVars (matching feature_names order)
    count : if given, parse only the first *count* constraints

    Returns
    -------
    list of CPMpy constraint expressions
    """
    var_by_name = {v.name: v for v in variables}

    strs = constraint_strs[:count] if count is not None else constraint_strs
    constraints = []
    for s in strs:
        c = _parse_one(s.strip(), var_by_name)
        if c is not None:
            constraints.append(c)
        else:
            log.warning("  Could not parse constraint: %s", s)
    return constraints


def _parse_one(s: str, var_by_name: dict):
    """Parse a single constraint string into a CPMpy expression."""
    # Strip outer whitespace
    s = s.strip()

    # Unary: just a feature name (core constraint)
    if s in var_by_name:
        return var_by_name[s]

    # Equivalence: (A) == (B)
    m = re.match(r"^\((.+?)\)\s*==\s*\((.+?)\)$", s)
    if m:
        a = _parse_one(m.group(1).strip(), var_by_name)
        b = _parse_one(m.group(2).strip(), var_by_name)
        if a is not None and b is not None:
            return a == b

    # Implication: (A) -> (B)
    m = re.match(r"^\((.+?)\)\s*->\s*\((.+)\)$", s)
    if m:
        lhs = _parse_one(m.group(1).strip(), var_by_name)
        rhs_str = m.group(2).strip()
        rhs = _parse_rhs(rhs_str, var_by_name)
        if lhs is not None and rhs is not None:
            return lhs.implies(rhs)

    return None


def _parse_rhs(s: str, var_by_name: dict):
    """Parse the RHS of an implication — can be a var, negation, or compound."""
    s = s.strip()

    # Negated variable: ~name
    if s.startswith("~"):
        name = s[1:].strip()
        if name in var_by_name:
            return ~var_by_name[name]

    # Plain variable
    if s in var_by_name:
        return var_by_name[s]

    # or([name1, name2, ...]) — completeness clause
    m = re.match(r"^or\(\[(.+)\]\)$", s)
    if m:
        names = [n.strip() for n in m.group(1).split(",")]
        vars_ = [var_by_name[n] for n in names if n in var_by_name]
        if len(vars_) >= 2:
            return cp.any(vars_)

    # (name1) or (name2) — binary or
    m = re.match(r"^\((.+?)\)\s+or\s+\((.+?)\)$", s)
    if m:
        parts = [m.group(1).strip(), m.group(2).strip()]
        vars_ = [var_by_name[n] for n in parts if n in var_by_name]
        if len(vars_) == 2:
            return cp.any(vars_)

    # (name1) + (name2) + ... == 1 — exactly-one
    m = re.match(r"^(.+?)\s*==\s*(\d+)$", s)
    if m:
        expr_str = m.group(1).strip()
        val = int(m.group(2))
        parts = [p.strip().strip("()") for p in expr_str.split("+")]
        vars_ = [var_by_name[n] for n in parts if n.strip() in var_by_name]
        if vars_:
            return sum(vars_) == val

    return None


# ── Pipeline ───────────────────────────────────────────────────────────


def refine_from_json(
    json_path: str,
    *,
    verify: bool = False,
    export_uvl: str | None = None,
) -> dict:
    """Load a JSON result, re-run tree inference + refinement, return updated result."""
    json_path = Path(json_path)
    with open(json_path) as f:
        result = json.load(f)

    if result.get("error"):
        log.info("  Skipping %s (has error: %s)", json_path, result["error"])
        return result

    uvl_path = result["model"]
    if not Path(uvl_path).exists():
        log.warning("  Model file not found: %s", uvl_path)
        result["error"] = f"model file not found: {uvl_path}"
        return result

    feature_names = result["feature_names"]
    n_ca_learned = result["learned"] - result.get("refined", 0)

    t0 = time.monotonic()

    # 1. Reconstruct CPMpy variables
    variables = [cp.boolvar(name=f) for f in feature_names]

    # 2. Parse only the CA-learned constraints (exclude previously refined)
    learned = parse_constraints(result["constraints"], variables, count=n_ca_learned)
    log.info(
        "  Parsed %d / %d CA constraints from %s",
        len(learned),
        n_ca_learned,
        json_path.name,
    )

    # 3. Build oracle from ground-truth UVL
    target = extract_target_constraints(uvl_path, variables, feature_names)
    oracle = ConstraintOracle(target)


    # END SETUP



    # 7-9. Unified tree inference + refinement
    n_learned_pre_tree = len(learned)
    try:
        learned, inferred, n_tree_queries = infer_and_refine_tree(
            feature_names, variables, learned, oracle,
        )
    except Exception as e:
        if "Collapse" in str(e):
            log.warning("  Collapse in infer_and_refine_tree: %s", e)
            
            inferred = infer_tree(feature_names, variables, learned)
            n_tree_queries = 0
        else:
            raise
    n_refined = len(learned) - n_learned_pre_tree
    log.info("  Refined: +%d constraints, %d tree queries", n_refined, n_tree_queries)
    result["completeness_added"] = 0  # subsumed by unified pipeline
    result["queries_completeness"] = 0

    # 6. Update result
    result["learned"] = len(learned)
    result["refined"] = n_refined
    result["tree_queries"] = n_tree_queries

    # 7. Validate tree
    validation_errors = _validate_tree(feature_names, inferred)
    result["tree_validation"] = {
        "valid": len(validation_errors) == 0,
        "errors": validation_errors,
    }

    if validation_errors:
        log.warning("  Tree validation failed, applying fixes...")
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
        log.info("  Fixed tree: %d parent nodes", len(inferred))


    # 8. Reconstruct enhanced constraint model
    enhanced_cl, cross_tree_cl = constraints_from_tree(
        feature_names, variables, inferred, learned
    )

    # 12. Cleanup: remove spurious cross-tree constraints
    if True:
        t_cleanup = time.monotonic()
        # enhanced_cl, removed_cl, n_cleanup_queries = cleanup_constraints(
        #     feature_names, variables, learned, inferred, oracle
        # )
        enhanced_cl, removed_cl, n_cleanup_queries = cleanup_dumb(
            feature_names, variables, learned, inferred, oracle
        )
        
        result["cleanup_removed"] = len(removed_cl)
        result["cleanup_removed_constraints"] = [str(c) for c in removed_cl]
        result["queries_cleanup"] = n_cleanup_queries
        # result["queries_total"] = (
        #     metrics.total_queries
        #     + n_completeness_queries
        #     + n_tree_queries
        #     + n_cleanup_queries
        # )
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

    # 9. Verify
    if verify:
        result["verification"] = verify_learned(learned, target, variables)
        eq_flat = result["verification"]["equivalent"]
        log.info("  verification (flat): equivalent=%s", eq_flat)
        result["verification_tree"] = verify_learned(enhanced_cl, target, variables)
        eq_tree = result["verification_tree"]["equivalent"]
        log.info("  verification (tree): equivalent=%s", eq_tree)

    # 10. Export UVL
    if export_uvl:
        out_path = Path(export_uvl)
        if out_path.is_dir():
            stem = Path(uvl_path).stem
            out_path = out_path / f"{stem}_learned.uvl"
        exported, skipped_ex = export_learned_to_uvl(
            feature_names, inferred, cross_tree_cl, str(out_path)
        )
        result["exported_uvl"] = str(out_path)
        log.info(
            "  exported to %s (%d constraints, %d skipped)",
            out_path,
            exported,
            skipped_ex,
        )

    result["time_refine"] = round(time.monotonic() - t0, 4)

    result.setdefault("_name", Path(result["model"]).stem)
    result.setdefault("_file", result["model"])
    deep_analysis([result])

    return result


# ── CLI ────────────────────────────────────────────────────────────────


def collect_json_paths(paths: list[str]) -> list[Path]:
    """Expand CLI args: files kept, directories globbed for *.json."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            log.warning("Skipping %s", p)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Re-run tree inference/refinement from saved JSON results",
    )
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        help="JSON result file(s) or directories to scan",
        default=[]
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run SAT equivalence check after refinement",
        default=True
    )
    # parser.add_argument(
    #     "--export-uvl",
    #     type=str,
    #     default="dbg",
    #     metavar="PATH",
    #     help="Export learned model to UVL (dir → {stem}_learned.uvl)",
    # )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Write updated JSON files into this directory",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-model output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
        default=True,
    )

    args = parser.parse_args()

    if len(args.paths) == 0:
        # args.paths.append("dbg/debug_challenge.json")
        args.paths.append("reference/model_20120915_487659597.json")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    json_files = collect_json_paths(args.paths)
    if not json_files:
        parser.error("No JSON files found")

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Found %d JSON result(s)", len(json_files))

    results = []
    for i, jp in enumerate(json_files, 1):
        log.info("[%d/%d] %s", i, len(json_files), jp)
        uvl_export = Path(jp).parent
        r = refine_from_json(
            str(jp),
            verify=args.verify,
            export_uvl=uvl_export,
        )
        results.append(r)

        if not args.quiet:
            print_result(r)

        if out_dir is not None:
            save_result(r, out_dir / jp.name)


if __name__ == "__main__":
    main()
