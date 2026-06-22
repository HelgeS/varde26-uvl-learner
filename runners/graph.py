"""Runner: graph-bias variant.

Uses the custom graph/conjunction-aware CA components in uvl_learner.bias.graph
(own FindScope/FindC/QGen built on networkx) for richer n-ary candidate
generation.

Usage:
    uv run python -m runners.graph --generate-example --verify
    uv run python -m runners.graph models/aircraft_fm.uvl --verify
"""

import argparse
import pickle
import signal
import time
from pathlib import Path

import cpmpy as cp
from pycona import ProblemInstance, ConstraintOracle, Metrics

from uvl_learner.oracle import extract_feature_names, extract_target_constraints
from uvl_learner.io import (
    TimeoutError,
    _timeout_handler,
    save_result,
    collect_uvl_paths,
    export_learned_to_uvl,
)
from uvl_learner.verify import verify_learned
from uvl_learner.acquire import ALGORITHMS
from uvl_learner.bias.pairwise import build_bias
from uvl_learner.bias.graph import GraphBias, GQuAcq, TrackAndCacheCAEnv
from uvl_learner.reconstruct.refine import infer_and_refine_tree
from uvl_learner.reconstruct.tree import (
    infer_tree,
    _validate_tree,
    _fix_multi_parent_tree,
)
from uvl_learner.reconstruct.extract import constraints_from_tree
from uvl_learner.reconstruct.cleanup import cleanup_dumb


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
        from diagnostics.report import deep_analysis

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
