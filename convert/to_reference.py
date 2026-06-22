"""
Extract pairwise constraints from a UVL model as a reference JSON.

Parses the UVL's SAT/CNF encoding, filters to binary and unit clauses
(the subset learnable by a flat pairwise CA bias), and saves in the
JSON format expected by refine_from_json.py.

This allows testing tree inference and refinement independently of
the costly CA process.

Usage:
    # Single model
    python -m convert.to_reference models/aircraft_fm.uvl

    # Directory (all *.uvl files)
    python -m convert.to_reference models/ --max-features 50

    # Then refine from the extracted reference
    python -m diagnostics.refine_from_json reference/aircraft_fm.json --verify
"""

import argparse
import json
import logging
import time
from pathlib import Path

from flamapy.core.discover import DiscoverMetamodels
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel

from uvl_learner.oracle import extract_feature_names

log = logging.getLogger("extract_reference")


def extract_pairwise_from_uvl(uvl_path: str) -> dict:
    """Extract pairwise constraints from a UVL model.

    Returns a result dict compatible with refine_from_json.py.
    """
    uvl_path = str(uvl_path)
    t0 = time.monotonic()

    # 1. Extract feature names
    feature_names = extract_feature_names(uvl_path)

    # 2. Get CNF from flamapy
    fm = FLAMAFeatureModel(uvl_path)
    fm._transform_to_sat()
    sat = fm.sat_model

    # Map SAT variable IDs to feature names
    id_to_name = {}
    for name, sat_id in sat.variables.items():
        if name in set(feature_names):
            id_to_name[sat_id] = name

    # 3. Extract binary and unit clauses
    all_clauses = sat.get_all_clauses().clauses
    n_cnf = len(all_clauses)

    # Track implications for equivalence detection
    # implications[a][b] = True means we have (a) -> (b)
    implications: dict[str, set[str]] = {}
    # exclusions: (a) -> (~b) — only store one direction (sorted)
    exclusions: set[tuple[str, str]] = set()
    # core features (unit positive clauses)
    core: list[str] = []
    n_skipped = 0

    for clause in all_clauses:
        # Only process clauses where all literals map to known features
        resolved = []
        for lit in clause:
            var_id = abs(lit)
            if var_id not in id_to_name:
                break
            resolved.append((id_to_name[var_id], lit > 0))
        else:
            if len(resolved) == 1:
                name, positive = resolved[0]
                if positive:
                    core.append(name)
                # Skip dead features (negative unit) — not in pairwise bias
            elif len(resolved) == 2:
                (a, a_pos), (b, b_pos) = resolved
                if not a_pos and b_pos:
                    # [-a, +b] → a -> b
                    implications.setdefault(a, set()).add(b)
                elif a_pos and not b_pos:
                    # [+a, -b] → b -> a
                    implications.setdefault(b, set()).add(a)
                elif not a_pos and not b_pos:
                    # [-a, -b] → a -> ~b (mutual exclusion)
                    exclusions.add((a, b) if a <= b else (b, a))
                # [+a, +b] → skip (not representable with positive LHS)
            else:
                n_skipped += 1
                continue

    # 4. Detect equivalences: a->b AND b->a → a == b
    equivalences: list[tuple[str, str]] = []
    remaining_implications: list[tuple[str, str]] = []

    for a, targets in sorted(implications.items()):
        for b in sorted(targets):
            if b in implications and a in implications[b]:
                # Equivalence — only emit once (alphabetical order)
                if a < b:
                    equivalences.append((a, b))
            else:
                remaining_implications.append((a, b))

    # 5. Build constraint strings
    constraint_strs = []

    # Core features first
    for name in core:
        constraint_strs.append(name)

    # Equivalences
    for a, b in equivalences:
        constraint_strs.append(f"({a}) == ({b})")

    # Implications
    for a, b in remaining_implications:
        constraint_strs.append(f"({a}) -> ({b})")

    # Exclusions
    for a, b in sorted(exclusions):
        constraint_strs.append(f"({a}) -> (~{b})")

    elapsed = round(time.monotonic() - t0, 4)

    log.info(
        "  %s: %d features, %d CNF clauses, %d pairwise (%d core, %d equiv, %d impl, %d excl), %d skipped n-ary",
        Path(uvl_path).stem,
        len(feature_names),
        n_cnf,
        len(constraint_strs),
        len(core),
        len(equivalences),
        len(remaining_implications),
        len(exclusions),
        n_skipped,
    )

    return {
        "model": uvl_path,
        "algorithm": "reference-extraction",
        "error": None,
        "features": len(feature_names),
        "feature_names": feature_names,
        "cnf_clauses": n_cnf,
        "cnf_skipped_nary": n_skipped,
        "bias_size": 0,
        "group_bias_max": 0,
        "time_bias": 0,
        "time_ca": 0,
        "queries_positive": 0,
        "queries_negative": 0,
        "n_skipped_collapses": 0,
        "converged": True,
        "queries_total": 0,
        "queries_membership": 0,
        "queries_recommendation": 0,
        "queries_generalization": 0,
        "time_ca_internal": 0,
        "learned": len(constraint_strs),
        "refined": 0,
        "constraints_learned": constraint_strs,
        "constraints": constraint_strs,
        "time_total": elapsed,
    }


# ── CLI ────────────────────────────────────────────────────────────────


def collect_uvl_paths(paths: list[str]) -> list[Path]:
    """Expand CLI args: files kept, directories globbed for *.uvl."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".uvl":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.uvl")))
        else:
            log.warning("Skipping %s", p)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract pairwise constraints from UVL models as reference JSON",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="UVL file(s) or directories to scan",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reference",
        metavar="DIR",
        help="Output directory for JSON files (default: reference/)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        metavar="N",
        help="Skip models with more than N features",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    uvl_files = collect_uvl_paths(args.paths)
    if not uvl_files:
        parser.error("No UVL files found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Found %d UVL file(s), output → %s/", len(uvl_files), out_dir)

    for i, uvl_path in enumerate(uvl_files, 1):
        log.info("[%d/%d] %s", i, len(uvl_files), uvl_path)

        if args.max_features:
            try:
                names = extract_feature_names(str(uvl_path))
                if len(names) > args.max_features:
                    log.info("  Skipping (%d features > %d)", len(names), args.max_features)
                    continue
            except Exception as e:
                log.warning("  Error reading features: %s", e)
                continue

        try:
            result = extract_pairwise_from_uvl(str(uvl_path))
        except Exception as e:
            log.warning("  Error: %s", e)
            result = {
                "model": str(uvl_path),
                "algorithm": "reference-extraction",
                "error": str(e),
            }

        out_path = out_dir / f"{uvl_path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("  Wrote %s", out_path)


if __name__ == "__main__":
    main()
