"""Convert UVL feature models into the Candy custom format.

For each input UVL model, two files are written into the output directory:
  - <name>.bias    — header with nbVars, domainSize, and the Gamma alphabet.
  - <name>.target  — one constraint per line, derived from the SAT/CNF encoding.

Constraint encoding (features are 0-indexed; "-i" denotes the negated literal):
  - 2-literal clause (one negative, one positive)  →  ImplXY a b   (a => b)
  - any other clause                                →  Any l1 l2 ...

Usage:
  uv run python3 uvl_to_candy.py models/sandwich.uvl models/aircraft_fm.uvl --out-dir candy/
  uv run python3 uvl_to_candy.py --small --out-dir candy/
"""

import argparse
from pathlib import Path

from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel

from ca_common import extract_feature_names

GAMMA = ["ImplXY", "Alternative", "Any"]


def _format_literal(feature_idx: int, negated: bool) -> str:
    return f"-{feature_idx}" if negated else str(feature_idx)


def convert_clauses_to_candy(uvl_path: Path) -> tuple[list[str], list[str]]:
    """Return (feature_names, target_lines) for a UVL model."""
    feature_names = extract_feature_names(str(uvl_path))
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    fm = FLAMAFeatureModel(str(uvl_path))
    fm._transform_to_sat()
    sat = fm.sat_model
    sat_id_to_idx = {sat_id: name_to_idx[name] for name, sat_id in sat.variables.items() if name in name_to_idx}

    target_lines: list[str] = []
    for clause in sat.get_all_clauses().clauses:
        literals = []
        skip = False
        for lit in clause:
            sat_id = abs(lit)
            if sat_id not in sat_id_to_idx:
                skip = True
                break
            literals.append((sat_id_to_idx[sat_id], lit < 0))
        if skip:
            continue

        if len(literals) == 2:
            (a_idx, a_neg), (b_idx, b_neg) = literals
            if a_neg and not b_neg:
                target_lines.append(f"ImplXY {a_idx} {b_idx}")
                continue
            if b_neg and not a_neg:
                target_lines.append(f"ImplXY {b_idx} {a_idx}")
                continue

        tokens = [_format_literal(idx, neg) for idx, neg in literals]
        target_lines.append("Any " + " ".join(tokens))

    return feature_names, target_lines


def write_candy(uvl_path: Path, out_dir: Path) -> None:
    feature_names, target_lines = convert_clauses_to_candy(uvl_path)
    n_vars = len(feature_names)
    stem = uvl_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    bias_path = out_dir / f"{stem}.bias"
    target_path = out_dir / f"{stem}.target"

    bias_lines = [
        f"nbVars {n_vars}",
        "domainSize 0 1",
        "",
        "Gamma",
        *GAMMA,
        "",
        "Features",
        *feature_names,
        "",
    ]
    bias_path.write_text("\n".join(bias_lines))
    target_path.write_text("\n".join(target_lines) + "\n")
    print(f"  wrote {bias_path} and {target_path} ({n_vars} vars, {len(target_lines)} clauses)")


SMALL_DEFAULTS = [
    "models/sandwich.uvl",
    "models/aircraft_fm.uvl",
    "models/movies_app_fm.uvl",
    "models/stack_fm.uvl",
    "models/connector_fm.uvl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UVL models to the Candy format.")
    parser.add_argument("paths", nargs="*", help="UVL files to convert")
    parser.add_argument("--out-dir", default="candy", help="Output directory (default: candy)")
    parser.add_argument(
        "--small", action="store_true", help="Convert a hard-coded set of small example models",
    )
    args = parser.parse_args()

    paths = list(args.paths)
    if args.small or not paths:
        paths.extend(SMALL_DEFAULTS)

    out_dir = Path(args.out_dir)
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"Skipping {p} (not found)")
            continue
        print(f"Converting {p}")
        write_candy(p, out_dir)


if __name__ == "__main__":
    main()
