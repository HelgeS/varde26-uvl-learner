"""Diagnose which target constraints are missing from learned results.

Given a result JSON (from ca_uvl_notree.py) and its source UVL model, this
script rebuilds both constraint sets and reports every target clause NOT
implied by the learned constraints, and every learned constraint NOT implied
by the target.

Usage:
    # Single result file (model path read from JSON)
    python diagnose_missing.py results/stack_fm.json

    # Explicit model override
    python diagnose_missing.py results/stack_fm.json --model models/stack_fm.uvl

    # Batch: all JSON files in a directory
    python diagnose_missing.py results/ --summary

    # Only show results with missing constraints
    python diagnose_missing.py results/ --only-missing
"""

import argparse
import json
import sys
from pathlib import Path

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from ca_common import extract_feature_names, extract_target_constraints
from tree_inference import constraints_from_tree


def _constraint_description(c) -> str:
    """Human-readable one-liner for a CPMpy constraint."""
    return str(c)


def _clause_to_feature_names(c) -> list[str]:
    """Extract feature names mentioned in a constraint."""
    names = []

    def _walk(expr):
        if isinstance(expr, NegBoolView):
            names.append(f"~{expr._bv.name}")
        elif isinstance(expr, _BoolVarImpl):
            names.append(expr.name)
        elif hasattr(expr, "args"):
            for a in expr.args:
                _walk(a)

    _walk(c)
    return names


def rebuild_learned_constraints(result: dict, variables: list):
    """Re-parse the string constraints from the result JSON into CPMpy objects.

    The JSON stores constraints as strings (e.g. "(A) -> (B)").  We rebuild
    them by matching variable names and parsing the structure.  For complex
    constraints that can't be trivially reconstructed, we fall back to the
    string representation for reporting.

    Returns a list of CPMpy constraint expressions.
    """
    var_of = {v.name: v for v in variables}

    learned_cl = []
    unparsed = []

    for s in result.get("constraints", []):
        c = _parse_constraint_str(s, var_of)
        if c is not None:
            learned_cl.append(c)
        else:
            unparsed.append(s)

    return learned_cl, unparsed


def _parse_constraint_str(s: str, var_of: dict):
    """Best-effort parse of a constraint string back to CPMpy.

    Handles the common forms produced by CPMpy's __str__:
      - "X"                    -> var (core)
      - "~X"                   -> ~var (dead)
      - "(X) -> (Y)"           -> X.implies(Y)
      - "(X) -> (~Y)"          -> X.implies(~Y)
      - "(X) == (Y)"           -> X == Y
      - "(X) -> (or(Y,Z,...))" -> X.implies(cp.any([Y,Z,...]))
      - "(X) -> ((Y + Z + ...) == 1)"  -> X.implies(sum == 1)
    """
    s = s.strip()

    # Implication: (LHS) -> (RHS)
    if ") -> (" in s:
        arrow_idx = s.index(") -> (")
        lhs_str = s[1:arrow_idx]  # strip leading '('
        rhs_str = s[arrow_idx + 6 : -1]  # strip trailing ')'
        lhs = _parse_atom(lhs_str, var_of)
        if lhs is None:
            return None
        rhs = _parse_rhs(rhs_str, var_of)
        if rhs is None:
            return None
        return lhs.implies(rhs)

    # Equality: (X) == (Y)
    if ") == (" in s:
        eq_idx = s.index(") == (")
        a_str = s[1:eq_idx]
        b_str = s[eq_idx + 6 : -1]
        a = _parse_atom(a_str, var_of)
        b = _parse_atom(b_str, var_of)
        if a is not None and b is not None:
            return a == b
        return None

    # Unary: plain variable or negation
    atom = _parse_atom(s, var_of)
    return atom


def _parse_atom(s: str, var_of: dict):
    """Parse a single variable reference: 'X' or '~X'."""
    s = s.strip()
    if s.startswith("~"):
        name = s[1:].strip()
        if name in var_of:
            return ~var_of[name]
    elif s in var_of:
        return var_of[s]
    return None


def _parse_rhs(s: str, var_of: dict):
    """Parse the RHS of an implication (may be atom, or(...), sum == 1, etc.)."""
    s = s.strip()

    # or(A,B,C,...) or or([A, B, C, ...])
    if s.startswith("or(") and s.endswith(")"):
        inner = s[3:-1]
        # CPMpy prints or([A, B, C]) with square brackets
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        parts = _split_args(inner)
        atoms = [_parse_atom(p, var_of) for p in parts]
        if all(a is not None for a in atoms):
            return cp.any(atoms)
        return None

    # (A + B + ...) == 1
    if "==" in s:
        eq_idx = s.index("==")
        sum_str = s[:eq_idx].strip()
        val_str = s[eq_idx + 2 :].strip()
        try:
            val = int(val_str)
        except ValueError:
            return None
        # Parse sum: strip parens, split on ' + '
        if sum_str.startswith("(") and sum_str.endswith(")"):
            sum_str = sum_str[1:-1]
        parts = sum_str.split(" + ")
        atoms = [_parse_atom(p.strip(), var_of) for p in parts]
        if all(a is not None for a in atoms) and atoms:
            return sum(atoms) == val
        return None

    # Negated atom
    return _parse_atom(s, var_of)


def _split_args(s: str) -> list[str]:
    """Split comma-separated args respecting parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def find_missing_constraints(
    learned_cl: list,
    target_cl: list,
    variables: list,
) -> dict:
    """Identify exactly which constraints are missing in each direction.

    Returns a dict with:
        missing_from_learned: list of (index, str) — target clauses not
            implied by learned constraints (false positives: assignments
            that satisfy learned but violate this target clause)
        extra_in_learned: list of (index, str) — learned constraints not
            implied by target (false negatives: assignments that satisfy
            target but violate this learned constraint)
        n_target: total target clauses
        n_learned: total learned constraints
    """
    missing_from_learned = []
    extra_in_learned = []

    # Target clauses not covered by learned
    for i, t in enumerate(target_cl):
        m = cp.Model(learned_cl + [~t])
        if m.solve():
            example = {v.name: bool(v.value()) for v in variables}
            missing_from_learned.append({
                "index": i,
                "constraint": str(t),
                "features": _clause_to_feature_names(t),
                "counterexample": example,
            })

    # Learned constraints not implied by target (over-constrained)
    for i, c in enumerate(learned_cl):
        try:
            neg = ~c
        except Exception:
            continue
        m = cp.Model(target_cl + [neg])
        if m.solve():
            example = {v.name: bool(v.value()) for v in variables}
            extra_in_learned.append({
                "index": i,
                "constraint": str(c),
                "features": _clause_to_feature_names(c),
                "counterexample": example,
            })

    return {
        "n_target": len(target_cl),
        "n_learned": len(learned_cl),
        "missing_from_learned": missing_from_learned,
        "extra_in_learned": extra_in_learned,
    }


def _rebuild_tree_info(result: dict) -> dict | None:
    """Convert JSON inferred_tree back to the tuple format constraints_from_tree expects.

    JSON format:  {parent: [[gtype, [children]], ...]}
    Tuple format: {parent: [(gtype, children), ...]}

    Returns None if inferred_tree is absent.
    """
    raw = result.get("inferred_tree")
    if not raw:
        return None
    return {
        parent: [(g[0], g[1]) for g in groups]
        for parent, groups in raw.items()
    }


def diagnose_result(
    result_path: str,
    model_override: str | None = None,
    include_tree: bool = False,
) -> dict:
    """Run full diagnosis on one result JSON file.

    When *include_tree* is True the diagnosis also rebuilds the tree-enhanced
    constraint set (structural constraints from the inferred tree + cross-tree
    learned constraints) and checks it against the same CNF target.  The
    results appear under ``tree_*`` keys in the returned dict.

    Returns a summary dict (or error dict).
    """
    result_path = Path(result_path)
    with open(result_path) as f:
        result = json.load(f)

    if result.get("error"):
        return {
            "file": str(result_path),
            "model": result.get("model"),
            "status": "error",
            "error": result["error"],
        }

    if "constraints" not in result:
        return {
            "file": str(result_path),
            "model": result.get("model"),
            "status": "no_constraints",
            "error": "No 'constraints' key in result JSON",
        }

    # Resolve model path
    model_path = model_override or result["model"]
    if not Path(model_path).exists():
        # Try relative to result file
        alt = result_path.parent / model_path
        if alt.exists():
            model_path = str(alt)
        else:
            return {
                "file": str(result_path),
                "model": model_path,
                "status": "model_not_found",
                "error": f"UVL model not found: {model_path}",
            }

    # Rebuild variables and constraints
    feature_names = result.get("feature_names") or extract_feature_names(model_path)
    variables = [cp.boolvar(name=f) for f in feature_names]

    target_cl = extract_target_constraints(model_path, variables, feature_names)
    learned_cl, unparsed = rebuild_learned_constraints(result, variables)

    # ── Flat learned diagnosis ──
    diag = find_missing_constraints(learned_cl, target_cl, variables)

    n_missing = len(diag["missing_from_learned"])
    n_extra = len(diag["extra_in_learned"])

    out = {
        "file": str(result_path),
        "model": model_path,
        "status": "ok",
        "features": len(feature_names),
        "n_target": diag["n_target"],
        "n_learned": diag["n_learned"],
        "n_unparsed": len(unparsed),
        "n_missing": n_missing,
        "n_extra": n_extra,
        "equivalent": n_missing == 0 and n_extra == 0,
        "missing_from_learned": diag["missing_from_learned"],
        "extra_in_learned": diag["extra_in_learned"],
        "unparsed_constraints": unparsed,
    }

    # ── Tree-enhanced diagnosis ──
    if include_tree:
        tree_info = _rebuild_tree_info(result)
        if tree_info is None:
            out["tree_status"] = "no_tree"
        else:
            enhanced_cl, cross_tree_cl = constraints_from_tree(
                feature_names, variables, tree_info, learned_cl,
            )
            tree_diag = find_missing_constraints(enhanced_cl, target_cl, variables)

            t_missing = len(tree_diag["missing_from_learned"])
            t_extra = len(tree_diag["extra_in_learned"])

            out["tree_status"] = "ok"
            out["tree_n_enhanced"] = len(enhanced_cl)
            out["tree_n_cross"] = len(cross_tree_cl)
            out["tree_n_missing"] = t_missing
            out["tree_n_extra"] = t_extra
            out["tree_equivalent"] = t_missing == 0 and t_extra == 0
            out["tree_missing"] = tree_diag["missing_from_learned"]
            out["tree_extra"] = tree_diag["extra_in_learned"]

    return out


def print_diagnosis(diag: dict, verbose: bool = True):
    """Pretty-print one diagnosis result."""
    print(f"\n{'=' * 70}")
    print(f"Result: {diag['file']}")
    print(f"Model:  {diag.get('model', '?')}")
    print(f"{'=' * 70}")

    if diag["status"] != "ok":
        print(f"  STATUS: {diag['status']} — {diag.get('error', '')}")
        return

    print(f"  Features:  {diag['features']}")
    print(f"  Target:    {diag['n_target']} clauses")
    print(f"  Learned:   {diag['n_learned']} constraints")
    if diag["n_unparsed"]:
        print(f"  Unparsed:  {diag['n_unparsed']} (could not reconstruct from string)")
    print(f"  Missing:   {diag['n_missing']} target clauses not implied by learned")
    print(f"  Extra:     {diag['n_extra']} learned constraints not implied by target")
    print(f"  Equivalent: {diag['equivalent']}")

    if verbose and diag["missing_from_learned"]:
        print(f"\n  MISSING target clauses (learned does NOT imply these):")
        for m in diag["missing_from_learned"]:
            print(f"    [{m['index']:3d}] {m['constraint']}")
            print(f"          features: {', '.join(m['features'])}")
            # Show only the relevant variables in the counterexample
            relevant = {k: v for k, v in m["counterexample"].items()
                        if k in [n.lstrip("~") for n in m["features"]]}
            print(f"          counterex: {relevant}")

    if verbose and diag["extra_in_learned"]:
        print(f"\n  EXTRA learned constraints (target does NOT imply these):")
        for e in diag["extra_in_learned"]:
            print(f"    [{e['index']:3d}] {e['constraint']}")
            print(f"          features: {', '.join(e['features'])}")

    if verbose and diag["unparsed_constraints"]:
        print(f"\n  UNPARSED constraints (string could not be reconstructed):")
        for s in diag["unparsed_constraints"]:
            print(f"    {s}")

    # ── Tree-enhanced section ──
    tree_status = diag.get("tree_status")
    if tree_status == "no_tree":
        print(f"\n  Tree: no inferred_tree in result JSON")
    elif tree_status == "ok":
        print(f"\n  --- After tree refinement ---")
        print(f"  Enhanced:  {diag['tree_n_enhanced']} constraints ({diag['tree_n_cross']} cross-tree)")
        print(f"  Missing:   {diag['tree_n_missing']} target clauses not implied by tree-enhanced")
        print(f"  Extra:     {diag['tree_n_extra']} tree-enhanced constraints not implied by target")
        print(f"  Equivalent: {diag['tree_equivalent']}")

        if verbose and diag["tree_missing"]:
            print(f"\n  TREE-MISSING target clauses:")
            for m in diag["tree_missing"]:
                print(f"    [{m['index']:3d}] {m['constraint']}")
                print(f"          features: {', '.join(m['features'])}")
                relevant = {k: v for k, v in m["counterexample"].items()
                            if k in [n.lstrip("~") for n in m["features"]]}
                print(f"          counterex: {relevant}")

        if verbose and diag["tree_extra"]:
            print(f"\n  TREE-EXTRA constraints (target does NOT imply these):")
            for e in diag["tree_extra"]:
                print(f"    [{e['index']:3d}] {e['constraint']}")
                print(f"          features: {', '.join(e['features'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose which target constraints are missing from learned results.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result JSON file(s) or directory containing .json files",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override UVL model path (only for single-file mode)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact summary table instead of full details",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only show results that have missing or extra constraints",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Also diagnose the tree-enhanced constraint set (structural + cross-tree)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write full diagnosis as JSON to this file",
    )

    args = parser.parse_args()

    # Collect JSON files
    json_files = []
    for p in args.paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json":
            json_files.append(p)
        elif p.is_dir():
            json_files.extend(sorted(p.glob("*.json")))
        else:
            print(f"Warning: skipping {p}", file=sys.stderr)

    if not json_files:
        parser.error("No JSON files found")

    all_diags = []
    for jf in json_files:
        model_arg = args.model if len(json_files) == 1 else None
        diag = diagnose_result(str(jf), model_override=model_arg, include_tree=args.tree)
        all_diags.append(diag)

        if args.only_missing and diag["status"] == "ok":
            flat_ok = diag.get("equivalent", True)
            tree_ok = diag.get("tree_equivalent", True) if args.tree else True
            if flat_ok and tree_ok:
                continue
        if args.summary:
            status = diag["status"]
            if status == "ok":
                eq = "EQ" if diag["equivalent"] else "NEQ"
                line = (
                    f"{jf.name:50s}  {eq:3s}  "
                    f"miss={diag['n_missing']:3d}  extra={diag['n_extra']:3d}  "
                    f"target={diag['n_target']:4d}  learned={diag['n_learned']:4d}"
                )
                if args.tree and diag.get("tree_status") == "ok":
                    teq = "EQ" if diag["tree_equivalent"] else "NEQ"
                    line += (
                        f"  | tree:{teq:3s}  "
                        f"miss={diag['tree_n_missing']:3d}  extra={diag['tree_n_extra']:3d}  "
                        f"enhanced={diag['tree_n_enhanced']:4d}"
                    )
                elif args.tree:
                    line += "  | tree:n/a"
                print(line)
            else:
                print(f"{jf.name:50s}  {status}")
        else:
            print_diagnosis(diag, verbose=True)

    if args.out:
        # Strip counterexamples for compact JSON output
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            json.dump(all_diags, f, indent=2, default=str)
        print(f"\nWrote {len(all_diags)} diagnoses to {out_path}")

    # Summary footer
    ok = [d for d in all_diags if d["status"] == "ok"]
    eq = [d for d in ok if d["equivalent"]]
    neq = [d for d in ok if not d["equivalent"]]
    errs = [d for d in all_diags if d["status"] != "ok"]
    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_diags)} files — {len(eq)} equivalent, {len(neq)} with gaps, {len(errs)} errors/skipped")
    if neq:
        total_missing = sum(d["n_missing"] for d in neq)
        total_extra = sum(d["n_extra"] for d in neq)
        print(f"  Flat:  missing={total_missing}, extra={total_extra}")
    if args.tree:
        tree_ok = [d for d in ok if d.get("tree_status") == "ok"]
        tree_eq = [d for d in tree_ok if d["tree_equivalent"]]
        tree_neq = [d for d in tree_ok if not d["tree_equivalent"]]
        print(f"  Tree:  {len(tree_eq)} equivalent, {len(tree_neq)} with gaps (of {len(tree_ok)} with tree)")
        if tree_neq:
            t_missing = sum(d["tree_n_missing"] for d in tree_neq)
            t_extra = sum(d["tree_n_extra"] for d in tree_neq)
            print(f"         missing={t_missing}, extra={t_extra}")


if __name__ == "__main__":
    main()
