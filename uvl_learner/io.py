"""Input/output helpers: path globbing, result JSON, timeouts, and UVL writing.

- ``collect_uvl_paths`` / ``save_result`` / ``TimeoutError`` — run plumbing.
- ``cpmpy_to_uvl`` / ``export_learned_to_uvl`` — render learned constraints and a
  reconstructed feature tree back into a ``.uvl`` file.
"""

import json
from pathlib import Path

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison


# ── Timeout ──────────────────────────────────────────────────────────


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ── Result IO ─────────────────────────────────────────────────────────


def save_result(result: dict, path: Path):
    """Write a single result dict as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result to {path}")


def collect_uvl_paths(paths: list[str]) -> list[Path]:
    """Expand CLI args: files are kept, directories are globbed for *.uvl."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".uvl":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.uvl")))
        else:
            print(f"Skipping {p} (not a .uvl file or directory)")
    return out


# ── Name quoting ──────────────────────────────────────────────────────────

# _SPECIAL = re.compile(r'[ ()\=><\!&|"\']')


def _safe_name(name: str) -> str:
    """Quote a feature name that contains UVL-special characters."""
    return f'"{name}"'  # if _SPECIAL.search(name) else name


def _wrap(s: str, condition: bool) -> str:
    return f"({s})" if condition else s


# ── CPMpy → UVL conversion ────────────────────────────────────────────────


def cpmpy_to_uvl(expr) -> str | None:
    """Recursively convert a CPMpy expression to a UVL propositional string.

    Returns None when the expression cannot be represented in standard UVL
    (e.g. sum / wsum constraints).
    """
    # Plain boolean variable
    if isinstance(expr, _BoolVarImpl):
        return _safe_name(expr.name)

    # Negated boolean variable  (~v)
    if isinstance(expr, NegBoolView):
        return f"!{_safe_name(expr._bv.name)}"

    if isinstance(expr, Operator):
        name = expr.name
        args = expr.args

        if name == "->":
            a, b = args
            a_s = cpmpy_to_uvl(a)
            b_s = cpmpy_to_uvl(b)
            if a_s is None or b_s is None:
                return None
            # Wrap LHS if it is itself a compound operator
            a_wrap = isinstance(a, Operator) and a.name in ("->", "or", "and")
            # Wrap RHS only for right-associative chain (nested ->)
            b_wrap = isinstance(b, Operator) and b.name == "->"
            return f"{_wrap(a_s, a_wrap)} => {_wrap(b_s, b_wrap)}"

        if name == "or":
            parts = []
            for arg in args:
                s = cpmpy_to_uvl(arg)
                if s is None:
                    return None
                # -> and <=> have lower precedence than |, so they need parens
                needs = isinstance(arg, Comparison) or (
                    isinstance(arg, Operator) and arg.name == "->"
                )
                parts.append(_wrap(s, needs))
            return " | ".join(parts)

        if name == "and":
            parts = []
            for arg in args:
                s = cpmpy_to_uvl(arg)
                if s is None:
                    return None
                # |, ->, <=> all have lower/equal-then-lower precedence than &
                needs = isinstance(arg, Comparison) or (
                    isinstance(arg, Operator) and arg.name in ("or", "->")
                )
                parts.append(_wrap(s, needs))
            return " & ".join(parts)

        if name == "not":
            a = args[0]
            a_s = cpmpy_to_uvl(a)
            if a_s is None:
                return None
            return f"!({a_s})"

        # sum, wsum, etc. — not representable in plain UVL
        return None

    if isinstance(expr, Comparison):
        if expr.name == "==":
            a, b = expr.args
            a_s = cpmpy_to_uvl(a)
            b_s = cpmpy_to_uvl(b)
            if a_s is None or b_s is None:
                return None
            a_wrap = isinstance(a, (Operator, Comparison))
            b_wrap = isinstance(b, (Operator, Comparison))
            return f"{_wrap(a_s, a_wrap)} <=> {_wrap(b_s, b_wrap)}"
        return None

    return None


# ── Feature-tree rendering ────────────────────────────────────────────────


def _render_tree(feature: str, tree_info: dict, indent: int) -> list[str]:
    """Recursively render UVL feature tree lines for *feature*.

    indent=1 for the root feature (under the top-level "features" keyword).
    Indentation unit: 4 spaces per level.
    """
    pad = "    " * indent
    lines = [f"{pad}{_safe_name(feature)}"]

    for gtype, children in tree_info.get(feature, []):
        if gtype in ("mandatory", "optional", "alternative", "or"):
            keyword = gtype
        else:
            keyword = f"or  // {gtype}"  # cardinality fallback

        lines.append(f"{'    ' * (indent + 1)}{keyword}")
        for child in children:
            lines.extend(_render_tree(child, tree_info, indent + 2))

    return lines


def _find_root(feature_names: list[str], tree_info: dict) -> str:
    """Return the feature that never appears as a child — the root."""
    all_children: set[str] = set()
    for groups in tree_info.values():
        for _, children in groups:
            all_children.update(children)

    for name in feature_names:
        if name not in all_children:
            return name

    return feature_names[0]  # fallback


# ── Full UVL export ───────────────────────────────────────────────────────


def export_learned_to_uvl(
    feature_names: list[str],
    tree_info: dict,
    learned_cl: list,
    output_path: str,
) -> tuple[int, int]:
    """Write a .uvl file reconstructed from the feature tree and learned constraints.

    Parameters
    ----------
    feature_names : list of feature name strings
    tree_info     : parent -> [(group_type, [child_names]), ...]
    learned_cl    : learned CPMpy constraint list
    output_path   : destination .uvl file path

    Returns
    -------
    (exported_count, skipped_count)
    """
    root = _find_root(feature_names, tree_info)

    # Feature-tree section
    tree_lines = _render_tree(root, tree_info, indent=1)

    # Constraints section
    constraint_lines: list[str] = []
    exported = 0
    skipped = 0
    for c in learned_cl:
        s = cpmpy_to_uvl(c)
        if s is not None:
            constraint_lines.append(f"    {s}")
            exported += 1
        else:
            skipped += 1

    # Assemble
    uvl_lines = ["features"] + tree_lines

    if len(constraint_lines) > 0:
        uvl_lines.extend(["", "constraints"] + constraint_lines)

    Path(output_path).write_text("\n".join(uvl_lines) + "\n", encoding="utf-8")
    return exported, skipped
