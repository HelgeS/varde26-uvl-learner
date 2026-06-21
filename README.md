# varde26-uvl-learner

UVL Learner project @ VARDE 2026.

Interactive **constraint acquisition (CA)** on UVL feature models. A PyCONA CA
algorithm queries a SAT-based oracle (a flamapy encoding of a `.uvl` file) to
learn constraints, which are then mapped back to UVL semantics (requires,
excludes, groups, tree structure).

This repository is the **tree-unknown** subset of a larger research codebase:
only feature *names* are known during learning; the tree structure is
reconstructed afterwards from the learned constraints.

## Setup

```bash
uv sync
```

Key dependencies (see `pyproject.toml`): `pycona` (CA algorithms), `cpmpy`
(constraint modeling), `flamapy` (UVL parsing + SAT/CNF encoding), `networkx`
(graph-bias variant), `python-sat`.

`models/` contains the UVL feature models used as oracles / inputs.

## Quick start

```bash
# Generate the built-in sandwich example and learn it, then SAT-verify
uv run python ca_uvl_notree.py --generate-example --verify

# Single model
uv run python ca_uvl_notree.py models/aircraft_fm.uvl --verify

# Batch over a directory, write one JSON per model, cap feature count
uv run python ca_uvl_notree.py models/ --out-dir results/ --max-features 50 --timeout 120

# Export the learned model back to UVL
uv run python ca_uvl_notree.py models/aircraft_fm.uvl --export-uvl out/
```

## The three runners

All three learn a **flat pairwise bias** with no tree knowledge, then run the
post-CA pipeline (tree inference → group refinement → constraint extraction →
cleanup → optional SAT verification). They differ only in how the bias / CA loop
is organized:

| Runner | Approach |
| --- | --- |
| `ca_uvl_notree.py` | Baseline. Binary pairwise bias (+ optional static n-ary group bias via `--group-bias-max`). "Skip-collapse" handles n-ary clauses the binary bias can't represent. |
| `ca_uvl_notree_grow_bias.py` | Iteratively **grows** the group bias: on a Collapse it widens the candidate group size and restarts CA, seeded from what was already learned. |
| `ca_uvl_notree_graph_bias.py` | Custom graph/conjunction-aware CA components (own FindScope/FindC/QGen) built on `networkx` for richer n-ary candidate generation. |

Common flags: `--verify`, `--export-uvl PATH`, `--out-dir DIR`,
`--algorithm {quacq,mquacq,mquacq2,growacq,pquacq,mineacq,genacq}`,
`--group-bias-max N`, `--no-cleanup`, `--no-skip-collapse`, `--deep`,
`--max-features N`, `--timeout S`.

## Post-CA pipeline (shared)

1. `infer_and_refine_tree()` — reconstruct a tree from learned constraints and
   recover missing `P => any(children)` completeness clauses via oracle queries.
2. `_validate_tree()` / `_fix_multi_parent_tree()` — enforce single-parent tree.
3. `constraints_from_tree()` — extract structural constraints + cross-tree residuals.
4. `cleanup_dumb()` — drop spurious cross-tree constraints (disable with `--no-cleanup`).
5. `verify_learned()` — SAT equivalence check vs. the oracle (`--verify`).
6. `export_learned_to_uvl()` — write a `.uvl` file (`--export-uvl`).

## File map — what's needed for what

### Core (required by all three runners)
- `ca_common.py` — feature-name / target-constraint extraction, `ALGORITHMS`
  registry, timeout + result-saving helpers.
- `tree_inference.py` — tree reconstruction, validation, constraint extraction,
  cleanup.
- `uvl_export.py` — `export_learned_to_uvl()` and `verify_learned()` (SAT
  equivalence).

### `--deep` diagnosis (optional)
Lazily imported only when a runner is given `--deep`:
- `report_results.py` — result reporting + `deep_analysis()`.
- `diagnose_missing.py` — classify/locate missing clauses.
- `diagnose_underconstraining.py` — bias-coverage + under-constraining analysis.
- `refine_from_json.py` — re-run tree inference/refinement from a saved results
  JSON without re-running CA (also usable standalone).

### Standalone utilities (independent of the runners)
- `uvl_to_candy.py` — convert UVL models to the Candy `.bias`/`.target` format
  (samples in `cafrmt/`). `uv run python uvl_to_candy.py models/sandwich.uvl --out-dir cafrmt/`
- `extract_reference.py` — extract the pairwise (binary+unit) constraint subset
  directly from a UVL's CNF as reference JSON, for testing the tree pipeline
  without CA. `uv run python extract_reference.py models/aircraft_fm.uvl`

### Dependency summary

| You want to run… | Needs |
| --- | --- |
| `ca_uvl_notree.py` | core (`ca_common`, `tree_inference`, `uvl_export`) |
| `ca_uvl_notree_grow_bias.py` | core (`ca_common`, `tree_inference`, `uvl_export`) |
| `ca_uvl_notree_graph_bias.py` | core + imports `build_bias` from `ca_uvl_notree.py` |
| any runner with `--deep` | + `report_results`, `diagnose_missing`, `diagnose_underconstraining`, `refine_from_json` |
| `uvl_to_candy.py` | `ca_common` |
| `extract_reference.py` | `ca_common` |
| `refine_from_json.py` (standalone) | core + `report_results` (+ deep chain) |

## CPMpy pitfall

`NegBoolView` is a **subclass** of `_BoolVarImpl`, so always check `NegBoolView`
first:

```python
# Correct
if isinstance(expr, _BoolVarImpl) and not isinstance(expr, NegBoolView): ...
```

`NegBoolView` has `.name == '~Foo'` and `._bv` for the underlying variable.
