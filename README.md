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
uv run python -m runners.pairwise --generate-example --verify

# Single model
uv run python -m runners.pairwise models/aircraft_fm.uvl --verify

# Batch over a directory, write one JSON per model, cap feature count
uv run python -m runners.pairwise models/ --out-dir results/ --max-features 50 --timeout 120

# Export the learned model back to UVL
uv run python -m runners.pairwise models/aircraft_fm.uvl --export-uvl out/
```

## Project layout

```
uvl_learner/          importable library — the pipeline stages
  oracle.py             UVL → feature names, variables, target, ConstraintOracle
  io.py                 path globbing, result JSON, timeouts, UVL writer
  verify.py             SAT equivalence check (the "validation" pillar)
  acquire.py            shared CA machinery: skip-collapse env, ALGORITHMS,
                          build_algorithm
  bias/                 candidate-constraint construction (one file per strategy)
    pairwise.py           static binary (+ optional n-ary group) bias
    grow.py               grow-on-collapse bias
    graph.py              networkx bias + custom FindScope/FindC/QGen components
  reconstruct/          post-CA tree pipeline (one file per stage)
    tree.py               infer_tree, validate, single-parent repair
    refine.py             infer_and_refine_tree (+ deprecated variants)
    extract.py            constraints_from_tree
    cleanup.py            drop spurious cross-tree constraints
    _common.py            shared helpers

runners/              the three entry points (run with `python -m runners.<name>`)
  pairwise.py           baseline static pairwise bias
  grow.py               grow-bias (own skip-collapse env)
  graph.py              graph-bias (uses uvl_learner.bias.graph)

diagnostics/          post-hoc analysis (also reachable via runner `--deep`)
  report.py  missing.py  underconstraining.py  refine_from_json.py
convert/              standalone data-format tools
  to_candy.py  to_reference.py
```

## The three runners

Each runner builds a bias, runs a pycona CA loop against the oracle, then runs the
shared post-CA pipeline (tree inference → group refinement → constraint extraction
→ cleanup → optional SAT verification). They differ only in the bias / CA loop:

| Runner | Approach |
| --- | --- |
| `runners.pairwise` | Baseline. Binary pairwise bias (+ optional static n-ary group bias via `--group-bias-max`). "Skip-collapse" handles n-ary clauses the binary bias can't represent. |
| `runners.grow` | Iteratively **grows** the group bias: on a Collapse it widens the candidate group size and restarts CA, seeded from what was already learned. Keeps its own skip-collapse env. |
| `runners.graph` | Custom graph/conjunction-aware CA components (own FindScope/FindC/QGen) built on `networkx`, in `uvl_learner.bias.graph`, for richer n-ary candidate generation. |

Common flags: `--verify`, `--export-uvl PATH`, `--out-dir DIR`,
`--algorithm {quacq,mquacq,mquacq2,growacq,pquacq,mineacq,genacq}`,
`--group-bias-max N`, `--no-cleanup`, `--no-skip-collapse`, `--deep`,
`--max-features N`, `--timeout S`.

## Post-CA pipeline (shared)

1. `reconstruct.refine.infer_and_refine_tree()` — reconstruct a tree from learned
   constraints and recover missing `P => any(children)` completeness clauses.
2. `reconstruct.tree._validate_tree()` / `_fix_multi_parent_tree()` — enforce single-parent tree.
3. `reconstruct.extract.constraints_from_tree()` — structural + cross-tree residuals.
4. `reconstruct.cleanup.cleanup_dumb()` — drop spurious cross-tree constraints (`--no-cleanup` to skip).
5. `verify.verify_learned()` — SAT equivalence check vs. the oracle (`--verify`).
6. `io.export_learned_to_uvl()` — write a `.uvl` file (`--export-uvl`).

## Diagnostics & conversion

```bash
# Fast overview from result JSON (no SAT); add --deep for SAT-based diagnosis
uv run python -m diagnostics.report results/

# Which target clauses are missing from a result
uv run python -m diagnostics.missing results/stack_fm.json

# Re-run the post-CA pipeline from a saved JSON without re-running CA
uv run python -m diagnostics.refine_from_json -p results/REAL-FM-5.json --verify

# Convert UVL → Candy .bias/.target  (samples in candy/)
uv run python -m convert.to_candy models/sandwich.uvl --out-dir candy/

# Extract the pairwise (binary+unit) reference subset as JSON
uv run python -m convert.to_reference models/aircraft_fm.uvl
```

Each runner's `--deep` flag lazily runs `diagnostics.report.deep_analysis()` on
the converged results.

## CPMpy pitfall

`NegBoolView` is a **subclass** of `_BoolVarImpl`, so always check `NegBoolView`
first:

```python
# Correct
if isinstance(expr, _BoolVarImpl) and not isinstance(expr, NegBoolView): ...
```

`NegBoolView` has `.name == '~Foo'` and `._bv` for the underlying variable.
