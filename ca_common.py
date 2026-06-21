"""Shared utilities for constraint acquisition on UVL feature models.

Functions and constants used by both ca_uvl.py (tree-known) and
ca_uvl_notree.py (tree-unknown), as well as diagnostic scripts.
"""

import json
from pathlib import Path

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from flamapy.core.discover import DiscoverMetamodels
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel
from pycona import (
    QuAcq,
    MQuAcq,
    MQuAcq2,
    GrowAcq,
    PQuAcq,
    MineAcq,
    GenAcq,
)

ALGORITHMS = {
    "quacq": QuAcq,
    "mquacq": MQuAcq,
    "mquacq2": MQuAcq2,
    "growacq": GrowAcq,
    "pquacq": PQuAcq,
    "mineacq": MineAcq,
    "genacq": GenAcq,
}


# ── Feature extraction ──────────────────────────────────────────────────


def extract_feature_names(uvl_path: str) -> list[str]:
    """Return flat list of feature names from a UVL file.

    The tree structure is intentionally discarded; only names are returned.
    """
    dm = DiscoverMetamodels()
    fm = dm.use_transformation_t2m(uvl_path, "fm")

    feature_names = []

    def walk(feature):
        feature_names.append(feature.name)
        for rel in feature.get_relations():
            for child in rel.children:
                walk(child)

    walk(fm.root)
    return feature_names


def extract_features(uvl_path: str) -> tuple[list[str], dict]:
    """Return (feature_names, tree_info) from a UVL file.

    tree_info maps parent_name -> [(group_type, [child_names]), ...].
    """
    dm = DiscoverMetamodels()
    fm = dm.use_transformation_t2m(uvl_path, "fm")

    feature_names = []
    tree_info = {}

    def walk(feature):
        feature_names.append(feature.name)
        for rel in feature.get_relations():
            children = [c.name for c in rel.children]
            if rel.is_mandatory():
                gtype = "mandatory"
            elif rel.is_optional():
                gtype = "optional"
            elif rel.is_or():
                gtype = "or"
            elif rel.is_alternative():
                gtype = "alternative"
            else:
                gtype = f"cardinality[{rel.card_min}..{rel.card_max}]"
            tree_info.setdefault(feature.name, []).append((gtype, children))
            for child in rel.children:
                walk(child)

    walk(fm.root)
    return feature_names, tree_info


# ── Target constraint extraction (oracle ground truth) ───────────────


def extract_target_constraints(uvl_path, variables, feature_names):
    """Convert flamapy's SAT/CNF encoding of the UVL model to CPMpy constraints."""
    fm = FLAMAFeatureModel(uvl_path)
    fm._transform_to_sat()
    sat = fm.sat_model

    var_by_name = {name: v for name, v in zip(feature_names, variables)}
    id_to_var = {}
    for name, sat_id in sat.variables.items():
        if name in var_by_name:
            id_to_var[sat_id] = var_by_name[name]

    constraints = []
    for clause in sat.get_all_clauses().clauses:
        literals = []
        for lit in clause:
            var_id = abs(lit)
            if var_id not in id_to_var:
                break
            literals.append(id_to_var[var_id] if lit > 0 else ~id_to_var[var_id])
        else:
            if len(literals) == 1:
                constraints.append(literals[0])
            else:
                constraints.append(cp.any(literals))

    return list(set(toplevel_list(constraints)))

def print_target_model(uvl_path, variables, feature_names):
    """Convert flamapy's SAT/CNF encoding of the UVL model to CPMpy constraints."""
    fm = FLAMAFeatureModel(uvl_path)
    fm._transform_to_sat()
    sat = fm.sat_model

    var_by_name = {name: v for name, v in zip(feature_names, variables)}
    id_to_var = {}
    for name, sat_id in sat.variables.items():
        if name in var_by_name:
            id_to_var[sat_id] = var_by_name[name]

    constraints = []
    for clause in sat.get_all_clauses().clauses:
        literals = []
        for lit in clause:
            var_id = abs(lit)
            if var_id not in id_to_var:
                break
            literals.append(id_to_var[var_id] if lit > 0 else ~id_to_var[var_id])
        else:
            if len(literals) == 1:
                constraints.append(literals[0])
            else:
                constraints.append(cp.any(literals))
                

    return list(set(toplevel_list(constraints)))



# ── Timeout ──────────────────────────────────────────────────────────


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ── Output ───────────────────────────────────────────────────────────


def save_result(result: dict, path: Path):
    """Write a single result dict as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result to {path}")


# ── CLI helpers ──────────────────────────────────────────────────────


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


def get_reference_configuration(uvl_path: str, X: list[cp.boolvar]) -> dict[str, bool]:
    fm = FLAMAFeatureModel(uvl_path)
    # TODO Not sure how to get only 1 configuration
    # Might take a long time for larger models
    all_valid_configs = fm.configurations()
    reference_cfg = {
        v.name: all_valid_configs[0].get_value(v.name)
        if all_valid_configs[0].is_selected(v.name)
        else False
        for v in X
    }
    return reference_cfg
