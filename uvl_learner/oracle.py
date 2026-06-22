"""UVL → oracle setup.

Parse a ``.uvl`` model into the inputs a constraint-acquisition run needs:

- ``extract_feature_names`` / ``extract_features`` — feature names (and, optionally,
  the tree structure used only as ground truth).
- ``extract_target_constraints`` — flamapy's SAT/CNF encoding as CPMpy constraints,
  i.e. the ground truth the oracle answers from.
- ``setup_problem`` — convenience wrapper returning everything a runner needs:
  ``(feature_names, variables, target, oracle)``.
"""

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from flamapy.core.discover import DiscoverMetamodels
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel
from pycona import ConstraintOracle


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


def get_reference_configuration(uvl_path: str, X: list) -> dict[str, bool]:
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


# ── Convenience: full problem setup ──────────────────────────────────


def setup_problem(uvl_path: str):
    """Build everything a runner needs from a UVL path.

    Returns ``(feature_names, variables, target, oracle)`` where the tree
    structure is discarded — only names are used to create the CPMpy variables,
    and the SAT/CNF encoding becomes the oracle's ground truth.
    """
    feature_names = extract_feature_names(uvl_path)
    variables = [cp.boolvar(name=f) for f in feature_names]
    target = extract_target_constraints(uvl_path, variables, feature_names)
    oracle = ConstraintOracle(target)
    return feature_names, variables, target, oracle
