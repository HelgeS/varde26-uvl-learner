"""Sample valid/invalid configurations from a UVL model as CSV.

Output: one column per feature (no header) + final column 1/0 (valid/invalid).
Invalid configurations are produced by flipping a single feature in a valid
configuration and confirming the result is unsatisfiable against the model.
"""

import argparse
import csv
import random
import sys

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from flamapy.core.discover import DiscoverMetamodels
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel


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


def build_model(uvl_path):
    names = extract_feature_names(uvl_path)
    vars_ = cp.boolvar(shape=len(names), name="f")
    cons = extract_target_constraints(uvl_path, list(vars_), names)
    return names, list(vars_), cons


def is_valid(vars_, cons, assignment):
    m = cp.Model(cons)
    for v, val in zip(vars_, assignment):
        m += v == bool(val)
    return m.solve()


def sample_valid(vars_, cons, n, rng):
    """Sample up to n distinct valid configurations using random objectives."""
    seen = set()
    results = []
    attempts = 0
    max_attempts = n * 20
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        m = cp.Model(cons)
        # Random linear objective to diversify solutions
        weights = [rng.choice([-1, 1]) for _ in vars_]
        m.maximize(sum(w * v for w, v in zip(weights, vars_)))
        if not m.solve():
            break
        assign = tuple(int(v.value()) for v in vars_)
        if assign in seen:
            continue
        seen.add(assign)
        results.append(assign)
    return results


def enumerate_all(vars_, cons):
    """Enumerate all 2^n configurations, classifying each as valid or invalid."""
    n = len(vars_)
    valids = []
    invalids = []
    for i in range(2**n):
        assign = tuple((i >> j) & 1 for j in range(n))
        if is_valid(vars_, cons, assign):
            valids.append(assign)
        else:
            invalids.append(assign)
    return valids, invalids


def make_invalid(vars_, cons, valid_assign, rng, max_tries=50):
    """Flip one bit until the result is invalid."""
    n = len(valid_assign)
    idxs = list(range(n))
    rng.shuffle(idxs)
    for i in idxs[:max_tries]:
        candidate = list(valid_assign)
        candidate[i] = 1 - candidate[i]
        if not is_valid(vars_, cons, candidate):
            return tuple(candidate)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uvl")
    ap.add_argument(
        "-n",
        type=int,
        required=True,
        help="number of configs (split ~half valid/half invalid)",
    )
    ap.add_argument("-o", "--out", default="-", help="output CSV path ('-' for stdout)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    names, vars_, cons = build_model(args.uvl)

    total_space = 2 ** len(names)
    if args.n >= total_space:
        # Enumerate the entire configuration space
        print(
            f"n={args.n} >= 2^{len(names)}={total_space}, enumerating all configurations",
            file=sys.stderr,
        )
        valids, invalids = enumerate_all(vars_, cons)
    else:
        n_valid = args.n // 2
        n_invalid = args.n - n_valid

        valids = sample_valid(vars_, cons, n_valid, rng)
        if len(valids) < n_valid:
            print(
                f"warning: only produced {len(valids)}/{n_valid} valid configs",
                file=sys.stderr,
            )

        # For invalids, sample extra valid bases (with repetition allowed)
        bases = sample_valid(vars_, cons, n_invalid, rng)
        if not bases and valids:
            bases = valids
        invalids = []
        i = 0
        while len(invalids) < n_invalid and bases:
            base = bases[i % len(bases)]
            i += 1
            if i > n_invalid * 20:
                break
            inv = make_invalid(vars_, cons, base, rng)
            if inv is not None:
                invalids.append(inv)

    if args.n < total_space and len(invalids) < (args.n - args.n // 2):
        print(
            f"warning: only produced {len(invalids)}/{args.n - args.n // 2} invalid configs",
            file=sys.stderr,
        )

    rows = [list(v) + [1] for v in valids] + [list(v) + [0] for v in invalids]
    rng.shuffle(rows)

    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    try:
        w = csv.writer(out)
        for row in rows:
            w.writerow(row)
    finally:
        if out is not sys.stdout:
            out.close()

    print(f"# features: {','.join(names)}", file=sys.stderr)


if __name__ == "__main__":
    main()
