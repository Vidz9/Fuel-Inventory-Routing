"""
main_pool_filter.py
===================
Phase-3-only experiment runner.

Instead of generating routes with branch-and-price, this script restores the
route pools that were generated in a previous run (stored in a results .pkl),
applies a route-pool filter, and solves the route-based MILP over the
filtered pool.  Because every filter sees exactly the same pool, differences
in outcome are attributable to the filter alone.

For every instance it runs every (filter, p) combination in
FILTERS_TO_RUN x P_VALUES.
    - p = 1.0 means no filtering: it is run once as "full" (not once per
      filter), and only if 1.0 is included in the p values.
    - "random" is only run if it is listed in FILTERS_TO_RUN / --filters,
      once for every seed in RANDOM_SEEDS.

Time budget
-----------
The integer solve gets the same budget it had in the original run:
    budget = TOTAL_TIME_LIMIT - generation_time / 60     (minutes)
where generation_time is read from ir.cg.generation_time in the source pkl.
Set FIXED_BUDGET_MIN to override this with a constant.

Resuming
--------
Results are saved after every single solve.  Re-running the script skips
(filter, p, seed, instance) combinations that are already in the output file.

Command-line overrides (optional) let you split the work over processes:
    python main_pool_filter.py --filters svm_rbf lr --p 0.4 0.5 --instances 11 12
    python main_pool_filter.py --range 11 20          # inclusive range
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import joblib
import numpy as np
import pandas as pd

from config import PenaltyConfig, ProblemConfig, SolverConfig
from data.loader import load_data
from instance import RouteRecord, create_instance
from solvers.column_gen import run_column_generation


# ===========================================================================
# Configuration
# ===========================================================================

DATA_DIR        = r"C:\Users\u894632\OneDrive - Tilburg University\Documents\Research\IRP\Data"
INSTANCES_FILE  = "instances_extended.pkl"
RESULTS_DIR     = "results"

SOURCE_PKL      = r"results\results_0.85demand_big_ws101_gf_t20.pkl"   # stored pools
OUTPUT_TAG      = "pool_filter_big_ws101_t20"                           # output file name

NETWORK_SIZE    = "big"
VT              = (3, 5)
TOTAL_TIME_LIMIT: float           = 20.0    # minutes, as in the original run
FIXED_BUDGET_MIN: Optional[float] = None    # e.g. 15.0 to ignore stored gen times

P_VALUES:     List[float] = [0.5, 0.8]   # add 1.0 to also solve the full pool
RANDOM_SEEDS: List[int]   = [42]                        # only used if "random" is run
INSTANCE_IDS:   Optional[List[int]]        = None   # explicit list, e.g. [11, 14, 17]
INSTANCE_RANGE: Optional[Tuple[int, int]] = None   # inclusive, e.g. (11, 20)


INSTANCE_RANGE = (11,30)

GUROBI_THREADS: Optional[int] = None        # None = Gurobi default

ALL_FEATURES = [
    "gen_warmstart", "gen_heuristic", "gen_suboptimal", "gen_optimal",
    "reduced_cost", "route_order", "n_depots", "n_stations",
    "duration", "n_stations_net", "total_demand", "usage_ratio", "avg_lp_value",
]
ROUTE_FEATURES = ["n_depots", "n_stations", "duration", "n_stations_net", "total_demand"]

# name -> (path to joblib (model, scaler), feature list the model was trained on)
# "random" is handled separately and needs no model.
FILTERS: Dict[str, tuple] = {
    "svm_rbf":         (r"results\svm_model.pkl",                  ALL_FEATURES),
    "svm_linear":      (r"results\svm_linear_model.pkl",           ALL_FEATURES),
    "lr":              (r"results\lr_model.pkl",                   ALL_FEATURES),
    "svm_route_props": (r"results\svm_route_properties_model.pkl", ROUTE_FEATURES),
    "random":          (None,                                      None),
}

# Filters run by default (random excluded; request it with --filters random)
FILTERS_TO_RUN: List[str] = ["svm_rbf", "svm_linear","lr", "svm_route_props","random"]


# ===========================================================================
# Gurobi environment
# ===========================================================================

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
if GUROBI_THREADS is not None:
    env.setParam("Threads", GUROBI_THREADS)
env.start()


# ===========================================================================
# Pool restoration
# ===========================================================================

def restore_pool(instance, stored_records: Dict[int, RouteRecord]) -> None:
    """
    Inject the stored routes into a fresh instance, preserving route ids,
    so that the generation-order feature and all metadata are identical to
    the original run.  Route 0 (dummy) already exists in a fresh instance.
    """
    routes = instance.routes
    data   = instance.data
    V, N   = data.V, data.N
    T      = len(next(iter(instance.demand.values())))

    for rid in sorted(stored_records):
        if rid == 0:
            continue
        if rid != routes.R:
            raise ValueError(f"Non-contiguous route ids: expected {routes.R}, got {rid}")

        rec   = stored_records[rid]
        stops = list(rec.stops)

        edges = np.zeros((N, N))
        for i, j in zip(stops[:-1], stops[1:]):
            edges[i, j] = 1

        routes.route_edges[rid] = edges
        routes.route_durations.append(float(rec.duration))
        routes.new_routes[rid]    = stops
        routes.reduced_costs[rid] = rec.reduced_cost if rec.reduced_cost is not None else 0.0

        for edge in zip(stops[:-1], stops[1:]):
            routes.edge_routes.setdefault(edge, []).append(rid)

        for v in range(V):
            for t in range(T):
                routes.feasible_routes[(v, t)].append(rid)

        # Copy the record (including RMP usage statistics) unchanged
        routes.route_records[rid] = RouteRecord(**vars(rec))
        routes.R += 1


# ===========================================================================
# Features and filters
# ===========================================================================

def _gen_encode(rec: RouteRecord) -> List[int]:
    gq = rec.generation_quality if rec.generation_quality else "heuristic"
    if rec.origin == "warmstart": return [1, 0, 0, 0]
    if gq == "heuristic":         return [0, 1, 0, 0]
    if gq == "suboptimal":        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def build_feature_frame(instance) -> pd.DataFrame:
    """One row per non-dummy route, all 13 features, indexed by route id."""
    data      = instance.data
    n_stn_net = len(data.station_indices)
    dem_by_idx = {idx: float(sum(instance.demand[data.Nodes[idx]]))
                  for idx in data.station_indices}

    rows, rids = [], []
    for rid, rec in instance.routes.route_records.items():
        if rec.origin == "dummy":
            continue
        rc = rec.reduced_cost if rec.reduced_cost is not None else 0.0
        rows.append(_gen_encode(rec) + [
            float(rc), int(rid), int(rec.n_depots), int(rec.n_stations),
            float(rec.duration), int(n_stn_net),
            float(sum(dem_by_idx.get(s, 0.0) for s in rec.stops)),
            float(rec.usage_ratio), float(rec.avg_lp_value),
        ])
        rids.append(rid)
    return pd.DataFrame(rows, columns=ALL_FEATURES, index=rids)


def select_routes(feat: pd.DataFrame, filter_name: str, p: float,
                  models: Dict, seed: Optional[int] = None) -> set:
    """Return the set of route ids to keep (dummy route 0 always kept)."""
    rids   = list(feat.index)
    n_keep = max(1, int(len(rids) * p))

    if filter_name == "random":
        keep = set(random.Random(seed).sample(rids, n_keep))
    else:
        model, scaler, cols = models[filter_name]
        probas = model.predict_proba(scaler.transform(feat[cols].values))[:, 1]
        keep   = {rids[i] for i in np.argsort(probas)[::-1][:n_keep]}
    return keep | {0}


def apply_keep_set(instance, keep: set, all_rids: List[int]) -> None:
    for vt in instance.routes.feasible_routes:
        instance.routes.feasible_routes[vt] = [r for r in all_rids if r in keep]


# ===========================================================================
# Integer solve
# ===========================================================================

def solve_restricted(instance, budget_min: float) -> dict:
    t0 = time.perf_counter()
    cg = run_column_generation(
        instance=instance,
        penalties=PenaltyConfig(),
        problem_cfg=ProblemConfig(T=VT[1]),
        solver_cfg=SolverConfig(time_limit=budget_min),
        env=env,
        integer_solution=True,
        only_integer=True,
        tree_level=0,
    )
    sol = cg.integer_solution or {}
    ob  = sol.get("Objective", {})
    obj = cg.integer_obj if cg.integer_solution else np.inf
    if not np.isnan(cg.obj_bound) and np.isfinite(obj) and abs(obj) > 1e-10:
        gap = abs(obj - cg.obj_bound) / abs(obj)
    else:
        gap = cg.mip_gap
    return {
        "obj":                  obj,
        "gap":                  gap,
        "bound":                cg.obj_bound,
        "transport_cost":       ob.get("transport_cost",       np.nan),
        "below_safety_penalty": ob.get("below_safety_penalty", np.nan),
        "below_zero_penalty":   ob.get("below_zero_penalty",   np.nan),
        "solve_time":           cg.solving_time,
        "wall_time":            time.perf_counter() - t0,
        "selected_route_ids":   sol.get("SelectedRouteIDs", {}),
    }


# ===========================================================================
# Main
# ===========================================================================

def _run_list(filters: List[str], p_values: List[float]) -> List[tuple]:
    """
    p = 1.0 keeps every route, so it is the same solve for every filter:
    it is run once, labelled "full", and only if 1.0 is requested.
    """
    runs = []
    if any(abs(p - 1.0) < 1e-9 for p in p_values):
        runs.append(("full", 1.0, None))
    for f in filters:
        for p in p_values:
            if abs(p - 1.0) < 1e-9:
                continue
            seeds = RANDOM_SEEDS if f == "random" else [None]
            runs += [(f, p, s) for s in seeds]
    return runs


def resolve_instance_ids(available: List[int],
                         instance_ids: Optional[List[int]],
                         instance_range: Optional[Tuple[int, int]]) -> List[int]:
    """Pick the instances to run: explicit list > inclusive range > all."""
    if instance_ids:
        requested = list(instance_ids)
    elif instance_range:
        lo, hi = instance_range
        requested = list(range(lo, hi + 1))
    else:
        return sorted(available)

    missing = [i for i in requested if i not in available]
    if missing:
        print(f"[warn] instances not in SOURCE_PKL, skipped: {missing}")
    ids = [i for i in requested if i in available]
    if not ids:
        raise ValueError(f"No requested instance is in SOURCE_PKL "
                         f"(available: {sorted(available)})")
    return ids


def _save(results: dict, out_pkl: Path, out_csv: Path) -> None:
    with open(out_pkl, "wb") as fh:
        pickle.dump(results, fh)
    rows = [{"filter": f, "p": p, "seed": s, "instance_id": i,
             **{k: v for k, v in r.items() if k not in ("selected_route_ids", "kept_ids")}}
            for (f, p, s, i), r in results.items()]
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main(filters: List[str], p_values: List[float],
         instance_ids: Optional[List[int]] = None,
         instance_range: Optional[Tuple[int, int]] = None) -> dict:
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    out_pkl = Path(RESULTS_DIR) / f"results_{OUTPUT_TAG}.pkl"
    out_csv = Path(RESULTS_DIR) / f"summary_{OUTPUT_TAG}.csv"

    results = {}
    if out_pkl.exists():
        with open(out_pkl, "rb") as fh:
            results = pickle.load(fh)
        print(f"Resuming: {len(results)} solves already in {out_pkl}")

    with open(SOURCE_PKL, "rb") as fh:
        source = pickle.load(fh)[VT]
    with open(INSTANCES_FILE, "rb") as fh:
        raw_instances = pickle.load(fh)
    data = load_data(network_size=NETWORK_SIZE, data_dir=DATA_DIR, one_compartment=True)

    unknown = [f for f in filters if f not in FILTERS]
    if unknown:
        raise ValueError(f"Unknown filters {unknown}; choose from {list(FILTERS)}")

    models = {}
    for name in filters:
        path, cols = FILTERS[name]
        if path is not None:
            model, scaler = joblib.load(path)
            models[name]  = (model, scaler, cols)
            print(f"[{name}] loaded {path} ({len(cols)} features)")

    runs = _run_list(filters, p_values)
    ids  = resolve_instance_ids(list(source.keys()), instance_ids, instance_range)
    print(f"Running {len(runs)} solves on {len(ids)} instances: {ids}")

    for inst_id in ids:
        ir = source[inst_id]
        budget = (FIXED_BUDGET_MIN if FIXED_BUDGET_MIN is not None
                  else TOTAL_TIME_LIMIT - ir.cg.generation_time / 60)
        print(f"\n=== Instance {inst_id} | pool={len(ir.cg.route_records)} routes | "
              f"budget={budget:.2f} min | stored full-pool obj={ir.cg.obj_val:.2f} ===")

        instance = create_instance(raw=raw_instances[inst_id], data=data,
                                   demand_coefficient=ir.demand_coefficient, V=VT[0])
        restore_pool(instance, ir.cg.route_records)
        all_rids = list(range(instance.routes.R))
        feat     = build_feature_frame(instance)

        for (fname, p, seed) in runs:
            key = (fname, p, seed, inst_id)
            if key in results:
                continue

            keep = set(all_rids) if fname == "full" else select_routes(feat, fname, p, models, seed)
            apply_keep_set(instance, keep, all_rids)
            print(f"  [{fname} p={p} seed={seed}] {len(keep) - 1}/{len(feat)} routes")

            res = solve_restricted(instance, budget)
            res.update({"n_kept": len(keep) - 1, "n_pool": len(feat),
                        "budget_min": budget, "kept_ids": sorted(keep)})
            results[key] = res
            print(f"    obj={res['obj']:.2f}  gap={res['gap']:.3f}  "
                  f"solve={res['solve_time']:.0f}s")

            apply_keep_set(instance, set(all_rids), all_rids)   # restore full pool
            _save(results, out_pkl, out_csv)

    print(f"\nDone. Results in {out_pkl} and {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters",   nargs="+", default=FILTERS_TO_RUN,
                        help=f"any of {list(FILTERS)}")
    parser.add_argument("--p",         nargs="+", type=float, default=P_VALUES)
    parser.add_argument("--instances", nargs="+", type=int,   default=None,
                        help="explicit instance ids, e.g. --instances 11 14 17")
    parser.add_argument("--range",     nargs=2,   type=int,   default=None,
                        metavar=("START", "END"),
                        help="inclusive instance range, e.g. --range 11 20")
    args = parser.parse_args()

    ids = args.instances if args.instances else (None if args.range else INSTANCE_IDS)
    rng = tuple(args.range) if args.range else (None if args.instances else INSTANCE_RANGE)
    main(args.filters, args.p, instance_ids=ids, instance_range=rng)