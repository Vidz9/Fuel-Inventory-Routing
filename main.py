"""
main.py
=======
Experiment runner for the IRP project.

For each instance and demand coefficient, runs six methods in sequence:

  1. MILP    — arc-flow MILP solved directly by Gurobi
  2. CG      — LP column generation + integer RMP
  3. BP      — Branch-and-Price seeded from the CG column pool
  4. W       — Heuristic routes + integer RMP (no CG)
  5. WCG     — Heuristic routes + CG + integer RMP
  6. WBP     — Heuristic routes + CG + Branch-and-Price

After each integer solve the solution is printed and node-level statistics
are computed and stored in the MethodResult.

Results are saved as:
    results/<tag>.pkl   — nested InstanceResult objects (full detail)
    results/<tag>.csv   — flat summary DataFrame (for quick inspection)
"""

from __future__ import annotations

import pickle
import joblib
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd

from config import PenaltyConfig, ProblemConfig, SolverConfig
from data.loader import IRPData, load_data
from instance import IRPInstance, RouteRecord, create_instance
from heuristics.route_generator import generate_routes
from heuristics.demand_route_generator import generate_demand_routes
from solvers.milp import MILPSolution, solve_milp_flow
from solvers.column_gen import CGResult, run_column_generation
from branching.branch_and_price import BranchAndPrice
from results import InstanceResult, MethodResult, NodeStat
from plotting import plot_route_level_distributions


# ===========================================================================
# Configuration
# ===========================================================================

N_INSTANCES:          int   = 20
TIME_LIMIT:           float = 20.0          # minutes total per instance
DEMAND_COEFFICIENTS:  list  = [0.85]
NETWORK_SIZE:         str   = "big"
DATA_DIR:             str   = r""           # put directory containing the data
INSTANCES_FILE:       str   = "instances_extended.pkl"
RESULTS_DIR:          str   = "results"
RANDOM_SEED:          int   = 11
PARAMETER_VALUES_VT:  list  = [(3, 5)]
MAX_WARMSTART_ROUTES:      int   = 102    # 0 = no warm start
DEMAND_AWARE_WARMSTART:    bool  = True   # True = demand-guided; False = random 
GEN_FRACTIONS:        list  = []   # generation fraction checkpoints; one BP result per entry
SVM_MODEL_PATH:       str   = r"results\svm_linear_model.pkl"  # path to saved (svm, scaler) tuple
SVM_TOP_P:            float = 0.5           # keep top 40% of routes by predicted probability
USE_SVM_FILTER:       bool  = True         # if False, skip filtered integer solve entirely
USE_RANDOM_FILTER:    bool  = False         # if True, also run random pool compression
RANDOM_FILTER_TOP_P:  float = 0.5           # fraction of pool to keep randomly
RANDOM_FILTER_SEED:   int   = 42            # random seed for reproducibility
USE_LR_FILTER:        bool  = False        # if True, also run logistic regression filter
LR_MODEL_PATH:        str   = r"results\lr_model.pkl"  # path to saved (lr, scaler) tuple
LR_TOP_P:             float = 0.5           # keep top 50% of routes by LR predicted probability
INSTANCE_RANGE:       tuple = (11, 30)      # inclusive range of instances to solve


# ===========================================================================
# Gurobi environment
# ===========================================================================

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()


# ===========================================================================
# obj_over_time normalisation
# ===========================================================================

def _ot_float(value) -> float:
    """
    Safely extract a float objective value from an obj_over_time entry.

    The MILP callback stores plain floats: {seconds: float}.
    The CG/BP callback stores dicts:       {seconds: {"obj": float, ...}}.
    This helper handles both cases uniformly.
    """
    if isinstance(value, dict):
        return float(value.get("obj", np.nan))
    if value is None:
        return np.nan
    return float(value)


def _normalise_ot(raw: Dict) -> Dict[int, float]:
    """Convert any obj_over_time dict to {seconds: float}."""
    return {k: _ot_float(v) for k, v in raw.items()}


# ===========================================================================
# Conversion helpers  (MethodResult from solver outputs)
# ===========================================================================

def _from_milp(sol: MILPSolution, elapsed: float) -> MethodResult:
    ob = sol.objective_breakdown
    return MethodResult(
        obj_val              = sol.obj_val,
        lp_obj               = None,
        transport_cost       = ob.get("transport_cost",       np.nan),
        below_safety_penalty = ob.get("below_safety_penalty", np.nan),
        below_zero_penalty   = ob.get("below_zero_penalty",   np.nan),
        mip_gap              = sol.gap,
        generation_time      = 0.0,
        solving_time         = elapsed,
        total_time           = elapsed,
        n_routes             = 0,
        # MILP stores {seconds: float} directly — normalise for uniformity
        obj_over_time        = _normalise_ot(sol.objective_over_time),
        levels               = sol.levels,
        routes               = sol.routes,
        loadings             = sol.loadings,
        unloadings           = sol.unloadings,
        flows                = sol.flows,
    )


def _from_cg(result: CGResult, n_routes: int, instance) -> MethodResult:
    sol = result.integer_solution or {}
    ob  = sol.get("Objective", {})
    # "Progression over time" is {seconds: {"obj": float, "bound": float, ...}}
    # — extract only the obj float so obj_over_time is always {seconds: float}
    raw_ot = sol.get("Progression over time", {})
    return MethodResult(
        obj_val              = result.integer_obj,
        lp_obj               = result.lp_obj,
        transport_cost       = ob.get("transport_cost",       np.nan),
        below_safety_penalty = ob.get("below_safety_penalty", np.nan),
        below_zero_penalty   = ob.get("below_zero_penalty",   np.nan),
        generation_time      = result.generation_time,
        solving_time         = result.solving_time,
        total_time           = result.generation_time + result.solving_time,
        n_routes             = n_routes,
        obj_over_time        = _normalise_ot(raw_ot),
        levels               = sol.get("Levels"),
        routes               = sol.get("Routes", {}),
        loadings             = sol.get("Loadings", {}),
        unloadings           = sol.get("Unloadings", {}),
        selected_route_ids   = sol.get("SelectedRouteIDs", {}),
        route_records        = dict(instance.routes.route_records),
        mip_gap              = result.mip_gap,
        iteration_log        = result.iteration_log or [],
    )


def _from_bp(
    bp:        BranchAndPrice,
    solution:  Dict,
    extra_gen: float,
    n_routes:  int,
    instance,
) -> MethodResult:
    sol   = solution or {}
    ob    = sol.get("Objective", {})
    raw_ot = sol.get("Progression over time", {})
    return MethodResult(
        obj_val              = bp.best_obj,
        lp_obj               = None,
        transport_cost       = ob.get("transport_cost",       np.nan),
        below_safety_penalty = ob.get("below_safety_penalty", np.nan),
        below_zero_penalty   = ob.get("below_zero_penalty",   np.nan),
        generation_time      = bp.generation_time_bp + extra_gen,
        solving_time         = bp.solving_time_bp,
        total_time           = bp.generation_time_bp + extra_gen + bp.solving_time_bp,
        n_routes             = n_routes,
        bp_nodes_explored    = bp.nodes_explored,
        obj_over_time        = _normalise_ot(raw_ot),
        levels               = sol.get("Levels"),
        routes               = sol.get("Routes", {}),
        loadings             = sol.get("Loadings", {}),
        unloadings           = sol.get("Unloadings", {}),
        selected_route_ids   = sol.get("SelectedRouteIDs", {}),
        route_records        = dict(instance.routes.route_records),
        mip_gap              = bp.final_mip_gap,
        iteration_log        = bp.iteration_log,
    )


# ===========================================================================
# Node-level statistics
# ===========================================================================

def _compute_node_stats(
    instance: IRPInstance,
    mr:       MethodResult,
) -> Tuple[Dict[int, NodeStat], Dict[int, NodeStat]]:
    """
    Compute per-station and per-depot solution statistics from a MethodResult.

    Parameters
    ----------
    instance : IRPInstance (for node metadata and demand)
    mr       : MethodResult with populated levels / loadings / unloadings / routes

    Returns
    -------
    (station_stats, depot_stats)  each a dict  node_idx -> NodeStat
    """
    if mr.levels is None:
        return {}, {}

    data    = instance.data
    Nodes   = data.Nodes
    T       = mr.levels.shape[1]

    N = data.N

    # The flow MILP stores arc flows in mr.flows instead of pre-aggregated
    # loadings/unloadings (those are left empty).  Derive them here when needed.
    # For every (v, t): net_in[node] = sum(incoming flows) - sum(outgoing flows).
    # Positive net_in  → fuel delivered to station (unloading).
    # Negative net_in  → fuel loaded from depot    (loading).
    unloadings = mr.unloadings
    loadings   = mr.loadings

    if not unloadings and mr.flows:
        # Derive from arc flows: mr.flows[(v,t)] = {(i,j): value}
        unloadings = {}
        loadings   = {}
        for (v, t), arc_dict in mr.flows.items():
            net = np.zeros(N)
            for (i, j), val in arc_dict.items():
                net[j] += val   # incoming to j
                net[i] -= val   # outgoing from i
            unloadings[(v, t)] = np.maximum(net,  0.0)  # positive → delivered
            loadings[(v, t)]   = np.maximum(-net, 0.0)  # negative → loaded

    # Build delivered[node_idx, t] = sum over trucks of unloading at that node on day t
    delivered = np.zeros((N, T))
    for (v, t), arr in unloadings.items():
        delivered[:, t] += arr

    # Build loaded[node_idx, t] = sum over trucks of loading at that node on day t
    loaded = np.zeros((N, T))
    for (v, t), arr in loadings.items():
        loaded[:, t] += arr

    # Count visits per node: a truck visits a node if it has any incoming
    # edge in its route on that day.
    # mr.routes[(v,t)] can be either:
    #   - an (N x N) numpy array  (CG/BP, compartment MILP)
    #   - a list of (i,j) tuples  (flow MILP)
    # Normalise both into an (N x N) array before summing.
    visits = np.zeros(N, dtype=int)
    for (v, t), edge_matrix in mr.routes.items():
        if edge_matrix is None:
            continue
        em = np.asarray(edge_matrix)
        if em.ndim == 2 and em.shape == (N, N):
            # Already the right shape
            pass
        elif em.ndim == 2 and em.shape[1] == 2:
            # List of (i,j) tuples converted to array of shape (n_edges, 2)
            dense = np.zeros((N, N))
            for i, j in em:
                dense[int(i), int(j)] = 1
            em = dense
        else:
            # Unexpected format — skip this entry
            continue
        # A node is visited if it has at least one incoming edge
        incoming = em.sum(axis=0)
        visits += (incoming > 0.5).astype(int)

    # ---- Station stats ----
    station_stats: Dict[int, NodeStat] = {}
    for idx in data.station_indices:
        node         = Nodes[idx]
        node_demand  = instance.demand.get(node, [0.0] * T)
        node_levels  = list(mr.levels[idx, :])
        node_deliv   = list(delivered[idx, :])
        safety       = getattr(node, "safety", 0.0)

        days_below_safety = sum(1 for lv in node_levels if lv < safety)
        days_below_zero   = sum(1 for lv in node_levels if lv < 0)

        station_stats[idx] = NodeStat(
            name              = node.location.name,
            node_idx          = idx,
            initial_level     = instance.initial_levels.get(node, np.nan),
            demand            = list(node_demand),
            delivered         = node_deliv,
            levels            = node_levels,
            safety_stock      = safety,
            days_below_safety = days_below_safety,
            days_below_zero   = days_below_zero,
            n_visits          = int(visits[idx]),
            qty_loaded        = 0.0,
        )

    # ---- Depot stats ----
    depot_stats: Dict[int, NodeStat] = {}
    for idx in data.depot_indices:
        node = Nodes[idx]
        depot_stats[idx] = NodeStat(
            name      = node.location.name,
            node_idx  = idx,
            n_visits  = int(visits[idx]),
            qty_loaded= float(loaded[idx, :].sum()),
        )

    return station_stats, depot_stats


# ===========================================================================
# Solution printer
# ===========================================================================

def _print_solution(
    instance: IRPInstance,
    mr:       MethodResult,
    label:    str,
) -> None:
    """
    Print a human-readable summary of an integer solution.

    Shows:
      - Which vehicle selected which route on which day (with stop names)
      - Per-station table: demand / delivered / level / visits / violations
      - Per-depot table:   visits / quantity loaded
    """
    if mr.levels is None or not mr.routes:
        print(f"\n[{label}] No integer solution available.")
        return

    data  = instance.data
    Nodes = data.Nodes
    T     = mr.levels.shape[1]
    V     = instance.data.V

    sep = "-" * 64

    # ------------------------------------------------------------------ routes
    print(f"\n{'='*64}")
    print(f"  {label} — Route assignments")
    print(f"{'='*64}")

    for v in range(V):
        for t in range(T):
            edge_matrix = mr.routes.get((v, t))
            if edge_matrix is None:
                continue
            # Normalise to N×N — edge_matrix can be an (N,N) array
            # (CG/BP) or a list of (i,j) tuples (flow MILP).
            raw = np.asarray(edge_matrix)
            N   = data.N
            if raw.ndim == 2 and raw.shape == (N, N):
                em = raw
            elif raw.ndim == 2 and raw.shape[1] == 2:
                em = np.zeros((N, N))
                for i, j in raw:
                    em[int(i), int(j)] = 1
            else:
                continue

            if em.sum() < 0.5:
                continue   # truck stayed at parking — no route

            # Reconstruct stop sequence from edge matrix
            parking = data.parking_index
            stop    = parking
            path    = [stop]
            visited = set()
            while True:
                nexts = np.where(em[stop, :] > 0.5)[0]
                if len(nexts) == 0:
                    break
                nxt = int(nexts[0])
                if nxt in visited:
                    break
                path.append(nxt)
                visited.add(stop)
                stop = nxt
                if stop == parking and len(path) > 1:
                    break

            stop_names = " → ".join(
                f"{Nodes[s].location.name}({Nodes[s].location_type[0]})"
                for s in path
            )
            print(f"  Truck {v+1}  Day {t+1}:  {stop_names}")

    # --------------------------------------------------------------- stations
    print(f"\n{sep}")
    print(f"  {label} — Station statistics")
    print(f"{sep}")

    hdr = f"  {'Station':<20} {'Visits':>6} {'∑Demand':>9} {'∑Delivered':>11} "
    hdr += f"{'Bel.Safety':>10} {'Bel.Zero':>8}"
    print(hdr)
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*11} {'-'*10} {'-'*8}")

    for idx, st in mr.station_stats.items():
        print(
            f"  {st.name:<20} "
            f"{st.n_visits:>6} "
            f"{sum(st.demand or []):>9.1f} "
            f"{sum(st.delivered or []):>11.1f} "
            f"{st.days_below_safety:>10} "
            f"{st.days_below_zero:>8}"
        )

    # Detail rows: demand vs delivered vs level per period
    for idx, st in mr.station_stats.items():
        print(f"\n  {st.name}  (safety={st.safety_stock:.1f}  "
              f"initial={st.initial_level:.1f})")
        day_hdr = "    " + "".join(f"  Day{t+1:>2}" for t in range(T))
        print(day_hdr)
        print("    Demand   " + "".join(f"{d:>7.1f}" for d in (st.demand or [])))
        print("    Delivered" + "".join(f"{d:>7.1f}" for d in (st.delivered or [])))
        print("    Level    " + "".join(f"{lv:>7.1f}" for lv in (st.levels or [])))

    # ----------------------------------------------------------------- depots
    print(f"\n{sep}")
    print(f"  {label} — Depot statistics")
    print(f"{sep}")

    hdr2 = f"  {'Depot':<20} {'Visits':>6} {'Qty Loaded':>11}"
    print(hdr2)
    print(f"  {'-'*20} {'-'*6} {'-'*11}")

    for idx, dp in mr.depot_stats.items():
        print(f"  {dp.name:<20} {dp.n_visits:>6} {dp.qty_loaded:>11.1f}")

    print()


# ===========================================================================
# Summary DataFrame
# ===========================================================================

def to_summary_df(results: Dict) -> pd.DataFrame:
    """
    Derive a flat summary DataFrame from the nested InstanceResult dict.

    results[demand_coeff][(V, T)][instance_id] = InstanceResult
    """
    def _safe_round(v, n):
        try:
            return round(float(v), n)
        except (TypeError, ValueError):
            return np.nan

    rows = []
    for demand_coeff, vt_dict in results.items():
        for (V, T), inst_dict in vt_dict.items():
            for inst_id, r in inst_dict.items():
                row: Dict = {
                    "demand_coefficient":    demand_coeff,
                    "V":                     V,
                    "T":                     T,
                    "instance_id":           inst_id,
                    "total_demand":          _safe_round(r.total_demand, 1),
                    "demand_capacity_ratio": _safe_round(r.demand_capacity_ratio, 3),
                }

                # Build tag -> MethodResult list: fixed methods + bp_fractions
                _tagged = [
                    ("MILP", r.milp),
                    ("CG",   r.cg),
                ]
                for _frac, _mr in sorted((r.bp_fractions or {}).items(),
                                         key=lambda x: str(x[0])):
                    if isinstance(_frac, tuple):
                        # (alpha, "filtered") key
                        _tag = f"BP_{int(_frac[0]*100)}_filtered"
                    else:
                        _tag = f"BP_{int(_frac*100)}"
                    _tagged.append((_tag, _mr))
                for tag, mr in _tagged:
                    if mr is None:
                        continue

                    ot = mr.obj_over_time   # always {seconds: float} now
                    row.update({
                        f"{tag}_obj":          _safe_round(mr.obj_val, 1),
                        f"{tag}_lp_obj":       _safe_round(mr.lp_obj, 1) if mr.lp_obj is not None else np.nan,
                        f"{tag}_transport":    _safe_round(mr.transport_cost, 1),
                        f"{tag}_below_safety": _safe_round(mr.below_safety_penalty, 1),
                        f"{tag}_below_zero":   _safe_round(mr.below_zero_penalty, 1),
                        f"{tag}_mip_gap":      _safe_round(mr.mip_gap, 4),
                        f"{tag}_gen_time":     _safe_round(mr.generation_time, 1),
                        f"{tag}_solve_time":   _safe_round(mr.solving_time, 1),
                        f"{tag}_total_time":   _safe_round(mr.total_time, 1),
                        f"{tag}_routes":       mr.n_routes,
                        f"{tag}_obj_60":       _safe_round(ot.get(60),  1),
                        f"{tag}_obj_120":      _safe_round(ot.get(120), 1),
                        f"{tag}_obj_300":      _safe_round(ot.get(300), 1),
                        f"{tag}_obj_600":      _safe_round(ot.get(600), 1),
                    })
                    if mr.bp_nodes_explored is not None:
                        row[f"{tag}_bp_nodes"] = mr.bp_nodes_explored

                rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# Per-instance pipeline
# ===========================================================================

def _attach_stats(mr: MethodResult, instance: IRPInstance, label: str) -> None:
    """Compute node stats, attach to mr, and print solution."""
    mr.station_stats, mr.depot_stats = _compute_node_stats(instance, mr)
    _print_solution(instance, mr, label)


def _remaining(t_start: float, total_limit_min: float, min_limit_min: float = 0.5) -> float:
    """Minutes remaining from total_limit after t_start, floored at min_limit."""
    elapsed = (time.perf_counter() - t_start) / 60
    return max(total_limit_min - elapsed, min_limit_min)


def _inject_warmstart_routes(instance: IRPInstance, shared_routes: list) -> None:
    """
    Copy pre-generated warm-start routes into instance.routes.

    Parameters
    ----------
    instance       : target IRPInstance (mutated in-place)
    shared_routes  : list of (stops, route_edges, duration, n_stations, n_depots)
                     as built in main() before the instance loop
    """
    routes  = instance.routes
    V       = instance.data.V
    T       = len(next(iter(instance.demand.values())))
    N       = instance.data.N

    for stops, edges, duration, n_stn, n_dep in shared_routes:
        R = routes.R

        # Skip duplicates
        if stops in routes.new_routes.values():
            continue

        routes.new_routes[R]      = list(stops)
        routes.route_durations.append(float(duration))
        routes.route_edges[R]     = edges.copy()
        routes.reduced_costs[R]   = 0.0

        for i in range(len(stops) - 1):
            edge = (stops[i], stops[i + 1])
            if edge in routes.edge_routes:
                routes.edge_routes[edge].append(R)
            else:
                routes.edge_routes[edge] = [R]

        for v in range(V):
            for t in range(T):
                routes.feasible_routes[(v, t)].append(R)

        routes.route_records[R] = RouteRecord(
            route_id     = R,
            origin       = "warmstart",
            stops        = list(stops),
            duration     = float(duration),
            n_stations   = n_stn,
            n_depots     = n_dep,
            tree_level   = None,
            reduced_cost = None,
        )

        routes.R += 1



def _generation_quality_encode(rec):
    """One-hot encode generation quality for SVM feature."""
    gq = rec.generation_quality if rec.generation_quality else "heuristic"
    if rec.origin == "warmstart":       return [1, 0, 0, 0]
    if gq == "heuristic":               return [0, 1, 0, 0]
    if gq == "suboptimal":              return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def _svm_filter_pool(instance, svm_model, svm_scaler, top_p, n_stations_net,
                     demand_dict=None):
    """
    Score all non-dummy routes and restrict instance.routes to the top_p
    fraction by predicted selection probability.
    Modifies instance.routes.feasible_routes in-place.

    demand_dict : instance.demand — dict Node -> list[float].
                  Used to compute exact total demand for visited stations.
    """
    routes  = instance.routes
    records = routes.route_records

    # Build node_number -> total demand lookup
    _demand_by_node = {}
    if demand_dict is not None:
        for _node, _dvals in demand_dict.items():
            _demand_by_node[_node.number] = float(sum(_dvals))

    rids, rows = [], []
    for rid, rec in records.items():
        if rec.origin == "dummy":
            continue
        rc = rec.reduced_cost if rec.reduced_cost is not None else 0.0
        gq = _generation_quality_encode(rec)
        # Exact total demand for stations visited by this route
        total_demand = sum(
            _demand_by_node.get(stop, 0.0)
            for stop in rec.stops
            if stop in _demand_by_node
        )
        rows.append(gq + [
            float(rc),
            int(rid),
            int(rec.n_depots),
            int(rec.n_stations),
            float(rec.duration),
            int(n_stations_net),
            float(total_demand),
            float(rec.usage_ratio),
            float(rec.avg_lp_value),
        ])
        rids.append(rid)

    if not rows:
        return set()

    X      = svm_scaler.transform(np.array(rows))
    probas = svm_model.predict_proba(X)[:, 1]

    k         = max(1, int(len(probas) * top_p))
    top_idx   = np.argsort(probas)[::-1][:k]
    keep_rids = {rids[i] for i in top_idx} | {0}   # always keep dummy

    for vt in routes.feasible_routes:
        routes.feasible_routes[vt] = [
            r for r in routes.feasible_routes[vt] if r in keep_rids
        ]

    print(f"  [SVM filter] Pool: {len(rids)} → {len(keep_rids)-1} routes "
          f"(top {top_p*100:.0f}%)")
    return keep_rids

def _random_filter_pool(instance, top_p, seed=None):
    """Randomly retain top_p fraction of non-dummy routes."""
    import random as _rand
    rng    = _rand.Random(seed)
    routes = instance.routes
    rids   = [rid for rid, rec in routes.route_records.items()
              if rec.origin != "dummy"]
    n_keep = max(1, int(len(rids) * top_p))
    keep   = set(rng.sample(rids, n_keep)) | {0}
    for vt in routes.feasible_routes:
        routes.feasible_routes[vt] = [
            r for r in routes.feasible_routes[vt] if r in keep
        ]
    print(f"  [Random filter] Pool: {len(rids)} → {n_keep} routes "
          f"({top_p*100:.0f}%, seed={seed})")
    return keep


def _lr_filter_pool(instance, lr_model, lr_scaler, top_p, n_stations_net,
                    demand_dict=None):
    """
    Score all non-dummy routes using logistic regression and restrict
    instance.routes to the top_p fraction by predicted selection probability.
    Modifies instance.routes.feasible_routes in-place.
    """
    routes  = instance.routes
    records = routes.route_records

    _demand_by_node = {}
    if demand_dict is not None:
        for _node, _dvals in demand_dict.items():
            _demand_by_node[_node.number] = float(sum(_dvals))

    rids, rows = [], []
    for rid, rec in records.items():
        if rec.origin == "dummy":
            continue
        rc = rec.reduced_cost if rec.reduced_cost is not None else 0.0
        gq = _generation_quality_encode(rec)
        total_demand = sum(
            _demand_by_node.get(stop, 0.0)
            for stop in rec.stops
            if stop in _demand_by_node
        )
        rows.append(gq + [
            float(rc), int(rid), int(rec.n_depots), int(rec.n_stations),
            float(rec.duration), int(n_stations_net), float(total_demand),
            float(rec.usage_ratio), float(rec.avg_lp_value),
        ])
        rids.append(rid)

    if not rows:
        return set()

    X      = lr_scaler.transform(np.array(rows))
    probas = lr_model.predict_proba(X)[:, 1]

    k         = max(1, int(len(probas) * top_p))
    top_idx   = np.argsort(probas)[::-1][:k]
    keep_rids = {rids[i] for i in top_idx} | {0}

    for vt in routes.feasible_routes:
        routes.feasible_routes[vt] = [
            r for r in routes.feasible_routes[vt] if r in keep_rids
        ]

    print(f"  [LR filter] Pool: {len(rids)} → {len(keep_rids)-1} routes "
          f"(top {top_p*100:.0f}%)")
    return keep_rids


def run_instance(
    raw_instance:            dict,
    data:                    IRPData,
    inst_id:                 int,
    V:                       int,
    T:                       int,
    demand_coefficient:      float,
    time_limit:              float,
    shared_warmstart_routes: list  = None,
    gen_fractions:           list  = None,
    svm_model                      = None,  # trained SVC
    svm_scaler                     = None,  # fitted StandardScaler
    svm_top_p:               float = 0.4,
    use_svm_filter:          bool  = True,  # if False skip filtered solve
    use_random_filter:       bool  = False, # if True also run random filter
    random_filter_top_p:     float = 0.5,
    random_filter_seed:      int   = 42,
    lr_model                       = None,  # trained LogisticRegression
    lr_scaler                      = None,  # fitted StandardScaler
    lr_top_p:                float = 0.5,
    use_lr_filter:           bool  = False, # if True also run LR filter
) -> InstanceResult:
    """Run MILP, CG and BP for one instance.
    If max_warmstart_routes > 0, routes are pre-generated to warm-start CG/BP."""
    gen_fractions = gen_fractions or GEN_FRACTIONS
    problem_cfg = ProblemConfig(T=T)
    penalties   = PenaltyConfig()
    solver_cfg  = SolverConfig(time_limit=time_limit)

    result = InstanceResult(
        instance_id=inst_id, V=V, T=T,
        demand_coefficient=demand_coefficient,
        network_size=NETWORK_SIZE,
    )

    # ---- Shared instance (MILP, CG, BP) ----
    instance = create_instance(
        raw=raw_instance, data=data,
        demand_coefficient=demand_coefficient, V=V,
    )

    total_demand   = float(sum(sum(d) for d in instance.demand.values()))
    total_capacity = float(sum(
        np.sum(instance.data.Trucks[v].compartment_capacities)
        for v in range(instance.data.V)
    ))
    result.total_demand          = total_demand
    result.demand_capacity_ratio = total_demand / total_capacity if total_capacity > 0 else np.nan

    # ------------------------------------------------------------------ MILP
    print("  [MILP] Solving arc-flow MILP ...")
    t0       = time.perf_counter()
    milp_sol = solve_milp_flow(instance, penalties, problem_cfg, solver_cfg, env)
    result.milp = _from_milp(milp_sol, time.perf_counter() - t0)
    _attach_stats(result.milp, instance, "MILP")
    print(f"  [MILP] obj={result.milp.obj_val:.2f}  gap={result.milp.mip_gap:.3f}  "
          f"time={result.milp.total_time:.1f}s")

    # ----------------------------------------- Optional warm-start for CG/BP
    if MAX_WARMSTART_ROUTES > 0 and DEMAND_AWARE_WARMSTART:
        # Demand-aware: generate per-instance routes using urgency scores
        print(f"  [WS] Generating {MAX_WARMSTART_ROUTES} demand-aware warm-start routes ...")
        generate_demand_routes(instance, problem_cfg,
                               max_routes=MAX_WARMSTART_ROUTES)
        for r in range(1, instance.routes.R):
            instance.routes.reduced_costs[r] = 0.0
        print(f"  [WS] {instance.routes.R - 1} demand-aware routes generated.")
    elif shared_warmstart_routes:
        # Topology-only: inject pre-generated shared routes
        print(f"  [WS] Injecting {len(shared_warmstart_routes)} shared warm-start routes ...")
        _inject_warmstart_routes(instance, shared_warmstart_routes)
        print(f"  [WS] {instance.routes.R - 1} routes injected.")

    # ------------------------------------------------------------------- CG
    # LP generation only — integer solve happens after BP tree exploration
    print("  [CG] Running root-node column generation (LP only) ...")
    t0_cg         = time.perf_counter()
    cg_gen_result = run_column_generation(
        instance=instance, penalties=penalties,
        problem_cfg=problem_cfg,
        solver_cfg=SolverConfig(time_limit=time_limit),
        env=env, lower_bound=0.0,
        integer_solution=False, only_integer=False,
    )
    R_root    = instance.routes.R          # pool size after root CG
    gen_time  = cg_gen_result.generation_time / 60   # minutes

    print(f"  [CG]  LP={cg_gen_result.lp_obj:.2f}  "
          f"routes={R_root}  gen={gen_time*60:.1f}s")

    # ------------------------------------------------------------------- BP
    # Tree exploration + all integer solves, all anchored to t0_cg
    print(f"  [BP] Starting branch-and-price with checkpoints {gen_fractions} ...")
    bp = BranchAndPrice(
        instance=instance, penalties=penalties,
        problem_cfg=problem_cfg, solver_cfg=solver_cfg,
        env=env, lower_bound=cg_gen_result.lp_obj,
    )
    # Build SVM filter callable
    _n_stn_net = len(instance.data.station_indices)
    if use_svm_filter and svm_model is not None and svm_scaler is not None:
        def _svm_fn(_inst):
            _svm_filter_pool(_inst, svm_model, svm_scaler, svm_top_p, _n_stn_net,
                             instance.demand)
    else:
        _svm_fn = None

    if use_random_filter:
        def _rand_fn(_inst):
            _random_filter_pool(_inst, random_filter_top_p, seed=random_filter_seed)
    else:
        _rand_fn = None

    if use_lr_filter and lr_model is not None and lr_scaler is not None:
        def _lr_fn(_inst):
            _lr_filter_pool(_inst, lr_model, lr_scaler, lr_top_p, _n_stn_net,
                            instance.demand)
    else:
        _lr_fn = None

    bp_results = bp.run_multi_fraction(
        total_time_limit=time_limit,
        gen_fractions=gen_fractions,
        method_start=t0_cg,
        gen_time=gen_time,
        root_R=R_root,
        lp_obj=cg_gen_result.lp_obj,
        svm_filter_fn=_svm_fn,
        random_filter_fn=_rand_fn,
        lr_filter_fn=_lr_fn,
    )

    # result.cg = root node integer solve (fraction 0 in bp_results)
    root_sol, root_cg, root_obj, root_gap, _root_r, _root_gen = bp_results.pop(0)
    root_cg_full = CGResult(
        lp_obj=cg_gen_result.lp_obj,
        integer_obj=root_obj,
        lp_solution=cg_gen_result.lp_solution,
        integer_solution=root_sol,
        generation_time=cg_gen_result.generation_time,
        solving_time=root_cg.solving_time if root_cg else 0.0,
        mip_gap=root_gap,
        obj_bound=root_cg.obj_bound if root_cg else np.nan,
        iteration_log=cg_gen_result.iteration_log or [],
    )
    result.cg = _from_cg(root_cg_full, R_root, instance)
    _attach_stats(result.cg, instance, "CG")
    print(f"  [CG]  int={result.cg.obj_val:.2f}  gap={result.cg.mip_gap:.3f}  "
          f"solve={result.cg.solving_time:.1f}s")

    # result.bp_fractions = one MethodResult per gen fraction
    result.bp_fractions = {}
    for frac, (sol, frac_cg, cp_obj, cp_gap, r_snap, cp_gen_time) in bp_results.items():
        # cp_gen_time = seconds of BP tree exploration up to this checkpoint
        # total generation = root CG gen time + BP tree time at checkpoint
        mr_bp = _from_bp(bp, sol or {}, gen_time * 60 + cp_gen_time, r_snap, instance)
        mr_bp.obj_val      = cp_obj
        mr_bp.mip_gap      = cp_gap
        mr_bp.obj_bound    = frac_cg.obj_bound    if frac_cg else np.nan
        mr_bp.solving_time = frac_cg.solving_time if frac_cg else 0.0
        mr_bp.total_time   = mr_bp.generation_time + mr_bp.solving_time
        _attach_stats(mr_bp, instance, f"BP-{frac}")
        result.bp_fractions[frac] = mr_bp
        print(f"  [BP α={frac}]  obj={mr_bp.obj_val:.2f}  nodes={bp.nodes_explored}  "
              f"routes={r_snap}  solve={mr_bp.solving_time:.1f}s")
    # Store SVM-filtered results under (frac, "filtered") keys
    # Store random-filtered results under (frac, "random_filtered") keys
    for _fkey, _fval in bp_results.items():
        if not (isinstance(_fkey, tuple) and len(_fkey) == 2
                and _fkey[1] in ("filtered", "random_filtered", "lr_filtered")):
            continue
        _ffrac = _fkey[0]
        _fsol, _fcg, _fobj, _fgap, _fr_snap, _fgen_time = _fval
        mr_filt = _from_bp(bp, _fsol or {}, gen_time * 60 + _fgen_time,
                           _fr_snap, instance)
        mr_filt.obj_val      = _fobj
        mr_filt.mip_gap      = _fgap
        mr_filt.obj_bound    = _fcg.obj_bound    if _fcg else np.nan
        mr_filt.solving_time = _fcg.solving_time if _fcg else 0.0
        mr_filt.total_time   = mr_filt.generation_time + mr_filt.solving_time
        _ftag = _fkey[1]  # "filtered" or "random_filtered"
        _attach_stats(mr_filt, instance, f"BP-{_ffrac}-{_ftag}")
        result.bp_fractions[(_ffrac, _ftag)] = mr_filt
        print(f"  [BP α={_ffrac} {_ftag}]  obj={mr_filt.obj_val:.2f}  "
              f"routes={_fr_snap}  solve={mr_filt.solving_time:.1f}s")

    if bp_results:
        _plain_fracs = [f for f in bp_results
                        if not (isinstance(f, tuple) and len(f)==2
                                and f[1]=="filtered")]
        if _plain_fracs:
            last_frac     = sorted(_plain_fracs)[-1]
            result.bp     = result.bp_fractions.get(last_frac)

    return result


# ===========================================================================
# Main loop
# ===========================================================================

def main() -> Dict:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = load_data(
        network_size=NETWORK_SIZE,
        data_dir=DATA_DIR,
        one_compartment=True,
    )
    print(f"Network: {data.N} nodes | {data.V} trucks | "
          f"{data.P} products | {data.n_stations} stations | {data.n_depots} depots")

    with open(INSTANCES_FILE, "rb") as f:
        instances = pickle.load(f)

    # Load SVM model for route pool filtering (None if not found)
    _svm_model, _svm_scaler = None, None
    try:
        _svm_model, _svm_scaler = joblib.load(SVM_MODEL_PATH)
        print(f"[SVM] Model loaded from {SVM_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[SVM] Model not found at {SVM_MODEL_PATH} — filtered results will be skipped.")

    _lr_model, _lr_scaler = None, None
    try:
        _lr_model, _lr_scaler = joblib.load(LR_MODEL_PATH)
        print(f"[LR] Model loaded from {LR_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[LR] Model not found at {LR_MODEL_PATH} — LR filtered results will be skipped.")

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    all_results: Dict = {}

    for demand_coeff in DEMAND_COEFFICIENTS:
        print(f"\n{'='*64}\nDemand coefficient: {demand_coeff}\n{'='*64}")
        all_results[demand_coeff] = {}

        for V, T in PARAMETER_VALUES_VT:
            print(f"\nV={V}, T={T}")
            all_results[demand_coeff][(V, T)] = {}

            # ---- Shared warm-start (topology-only, random generator) ----
            # Only used when DEMAND_AWARE_WARMSTART=False.
            # Demand-aware routes are generated per-instance inside run_instance.
            shared_warmstart_routes = []
            if MAX_WARMSTART_ROUTES > 0 and not DEMAND_AWARE_WARMSTART:
                print(f"  [WS] Generating {MAX_WARMSTART_ROUTES} shared warm-start routes ...")
                _ws_instance = create_instance(
                    raw=instances[INSTANCE_RANGE[0]],
                    data=data, V=V,
                    demand_coefficient=demand_coeff,
                )
                _ws_problem_cfg = ProblemConfig(T=T)
                generate_routes(_ws_instance, _ws_problem_cfg,
                                max_routes=MAX_WARMSTART_ROUTES)
                for r in range(1, _ws_instance.routes.R):
                    rec = _ws_instance.routes.route_records[r]
                    shared_warmstart_routes.append((
                        _ws_instance.routes.new_routes[r],
                        _ws_instance.routes.route_edges[r],
                        _ws_instance.routes.route_durations[r],
                        rec.n_stations,
                        rec.n_depots,
                    ))
                print(f"  [WS] {len(shared_warmstart_routes)} shared routes ready.")

            for inst_id in range(INSTANCE_RANGE[0], INSTANCE_RANGE[1] + 1):
                print(f"\n--- Instance {inst_id} / {N_INSTANCES} ---")

                inst_result = run_instance(
                    raw_instance=instances[inst_id],
                    data=data, inst_id=inst_id,
                    V=V, T=T,
                    demand_coefficient=demand_coeff,
                    time_limit=TIME_LIMIT,
                    shared_warmstart_routes=shared_warmstart_routes,
                    gen_fractions=GEN_FRACTIONS,
                    svm_model=_svm_model,
                    svm_scaler=_svm_scaler,
                    svm_top_p=SVM_TOP_P,
                    use_svm_filter=USE_SVM_FILTER,
                    use_random_filter=USE_RANDOM_FILTER,
                    random_filter_top_p=RANDOM_FILTER_TOP_P,
                    random_filter_seed=RANDOM_FILTER_SEED,
                    lr_model=_lr_model,
                    lr_scaler=_lr_scaler,
                    lr_top_p=LR_TOP_P,
                    use_lr_filter=USE_LR_FILTER,
                )
                all_results[demand_coeff][(V, T)][inst_id] = inst_result
                _save(all_results, demand_coeff,
                      max_warmstart_routes=MAX_WARMSTART_ROUTES,
                      gen_fractions=GEN_FRACTIONS,
                      time_limit=TIME_LIMIT)

        summary = to_summary_df({demand_coeff: all_results[demand_coeff]})
        numeric = summary.select_dtypes(include="number")
        print(f"\nAverages for demand_coeff={demand_coeff}:")
        print(numeric.mean().round(1).to_string())

        # Plot and save route level distribution histograms
        for V, T in PARAMETER_VALUES_VT:
            plot_route_level_distributions(
                results      = {demand_coeff: all_results[demand_coeff]},
                results_dir  = RESULTS_DIR,
                demand_coeff = demand_coeff,
                V            = V,
                T            = T,
                save         = True,
                show         = False,
            )

    return all_results


def _save(all_results: Dict, demand_coeff: float,
          max_warmstart_routes: int, gen_fractions: list,
          time_limit: float) -> None:
    fracs_str = "_".join(str(int(f * 100)) for f in sorted(gen_fractions))
    tag       = (f"{demand_coeff}demand_{NETWORK_SIZE}"
                 f"_ws{max_warmstart_routes}"
                 f"_gf{fracs_str}"
                 f"_t{int(time_limit)}")
    rich_path = Path(RESULTS_DIR) / f"results_{tag}.pkl"
    csv_path  = Path(RESULTS_DIR) / f"summary_{tag}.csv"

    with open(rich_path, "wb") as f:
        pickle.dump(all_results[demand_coeff], f)

    try:
        summary = to_summary_df({demand_coeff: all_results[demand_coeff]})
        summary.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"  [warn] Could not write CSV summary: {e}")

    print(f"  Saved → {rich_path}  |  {csv_path}")


if __name__ == "__main__":
    main()