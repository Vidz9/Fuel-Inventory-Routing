"""
solvers/column_gen.py
=====================
Column-generation procedure for the IRP LP relaxation and integer re-solve.

The loop alternates between:
  1. Solving the Restricted Master Problem (RMP) to get dual variables.
  2. Solving a pricing subproblem for each (truck, day) to find a new
     route with a negative reduced cost.

When no more improving columns exist the LP relaxation is optimal.
Optionally an integer re-solve is performed on the final column pool.

Public API
----------
    CGResult          — typed container for all outputs
    run_column_generation(instance, penalties, problem_cfg, solver_cfg, env,
                          lower_bound, integer_solution, only_integer,
                          tree_level)
                      -> CGResult
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import PenaltyConfig, ProblemConfig, SolverConfig
from instance import IRPInstance, RouteRecord
from solvers.master import RMPSolution, build_rmp_model, solve_rmp
from solvers.subproblem import SubproblemResult, solve_subproblem



# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class CGResult:
    """
    All outputs of the column-generation procedure.

    Attributes
    ----------
    lp_obj              : optimal LP relaxation objective (np.inf if CG failed)
    integer_obj         : objective of the integer re-solve (np.inf if skipped)
    lp_solution         : RMPSolution from the final LP solve
    integer_solution    : solution dict from the integer re-solve (or None)
    generation_time     : wall-clock seconds spent in the CG loop
    solving_time        : wall-clock seconds spent on the integer re-solve
    """
    lp_obj:           float
    integer_obj:      float
    lp_solution:      Optional[RMPSolution]
    integer_solution: Optional[Dict]
    generation_time:  float
    solving_time:     float
    mip_gap:          float   = np.nan
    obj_bound:        float   = np.nan
    iteration_log:    list    = field(default_factory=list)  # list of {iteration, obj_val, coalition}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_route_record(
    route_id:            int,
    stops:               list,
    duration:            float,
    data,
    origin:              str,
    tree_level:          Optional[int]   = None,
    reduced_cost:        Optional[float] = None,
    generation_quality:  Optional[str]   = None,
) -> RouteRecord:
    """
    Build a RouteRecord from a finished route and its metadata.

    Counts station and depot visits by checking each stop's location_type,
    excluding the parking node that opens and closes every route.
    """
    station_indices = set(data.station_indices)
    depot_indices   = set(data.depot_indices)

    # Exclude the first and last stop (both parking)
    interior = stops[1:-1]
    n_stations = sum(1 for s in interior if s in station_indices)
    n_depots   = sum(1 for s in interior if s in depot_indices)

    return RouteRecord(
        route_id            = route_id,
        origin              = origin,
        stops               = list(stops),
        duration            = duration,
        n_stations          = n_stations,
        n_depots            = n_depots,
        tree_level          = tree_level,
        reduced_cost        = reduced_cost,
        generation_quality  = generation_quality,
    )


def _add_column(
    instance:    IRPInstance,
    result:      SubproblemResult,
    tree_level:  int,
) -> None:
    """
    Register a new column (route) into the instance's RoutePool in-place.

    Updates: route_edges, route_durations, new_routes, edge_routes,
             feasible_routes, reduced_costs, route_records.

    Parameters
    ----------
    instance   : IRPInstance being solved
    result     : SubproblemResult from the pricing subproblem
    tree_level : depth in the B&P tree where this route was discovered
                 (0 = root CG, 1 = first branch level, etc.)
    """
    routes = instance.routes
    data   = instance.data
    V, T   = data.V, len(next(iter(instance.demand.values())))
    R      = routes.R                        # id of the new route

    routes.route_edges[R]   = result.route_edges
    routes.route_durations.append(result.route_duration)
    routes.new_routes[R]    = result.route_stops
    routes.reduced_costs[R] = result.reduced_cost

    # Update edge -> route index
    stops = result.route_stops
    for i in range(len(stops) - 1):
        edge = (stops[i], stops[i + 1])
        if edge in routes.edge_routes:
            routes.edge_routes[edge].append(R)
        else:
            routes.edge_routes[edge] = [R]

    # Make route available for every (truck, day)
    for v in range(V):
        for t in range(T):
            routes.feasible_routes[(v, t)].append(R)

    # Store route metadata
    routes.route_records[R] = _make_route_record(
        route_id            = R,
        stops               = result.route_stops,
        duration            = result.route_duration,
        data                = data,
        origin              = "cg",
        tree_level          = tree_level,
        reduced_cost        = result.reduced_cost,
        generation_quality  = result.generation_quality,
    )

    routes.R += 1


def _extract_integer_solution(
    model:     gp.Model,
    vars_W:    gp.tupledict,
    vars_F:    gp.tupledict,
    vars_I:    gp.tupledict,
    vars_B:    gp.tupledict,
    vars_Im:   gp.tupledict,
    instance:  IRPInstance,
    penalties: PenaltyConfig,
) -> Dict:
    """Extract solution dict and objective breakdown from a solved integer model."""
    data   = instance.data
    routes = instance.routes
    V, N   = data.V, data.N
    T      = len(next(iter(instance.demand.values())))
    R      = routes.R

    route_edges     = routes.route_edges
    route_durations = routes.route_durations

    # ---- Routes ----
    solution: Dict = {
        "Routes":          {},
        "SelectedRouteIDs": {},   # (v,t) -> route_id of the selected route
        "Loadings":        {},
        "Unloadings":      {},
        "Levels":          np.zeros((N, T)),
    }

    for (v, t, r), var in vars_W.items():
        if var.X > 0.5:
            solution["Routes"][(v, t)]          = copy.deepcopy(route_edges[r])
            solution["SelectedRouteIDs"][(v, t)] = r

    # ---- Flow -> loadings / unloadings ----
    flow = np.zeros((V, T, N, N))
    for (v, t, i, j), var in vars_F.items():
        if var.X > 0.01:
            flow[v, t, i, j] = var.X

    for v in range(V):
        for t in range(T):
            solution["Loadings"][(v, t)]   = np.zeros(N)
            solution["Unloadings"][(v, t)] = np.zeros(N)
            for i in range(N):
                net_in  = np.sum(flow[v, t, :, i]) - np.sum(flow[v, t, i, :])
                net_out = -net_in
                solution["Unloadings"][(v, t)][i] = max(0.0, net_in)
                solution["Loadings"][(v, t)][i]   = max(0.0, net_out)

    # ---- Inventory levels ----
    for (t, n), var in vars_I.items():
        solution["Levels"][n, t] = var.X

    # ---- Objective breakdown ----
    transport_cost = sum(
        var.X * route_durations[r]
        for (v, t, r), var in vars_W.items()
        if var.X > 0.1
    )
    below_safety = penalties.phi1 * sum(
        vars_B[t, i].X
        for t in range(T) for i in range(N)
        if vars_B[t, i].X > 1e-6
    )
    below_zero = penalties.phi2 * sum(
        vars_Im[t, i].X
        for t in range(T) for i in range(N)
        if vars_Im[t, i].X > 1e-6
    )
    solution["Objective"] = {
        "transport_cost":       transport_cost,
        "below_safety_penalty": below_safety,
        "below_zero_penalty":   below_zero,
        "total":                transport_cost + below_safety + below_zero,
    }
    solution["Progression over time"] = {}  # populated by callback in run_column_generation

    return solution


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_column_generation(
    instance:         IRPInstance,
    penalties:        PenaltyConfig,
    problem_cfg:      ProblemConfig,
    solver_cfg:       SolverConfig,
    env:              gp.Env,
    lower_bound:      float = -np.inf,
    integer_solution: bool  = False,
    only_integer:     bool  = False,
    tree_level:       int   = 0,
    deadline:         float = float("inf"),  # absolute wall-clock time (perf_counter)
) -> CGResult:
    """
    Run column generation to solve the LP relaxation, then optionally
    solve the integer program over the generated column pool.

    Parameters
    ----------
    instance         : IRPInstance (mutated in-place as routes are added)
    penalties        : phi1, phi2 penalty weights
    problem_cfg      : daily_driving_limit etc.
    solver_cfg       : time limits, CG tolerance
    env              : silent Gurobi environment
    lower_bound      : known lower bound — stop early if LP hits it
    integer_solution : if True, solve the integer RMP after CG completes
    only_integer     : if True, skip CG and go straight to integer solve
    tree_level       : B&P tree depth at which this CG is called.
                       0 = root (standard CG), 1+ = branching nodes.
                       Stored in each RouteRecord generated here.

    Returns
    -------
    CGResult
    """
    data = instance.data
    V    = data.V
    T    = len(next(iter(instance.demand.values())))

    # -----------------------------------------------------------------------
    # Phase 1: LP column-generation loop
    # -----------------------------------------------------------------------
    non_improvement     = 0
    iteration           = 0

    # -----------------------------------------------------------------------
    # Subproblem selection policy
    # "feasibility" : price the (v,t) pair with fewest feasible routes first
    # "duals"       : price the (v,t) pair with most promising dual score first
    # -----------------------------------------------------------------------
    SELECTION_POLICY = "duals"

    def _feasibility_order() -> list:
        return sorted(
            [(v, t) for v in range(V) for t in range(T)],
            key=lambda vt: len(instance.routes.feasible_routes.get(vt, [])),
        )

    # Initial order (before first RMP solve duals are unavailable)
    vt_order = _feasibility_order()
    vt_index = 0   # current position in vt_order

    # Build edge set and travel costs once — they depend only on network topology
    E_all  = instance.data.feasible_edges   # precomputed in IRPData
    c_edge = {
        (i, j): instance.data.Durations[
            instance.data.Nodes[i].location.name][
            instance.data.Nodes[j].location.name] + 0.15
        for (i, j) in E_all
    }

    def _dual_score(sigma: np.ndarray, iota: np.ndarray) -> list:
        """
        Return (v,t) pairs sorted by most promising duals first.

        For each pair the score is a lower bound on its best achievable
        reduced cost:
            score(v,t) = sum_{(i,j) in E} min(0, c[i,j] - cap_v * sigma[v,t,i,j])
                         - iota[v,t]
        The most negative score indicates the pair most likely to yield
        a route with a negative reduced cost.
        """
        scores = {}
        for v in range(V):
            cap_v = instance.data.Trucks[v].total_capacity
            for t in range(T):
                dual_lb = sum(
                    min(0.0, c_edge[i, j] - cap_v * sigma[v, t, i, j])
                    for (i, j) in E_all
                ) - iota[v, t]
                scores[(v, t)] = dual_lb
        return sorted(scores.keys(), key=lambda vt: scores[vt])
    lp_rmp_sol: Optional[RMPSolution] = None
    last_route_added_id = -1
    last_rmp_R = -1   # pool size at the last RMP solve; -1 forces solve on first iteration
    iteration_log: list = []   # one entry per RMP solve: {iteration, obj_val, coalition}

    cg_start = time.perf_counter()

    while not only_integer:
        # ---- Hard deadline check — terminate immediately if expired ----
        if time.perf_counter() >= deadline:
            print(f"[CG] Generation deadline reached at iteration {iteration}. Stopping.")
            break

        print(f"[CG] Iteration {iteration}  |  routes in pool: {instance.routes.R}")

        # ---- Solve RMP only if the pool has grown since the last solve ----
        if instance.routes.R > last_rmp_R:
            model, rmp_vars = build_rmp_model(
                instance=instance,
                penalties=penalties,
                env=env,
                continuous=True,
            )
            lp_rmp_sol = solve_rmp(model, rmp_vars, instance)
            last_rmp_R = instance.routes.R
            # Update per-route RMP usage statistics
            instance.routes.update_rmp_stats(
                lp_rmp_sol.W_sol, instance.data.V,
                len(next(iter(instance.demand.values())))
            )

            # ---- Check lower-bound convergence ----
            if abs(lp_rmp_sol.obj_val - lower_bound) < solver_cfg.cg_optimality_tol:
                print(f"[CG] LP objective reached lower bound ({lower_bound:.6f}). Stopping.")
                break

            # ---- Warn if last added route was never used (degenerate) ----
            if last_route_added_id >= 0:
                R_prev = last_route_added_id
                if np.sum(lp_rmp_sol.routes_selected[:, :, R_prev]) == 0:
                    print(f"[CG] Route {R_prev} (just added) not used in current RMP solution.")
                    if R_prev in instance.routes.route_records:
                        instance.routes.route_records[R_prev].degenerate = True

            # ---- Reorder (v,t) pairs according to selection policy ----
            if SELECTION_POLICY == "duals":
                vt_order = _dual_score(lp_rmp_sol.sigma, lp_rmp_sol.iota)
            else:
                vt_order = _feasibility_order()
            vt_index = 0

        # ---- Solve pricing subproblem ----
        current_truck, current_day = vt_order[vt_index]
        sub_results = solve_subproblem(
            instance=instance,
            sigma=lp_rmp_sol.sigma,
            iota=lp_rmp_sol.iota,
            truck=current_truck,
            day=current_day,
            problem_cfg=problem_cfg,
            solver_cfg=solver_cfg,
            env=env,
        )

        # ---- Record coalition for this iteration (every iteration, not just when pool grows) ----
        if lp_rmp_sol is not None:
            iteration_log.append({
                "iteration": iteration,
                "obj_val":   lp_rmp_sol.obj_val,
                "coalition": {
                    (v, t, r): w
                    for (v, t, r), w in lp_rmp_sol.W_sol.items()
                    if w > 1e-6
                },
            })

        # sub_results is now a list of SubproblemResult objects.
        # Check infeasibility on the first (and typically only Gurobi) result.
        if sub_results[0].infeasible:
            print("[CG] Subproblem infeasible. This node is pruned.")
            generation_time = time.perf_counter() - cg_start
            return CGResult(
                lp_obj=np.inf,
                integer_obj=np.inf,
                lp_solution=None,
                integer_solution=None,
                generation_time=generation_time,
                solving_time=0.0,
            )

        # ---- Add all useful, non-duplicate columns from the result list ----
        n_added = 0
        for sub_result in sub_results:
            if (sub_result.is_useful
                    and sub_result.route_stops not in instance.routes.new_routes.values()):
                last_route_added_id = instance.routes.R
                _add_column(instance, sub_result, tree_level=tree_level)
                _rec = instance.routes.route_records[last_route_added_id]
                print(f"[CG] New column added.  RC={sub_result.reduced_cost:.4f}  "
                      f"dur={_rec.duration:.2f}h  depots={_rec.n_depots}  "
                      f"stations={_rec.n_stations}  stops={sub_result.route_stops}")
                n_added += 1

        if n_added > 0:
            non_improvement = 0
        else:
            non_improvement += 1
            # ---- Termination: all (truck, day) pairs exhausted without improvement ----
            if non_improvement >= V * T:
                print("[CG] No improving column found for any (truck, day). LP optimal.")
                break
            # ---- Advance to next (truck, day) in feasibility-sorted order ----
            vt_index = (vt_index + 1) % len(vt_order)

        iteration += 1

    generation_time = time.perf_counter() - cg_start

    # LP-only mode: return without integer solve
    if not integer_solution and not only_integer:
        return CGResult(
            lp_obj=lp_rmp_sol.obj_val if lp_rmp_sol else np.inf,
            integer_obj=np.inf,
            lp_solution=lp_rmp_sol,
            integer_solution=None,
            generation_time=generation_time,
            solving_time=0.0,
            iteration_log=iteration_log,
        )

    # -----------------------------------------------------------------------
    # Phase 2: Integer re-solve over the generated column pool
    # -----------------------------------------------------------------------
    n_active = len(set().union(*instance.routes.feasible_routes.values()))
    print(f"[CG] Starting integer re-solve over {n_active} routes "
        f"(pool size {instance.routes.R}).")
    int_start = time.perf_counter()

    int_model, int_vars = build_rmp_model(
        instance=instance,
        penalties=penalties,
        env=env,
        continuous=False,        # W is now BINARY
    )
    int_model.setParam("TimeLimit", 60 * solver_cfg.time_limit)

    # ---- Callback: record objective at checkpoints ----
    record_times = solver_cfg.record_times_bp
    recorded_data: Dict = {}
    _next_idx = {"i": 0}

    def _record_callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        i = _next_idx["i"]
        if i >= len(record_times):
            return
        if runtime >= record_times[i]:
            checkpoint = record_times[i]
            obj_best   = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            obj_bound  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            gap = (
                abs(obj_best - obj_bound) / abs(obj_best)
                if abs(obj_best) > 1e-10 else np.inf
            )
            recorded_data[checkpoint] = {
                "obj":      model.cbGet(GRB.Callback.MIPSOL_OBJ),
                "bound":    obj_bound,
                "gap":      gap,
                "solution": {v.VarName: model.cbGetSolution(v) for v in model.getVars()},
            }
            _next_idx["i"] += 1

    int_model.optimize(_record_callback)
    solving_time = time.perf_counter() - int_start

    print(f"[CG] Integer objective = {int_model.ObjVal:.6f}")

    int_solution = _extract_integer_solution(
        model=int_model,
        vars_W=int_vars.W,
        vars_F=int_vars.F,
        vars_I=int_vars.I,
        vars_B=int_vars.B,
        vars_Im=int_vars.Im,
        instance=instance,
        penalties=penalties,
    )
    # Merge the richer recorded_data (from callback) into the solution
    int_solution["Progression over time"] = recorded_data

    lp_obj  = lp_rmp_sol.obj_val if (lp_rmp_sol and not only_integer) else np.inf
    mip_gap   = int_model.MIPGap    if int_model.SolCount > 0 else np.nan
    obj_bound = int_model.ObjBound  # always available after optimize(), even without incumbent

    return CGResult(
        lp_obj=lp_obj,
        integer_obj=int_model.ObjVal,
        lp_solution=lp_rmp_sol,
        integer_solution=int_solution,
        generation_time=generation_time,
        solving_time=solving_time,
        mip_gap=mip_gap,
        obj_bound=obj_bound,
        iteration_log=iteration_log,
    )