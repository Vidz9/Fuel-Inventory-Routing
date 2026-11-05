"""
heuristics/demand_route_generator.py
=====================================
Demand-aware warm-start route generator.

Instead of a hand-crafted construction heuristic, this module:
  1. Computes synthetic dual variables (sigma) based on instance demand
     and station capacity — arcs leading to high-demand stations get high
     sigma, making them attractive to the stochastic subproblem sampler.
  2. Samples multiple random temperatures and trucks to generate diverse routes
     via the existing _stochastic_subproblem routine.
  3. Accepts all generated routes regardless of reduced cost (warm-start
     routes are not subject to RC filtering).

Public API
----------
    generate_demand_routes(instance, problem_cfg, max_routes) -> int
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Set, Tuple

from config import ProblemConfig
from instance import IRPInstance, RouteRecord
from solvers.subproblem import _stochastic_subproblem


# ---------------------------------------------------------------------------
# Synthetic dual construction
# ---------------------------------------------------------------------------

def _make_demand_sigma(
    instance: IRPInstance,
    truck_idx: int,
) -> np.ndarray:
    """
    Build a synthetic sigma array shaped (V, T, N, N) where only the
    slice [truck_idx, :, :, :] is non-zero.

    For each arc (i -> j) where j is a station node:

        sigma[v, t, i, j] = urgency[j] / cap_v

    where urgency[j] = max(0, safety[j] - (initial_level[j] - total_demand[j]))
                       / node_capacity[j]

    This makes w[i,j] = c[i,j] - cap_v * sigma[v,t,i,j]
                      = c[i,j] - urgency[j]

    so arcs leading to urgent stations have lower (more negative) w and
    are therefore sampled with higher probability by the stochastic heuristic.

    A small amount of noise is added so that repeated calls with the same
    truck produce different route samples.
    """
    data  = instance.data
    V, N  = data.V, data.N
    T     = len(next(iter(instance.demand.values())))
    cap_v = float(sum(data.Trucks[truck_idx].compartment_capacities))

    sigma = np.zeros((V, T, N, N))

    for node, init_level in instance.initial_levels.items():
        s = node.number
        total_demand = sum(instance.demand[node])
        node_cap     = max(node.capacity, 1e-6)
        urgency      = max(0.0, node.safety - (init_level - total_demand)) / node_cap

        if urgency < 1e-8:
            continue

        # Set sigma for ALL arcs (i -> s) for this truck across all days
        for i in range(N):
            if i == s:
                continue
            for t in range(T):
                # Base sigma + per-day noise for diversity
                noise = np.random.uniform(0.0, 0.3 * urgency)
                sigma[truck_idx, t, i, s] = (urgency + noise) / max(cap_v, 1e-6)

    return sigma


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_demand_routes(
    instance:    IRPInstance,
    problem_cfg: ProblemConfig,
    max_routes:  int   = 100,
    temperatures: List[float] = None,
) -> int:
    """
    Generate demand-aware warm-start routes using the stochastic subproblem
    sampler with synthetic demand-based duals.

    For each sample:
      - Pick a random truck and day
      - Build synthetic sigma duals reflecting station urgency
      - Run _stochastic_subproblem with a random temperature
      - Accept all unique feasible routes (no RC filter for warm-start)

    Parameters
    ----------
    max_routes   : target pool size (including dummy route 0)
    temperatures : list of temperatures to cycle through
    """
    if temperatures is None:
        temperatures = [0.2, 0.5, 1.0, 2.0]

    data   = instance.data
    routes = instance.routes

    Nodes           = data.Nodes
    Durations       = data.Durations
    depot_indices   = data.depot_indices
    station_indices = data.station_indices
    station_set:    Set[int] = set(station_indices)
    depot_set:      Set[int] = set(depot_indices)
    parking         = data.parking_index
    daily_limit     = problem_cfg.daily_driving_limit
    V               = data.V
    T               = len(next(iter(instance.demand.values())))
    N               = data.N

    # Build edge set and adjacency (no forbidden/mandatory during warm-start)
    E: List[Tuple[int, int]] = list(data.feasible_edges)
    out_edges: Dict[int, List[int]] = {n: [] for n in range(N)}
    in_edges:  Dict[int, List[int]] = {n: [] for n in range(N)}
    for (i, j) in E:
        out_edges[i].append(j)
        in_edges[j].append(i)

    # Base travel costs c[i,j] — same for all trucks
    c: Dict[Tuple[int, int], float] = {
        (i, j): Durations[Nodes[i].location.name][Nodes[j].location.name] + 0.15
        for (i, j) in E
    }

    # ------------------------------------------------------------------
    # Register a completed route
    # ------------------------------------------------------------------
    def _add_route(stops: List[int], edges: np.ndarray, dur: float) -> bool:
        R = routes.R
        if dur > daily_limit:
            return False
        if stops in routes.new_routes.values():
            return False

        routes.new_routes[R]    = list(stops)
        routes.route_durations.append(dur)
        routes.route_edges[R]   = edges.copy()
        routes.reduced_costs[R] = 0.0

        for k in range(len(stops) - 1):
            edge = (stops[k], stops[k+1])
            routes.edge_routes.setdefault(edge, []).append(R)

        for v in range(V):
            for t in range(T):
                routes.feasible_routes[(v, t)].append(R)

        interior   = stops[1:-1]
        n_stations = sum(1 for s in interior if s in station_set)
        n_depots   = sum(1 for s in interior if s in depot_set)
        routes.route_records[R] = RouteRecord(
            route_id           = R,
            origin             = "warmstart",
            stops              = list(stops),
            duration           = dur,
            n_stations         = n_stations,
            n_depots           = n_depots,
            tree_level         = None,
            reduced_cost       = None,
            generation_quality = "heuristic",
        )
        routes.R += 1
        return True

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    attempts     = 0
    max_attempts = max_routes * 30
    temp_idx     = 0

    while routes.R < max_routes and attempts < max_attempts:
        attempts += 1

        # Sample a random truck and day
        truck_idx = int(np.random.randint(V))
        day_idx   = int(np.random.randint(T))
        cap_v     = float(sum(data.Trucks[truck_idx].compartment_capacities))

        # Build synthetic demand-based sigma for this truck
        sigma = _make_demand_sigma(instance, truck_idx)

        # Compute modified arc costs w[i,j] = c[i,j] - cap_v * sigma[truck,day,i,j]
        w: Dict[Tuple[int, int], float] = {
            (i, j): c[i, j] - cap_v * sigma[truck_idx, day_idx, i, j]
            for (i, j) in E
        }

        # Sample temperature cyclically for diversity
        temp = temperatures[temp_idx % len(temperatures)]
        temp_idx += 1

        edges, stops, _ = _stochastic_subproblem(
            N             = N,
            parking       = parking,
            out_edges     = out_edges,
            c             = c,
            w             = w,
            daily_limit   = daily_limit,
            mandatory_set = set(),   # no branching constraints during warm-start
            forbidden_set = set(),
            temperature   = temp,
        )

        if stops is None or len(stops) <= 2:
            continue

        dur = sum(c[stops[k], stops[k+1]] for k in range(len(stops) - 1))
        _add_route(stops, edges, dur)

    return routes.R