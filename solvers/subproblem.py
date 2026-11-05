"""
solvers/subproblem.py
=====================
Pricing subproblem for the column-generation procedure.

Given the dual variables (sigma, iota) from the current RMP solution,
we solve a shortest-path / TSP-style MIP for each (truck, day) pair to
find a route with a negative reduced cost.  If one is found, it becomes
a new column in the RMP.

Strategy
--------
1. A greedy heuristic is run first.  At each node it greedily extends the
   route by the outgoing edge with the most negative reduced-cost weight,
   subject to the time limit.  If the resulting route already has a negative
   reduced cost, Gurobi is skipped entirely.
2. If the greedy route is not good enough, Gurobi is called with the greedy
   route as a warm-start incumbent.

Public API
----------
    solve_subproblem(...)  ->  SubproblemResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import ProblemConfig, SolverConfig
from data.loader import IRPData
from instance import IRPInstance
from solvers.milp import build_edge_set


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SubproblemResult:
    """
    Output of one subproblem solve for a single (truck, day) pair.

    Attributes
    ----------
    reduced_cost    : objective value of the subproblem minus iota dual.
                      Negative means a useful new column was found.
    route_duration  : total driving time of the new route (hours)
    route_edges     : (N×N) binary array — which arcs are in the route
    route_stops     : ordered list of node indices, starting and ending
                      at the parking node index
    truck           : index of the truck this route was solved for
    day             : planning day this route was solved for
    infeasible      : True if the subproblem was infeasible (signals that
                      the B&P node should be pruned)
    """
    reduced_cost:   float
    route_duration: float
    route_edges:    Optional[np.ndarray]
    route_stops:    Optional[List[int]]
    truck:          int
    day:            int
    infeasible:          bool  = False
    generation_quality:  str   = "optimal"  # "heuristic" | "suboptimal" | "optimal"

    @property
    def has_negative_reduced_cost(self) -> bool:
        return self.reduced_cost <= -1e-4

    @property
    def is_useful(self) -> bool:
        """True when the route has a negative reduced cost and is not degenerate."""
        return self.has_negative_reduced_cost and self.route_stops is not None


# ---------------------------------------------------------------------------
# Greedy heuristic
# ---------------------------------------------------------------------------

def _greedy_subproblem(
    N:             int,
    parking:       int,
    out_edges:     Dict[int, List[int]],
    c:             Dict[Tuple[int, int], float],
    w:             Dict[Tuple[int, int], float],
    daily_limit:   float,
    mandatory_set: Set[Tuple[int, int]],
    forbidden_set: Set[Tuple[int, int]],
) -> Tuple[Optional[np.ndarray], Optional[List[int]], float]:
    """
    Build a route greedily and return (route_edges, route_stops, obj_val).

    At each node the outgoing edge with the most negative w[i,j] is chosen,
    provided:
      - the destination has not been visited yet, and
      - after taking that edge, the truck can still return to parking within
        the daily driving limit.

    The route extends as long as there exists at least one beneficial
    (w < 0) reachable edge.  If mandatory edges are present and cannot
    all be satisfied by the greedy path, None is returned to signal
    fallback to Gurobi.

    Returns
    -------
    (route_edges, route_stops, obj_val)  or  (None, None, 0.0) on failure
    """
    visited       = {parking}
    route_stops   = [parking]
    route_edges   = np.zeros((N, N))
    duration_used = 0.0
    obj_val       = 0.0
    current       = parking

    while True:
        best_j = None
        best_w = np.inf

        # If there is a mandatory edge leaving the current node, we must take it
        mandatory_j = next(
            (j for (i, j) in mandatory_set if i == current and j not in visited),
            None,
        )

        if mandatory_j is not None:
            c_ij        = c.get((current, mandatory_j), np.inf)
            c_j_parking = c.get((mandatory_j, parking), 1)
            if duration_used + c_ij + c_j_parking <= daily_limit:
                best_j = mandatory_j
            else:
                break   # mandatory edge unreachable within time — fall back to Gurobi
        else:
            for j in out_edges[current]:
                if j in visited:
                    continue
                # Skip forbidden edges from current node
                if (current, j) in forbidden_set:
                    continue
                # Can we reach j and still return to parking?
                c_ij        = c[current, j]
                c_j_parking = c.get((j, parking), 1)
                if duration_used + c_ij + c_j_parking > daily_limit:
                    continue
                if w[current, j] < best_w:
                    best_w = w[current, j]
                    best_j = j

        if best_j is None:
            break   # no reachable edge — stop extending

        # Move to best_j
        route_edges[current, best_j] = 1
        obj_val       += w[current, best_j]
        duration_used += c[current, best_j]
        visited.add(best_j)
        route_stops.append(best_j)
        current = best_j

    print(route_stops)

    # Close the route back to parking
    if current == parking:
        # Route never left parking — greedy found nothing useful
        return None, None, 0.0

    if (current, parking) not in c:
        return None, None, 0.0

    route_edges[current, parking] = 1
    obj_val       += w.get((current, parking), c[current, parking])
    duration_used += c[current, parking]
    route_stops.append(parking)

    # Check mandatory edges are all satisfied
    for edge in mandatory_set:
        if route_edges[edge[0], edge[1]] < 0.5:
            return None, None, 0.0   # mandatory edge missing — fall back to Gurobi
        
    for edge in forbidden_set:
        if route_edges[edge[0], edge[1]] > 0.5:
            return None, None, 0.0   # forbidden edge included — fall back to Gurobi

    return route_edges, route_stops, obj_val


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _stochastic_subproblem(
    N:             int,
    parking:       int,
    out_edges:     Dict[int, List[int]],
    c:             Dict[Tuple[int, int], float],
    w:             Dict[Tuple[int, int], float],
    daily_limit:   float,
    mandatory_set: Set[Tuple[int, int]],
    forbidden_set: Set[Tuple[int, int]],
    temperature:   float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[List[int]], float]:
    """
    Stochastic route construction for the pricing subproblem.

    At each node, instead of always taking the best edge (greedy), the next
    node is sampled with probability proportional to exp(-w[i,j] / temperature).
    More negative w[i,j] (more attractive arcs) get exponentially higher
    probability, but worse arcs still have a chance — this generates diverse
    routes that the greedy misses.

    Parameters
    ----------
    temperature : controls randomness. Lower -> more greedy. Higher -> more random.

    Returns
    -------
    (route_edges, route_stops, obj_val)  or  (None, None, 0.0) if route infeasible
    """
    visited       = {parking}
    route_stops   = [parking]
    route_edges   = np.zeros((N, N))
    duration_used = 0.0
    obj_val       = 0.0
    current       = parking

    while True:
        # Mandatory edge takes priority
        mandatory_j = next(
            (j for (i, j) in mandatory_set if i == current and j not in visited),
            None,
        )

        if mandatory_j is not None:
            c_ij        = c.get((current, mandatory_j), np.inf)
            c_j_parking = c.get((mandatory_j, parking), 1)
            if duration_used + c_ij + c_j_parking <= daily_limit:
                best_j = mandatory_j
            else:
                return None, None, 0.0   # mandatory arc infeasible
        else:
            # Collect feasible candidates
            candidates = []
            weights_w  = []
            for j in out_edges[current]:
                if j in visited:
                    continue
                if (current, j) in forbidden_set:
                    continue
                c_ij        = c[current, j]
                c_j_parking = c.get((j, parking), 1)
                if duration_used + c_ij + c_j_parking > daily_limit:
                    continue
                candidates.append(j)
                weights_w.append(w[current, j])

            if not candidates:
                break   # no feasible arc — close route

            # Sample proportional to exp(-w / temperature)
            # Negative w gets high score; we subtract min for numerical stability
            scores = np.array(weights_w)
            scores = -(scores - scores.min()) / max(temperature, 1e-6)
            probs  = np.exp(scores)
            probs /= probs.sum()
            best_j = candidates[int(np.random.choice(len(candidates), p=probs))]

        route_edges[current, best_j] = 1
        obj_val       += w[current, best_j]
        duration_used += c[current, best_j]
        visited.add(best_j)
        route_stops.append(best_j)
        current = best_j

    if current == parking:
        return None, None, 0.0
    if (current, parking) not in c:
        return None, None, 0.0

    route_edges[current, parking] = 1
    obj_val       += w[current, parking]
    duration_used += c[current, parking]
    route_stops.append(parking)

    for edge in mandatory_set:
        if route_edges[edge[0], edge[1]] < 0.5:
            return None, None, 0.0
    for edge in forbidden_set:
        if route_edges[edge[0], edge[1]] > 0.5:
            return None, None, 0.0

    return route_edges, route_stops, obj_val


def solve_subproblem(
    instance:      IRPInstance,
    sigma:         np.ndarray,
    iota:          np.ndarray,
    truck:         int,
    day:           int,
    problem_cfg:   ProblemConfig,
    solver_cfg:    SolverConfig,
    env:           gp.Env,
) -> List[SubproblemResult]:
    """
    Solve the pricing subproblem for a given truck and day.

    The objective is to find the feasible route with the most negative
    reduced cost:

        min  sum_{(i,j) in E} w[i,j] * X[i,j]  -  iota[truck, day]

    where the modified arc cost is:

        w[i,j] = c[i,j]  -  capacity_v * sigma[v, t, i, j]

    and  c[i,j] = duration[i,j] + 0.15  (driving time + stop time).

    A greedy heuristic is tried first.  If it finds a route with a
    negative reduced cost, Gurobi is skipped.  Otherwise Gurobi is called
    with the greedy route as a warm-start incumbent.

    Parameters
    ----------
    instance     : IRPInstance (for Nodes, Durations, truck capacities,
                   mandatory/forbidden edges)
    sigma        : dual of Sigma constraints  shape (V, T, N, N)
    iota         : dual of Iota  constraints  shape (V, T)
    truck        : vehicle index
    day          : planning day index
    problem_cfg  : ProblemConfig (daily_driving_limit)
    solver_cfg   : SolverConfig  (subproblem time limits)
    env          : Gurobi environment

    Returns
    -------
    SubproblemResult
    """
    data   = instance.data
    routes = instance.routes

    Nodes       = data.Nodes
    Durations   = data.Durations
    Trucks      = data.Trucks
    N           = data.N
    parking     = data.parking_index

    daily_limit = problem_cfg.daily_driving_limit
    cap_v       = Trucks[truck].total_capacity

    mandatory_set: Set[Tuple[int, int]] = set(routes.mandatory_edges[(truck, day)])
    forbidden_set: Set[Tuple[int, int]] = set(routes.forbidden_edges[(truck, day)])

    # ------------------------------------------------------------------
    # Build feasible edge set
    # ------------------------------------------------------------------
    E: List[Tuple[int, int]] = [
        (i, j)
        for (i, j) in build_edge_set(Nodes)
        if (i, j) not in forbidden_set
    ]

    out_edges: Dict[int, List[int]] = {n: [] for n in range(N)}
    in_edges:  Dict[int, List[int]] = {n: [] for n in range(N)}
    for (i, j) in E:
        out_edges[i].append(j)
        in_edges[j].append(i)

    # ------------------------------------------------------------------
    # Pre-compute arc costs
    # ------------------------------------------------------------------
    c: Dict[Tuple[int, int], float] = {}
    w: Dict[Tuple[int, int], float] = {}
    for (i, j) in E:
        dur     = Durations[Nodes[i].location.name][Nodes[j].location.name]
        c[i, j] = dur + 0.15
        w[i, j] = c[i, j] - cap_v * sigma[truck, day, i, j]

    # ------------------------------------------------------------------
    # Phase 1: greedy + 10 stochastic heuristics
    # All routes with negative RC and not duplicates are collected.
    # If any found, return the full list — Gurobi is skipped.
    # The best route (lowest RC) warm-starts Gurobi if we fall through.
    # ------------------------------------------------------------------
    N_STOCHASTIC = 10
    TEMPERATURES = [0.3, 0.5, 1.0, 1.5, 3.0]

    heuristic_results: List[SubproblemResult] = []
    best_heuristic_edges = None
    best_heuristic_stops = None
    best_heuristic_rc    = np.inf

    # Deterministic greedy
    greedy_edges, greedy_stops, greedy_obj = _greedy_subproblem(
        N=N, parking=parking, out_edges=out_edges,
        c=c, w=w, daily_limit=daily_limit,
        mandatory_set=mandatory_set,
        forbidden_set=forbidden_set,
    )
    if greedy_stops is not None:
        greedy_rc = greedy_obj - iota[truck, day]
        if greedy_rc < best_heuristic_rc:
            best_heuristic_rc    = greedy_rc
            best_heuristic_edges = greedy_edges
            best_heuristic_stops = greedy_stops
        if (greedy_rc <= -1e-4
                and greedy_stops not in instance.routes.new_routes.values()):
            duration = sum(c[greedy_stops[k], greedy_stops[k+1]]
                           for k in range(len(greedy_stops) - 1))
            heuristic_results.append(SubproblemResult(
                reduced_cost=greedy_rc, route_duration=duration,
                route_edges=greedy_edges, route_stops=greedy_stops,
                truck=truck, day=day,
                generation_quality="heuristic",
            ))

    # Stochastic samples
    seen_stops = {tuple(greedy_stops) if greedy_stops else ()}
    for _attempt in range(N_STOCHASTIC):
        temp = TEMPERATURES[_attempt % len(TEMPERATURES)]
        s_edges, s_stops, s_obj = _stochastic_subproblem(
            N=N, parking=parking, out_edges=out_edges,
            c=c, w=w, daily_limit=daily_limit,
            mandatory_set=mandatory_set,
            forbidden_set=forbidden_set,
            temperature=temp,
        )
        if s_stops is None:
            continue
        if tuple(s_stops) in seen_stops:
            continue          # skip duplicates within this batch
        seen_stops.add(tuple(s_stops))

        s_rc = s_obj - iota[truck, day]
        if s_rc < best_heuristic_rc:
            best_heuristic_rc    = s_rc
            best_heuristic_edges = s_edges
            best_heuristic_stops = s_stops

        if (s_rc <= -1e-4
                and s_stops not in instance.routes.new_routes.values()):
            duration = sum(c[s_stops[k], s_stops[k+1]]
                           for k in range(len(s_stops) - 1))
            heuristic_results.append(SubproblemResult(
                reduced_cost=s_rc, route_duration=duration,
                route_edges=s_edges, route_stops=s_stops,
                truck=truck, day=day,
                generation_quality="heuristic",
            ))

    if heuristic_results:
        print(f"[Subproblem] Heuristic found {len(heuristic_results)} useful route(s). "
              f"Best RC={min(r.reduced_cost for r in heuristic_results):.4f}")
        return heuristic_results

    # ------------------------------------------------------------------
    # Phase 2: Gurobi MIP (warm-started with greedy incumbent if available)
    # ------------------------------------------------------------------
    sub = gp.Model("Subproblem", env=env)

    X = sub.addVars(E, vtype=GRB.BINARY, lb=0, name="X")
    S = sub.addVars(N,  vtype=GRB.CONTINUOUS, lb=0, name="S")

    # Warm-start: inject best heuristic solution as initial incumbent
    if best_heuristic_stops is not None:
        for (i, j) in E:
            X[i, j].Start = float(best_heuristic_edges[i, j])

    # ---- Mandatory branching arcs ----
    for (i, j) in mandatory_set:
        if (i, j) in X:
            sub.addConstr(X[i, j] == 1)

    # ---- Structural routing constraints ----
    sub.addConstr(
        gp.quicksum(c[i, j] * X[i, j] for (i, j) in E) <= daily_limit
    )
    for n in range(N):
        out_sum = gp.quicksum(X[n, j] for j in out_edges[n])
        in_sum  = gp.quicksum(X[i, n] for i in in_edges[n])
        sub.addConstr(out_sum == in_sum)
    for n in range(N):
        sub.addConstr(gp.quicksum(X[n, j] for j in out_edges[n]) <= 1)
    sub.addConstr(gp.quicksum(X[parking, j] for j in out_edges[parking]) == 1)
    sub.addConstr(S[parking] == 0)
    for (i, j) in E:
        if j != parking:
            sub.addConstr(S[j] >= S[i] - 1000 * (1 - X[i, j]) + 1)

    # ---- Objective ----
    sub.setObjective(
        gp.quicksum(w[i, j] * X[i, j] for (i, j) in E),
        GRB.MINIMIZE,
    )
    sub.setParam("TimeLimit", solver_cfg.subproblem_time_limit)

    # ---- Early-stop callback ----
    early_stop_time      = solver_cfg.subproblem_early_stop_time
    early_stop_threshold = solver_cfg.subproblem_early_stop_threshold

    def _early_stop(model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            runtime  = model.cbGet(GRB.Callback.RUNTIME)
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if runtime > early_stop_time and best_obj < early_stop_threshold:
                best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                gap = abs(best_obj - best_bound) / abs(best_obj) if abs(best_obj) > 1e-10 else float("inf")
                print(f"[Subproblem] Early stop — obj={best_obj:.6f}  bound={best_bound:.6f}  gap={gap*100:.2f}%")
                model.terminate()

    sub.optimize(_early_stop)

    # ------------------------------------------------------------------
    # Handle infeasibility
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Handle infeasibility
    # ------------------------------------------------------------------
    if sub.status == GRB.Status.INFEASIBLE:
        print(f"[Subproblem] Infeasible for truck={truck}, day={day}")
        return [SubproblemResult(
            reduced_cost=np.inf,
            route_duration=np.inf,
            route_edges=None,
            route_stops=None,
            truck=truck,
            day=day,
            infeasible=True,
        )]

    if sub.status not in (GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT,
                          GRB.Status.INTERRUPTED):
        print(f"[Subproblem] Unexpected status {sub.status} for truck={truck}, day={day}")

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    reduced_cost    = sub.ObjVal - iota[truck, day]
    route_duration  = 0.0
    new_route_edges = np.zeros((N, N))
    route_stops     = [parking]

    if reduced_cost <= -1e-4:
        for (i, j) in E:
            if abs(X[i, j].X) > 0.5:
                new_route_edges[i, j] = 1
                route_duration += c[i, j]

        current = parking
        while True:
            next_node = int(np.argmax(new_route_edges[current]))
            route_stops.append(next_node)
            current = next_node
            if current == parking:
                break

    _quality = ("optimal"
                if sub.status == GRB.Status.OPTIMAL
                else "suboptimal")
    return [SubproblemResult(
        reduced_cost=reduced_cost,
        route_duration=route_duration,
        route_edges=new_route_edges,
        route_stops=route_stops,
        truck=truck,
        day=day,
        generation_quality=_quality,
    )]