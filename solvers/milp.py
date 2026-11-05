"""
solvers/milp.py
===============
Two MILP formulations for the Inventory Routing Problem.

``solve_milp_flow``
    Arc-flow formulation (the primary, cleaner model).
    Routes and flows are indexed over a restricted edge set E,
    which only includes product-compatible arcs. This gives a
    tighter LP relaxation and faster solve times.

``solve_milp_compartment``
    Original compartment-based formulation.
    Explicitly models each truck compartment, loading/unloading
    quantities, and in-transit quantities Q.  More variables but
    needed when one_compartment=False to enforce product separation
    between compartments.

Both functions accept an IRPInstance and return a MILPSolution.

Shared helpers
--------------
    build_edge_set(Nodes)
        Build the set of valid arcs respecting product compatibility.
        Used by both MILP formulations and by the greedy heuristic.

    make_checkpoint_callback(record_times)
        Factory that returns a Gurobi callback recording the incumbent
        objective at a list of wall-clock time checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import PenaltyConfig, ProblemConfig, SolverConfig
from data.models import Node
from instance import IRPInstance


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class MILPSolution:
    """
    All outputs from a single MILP solve.

    Attributes
    ----------
    obj_val             : best objective value found (np.inf if no solution)
    gap                 : MIP optimality gap at termination
    routes              : dict  (v,t) -> (N×N) binary edge matrix
    flows               : dict  (v,t) -> dict {(i,j): flow_value}
    loadings            : dict  (v,t) -> array length N  (amounts loaded)
    unloadings          : dict  (v,t) -> array length N  (amounts unloaded)
    levels              : array shape (N, T)  inventory levels
    objective_breakdown : dict with keys transport_cost, below_safety_penalty,
                          below_zero_penalty, total
    objective_over_time : dict  checkpoint_seconds -> obj_val
    """
    obj_val:              float
    gap:                  float
    routes:               Dict[Tuple[int, int], np.ndarray]
    flows:                Dict[Tuple[int, int], Dict[Tuple[int, int], float]]
    loadings:             Dict[Tuple[int, int], np.ndarray]
    unloadings:           Dict[Tuple[int, int], np.ndarray]
    levels:               np.ndarray
    objective_breakdown:  Dict[str, float]
    objective_over_time:  Dict[int, float]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_edge_set(Nodes: List[Node]) -> List[Tuple[int, int]]:
    """
    Return the set of valid arcs for the arc-flow formulation.

    Only product-compatible arcs are included, which tightens the
    LP relaxation compared to a full N×N edge set.

    Rules
    -----
    1. station  -> station  (same product only)
    2. station  -> depot    (any depot)
    3. depot    -> station  (same product only)
    4. parking  -> depot    (route starts by going to a depot to load)
    5. station  -> parking  (route ends by returning to parking from a station)
    """
    E: List[Tuple[int, int]] = []
    for ni in Nodes:
        for nj in Nodes:
            if ni is nj:
                continue
            i, j = ni.number, nj.number
            ti, tj = ni.location_type, nj.location_type

            if ti == "Station" and tj == "Station" and ni.product == nj.product:
                E.append((i, j))
            elif ti == "Station" and tj == "Depot":
                E.append((i, j))
            elif ti == "Depot" and tj == "Station" and ni.product == nj.product:
                E.append((i, j))
            elif ti == "Parking" and tj == "Depot":
                E.append((i, j))
            elif ti == "Station" and tj == "Parking":
                E.append((i, j))
    return E


def make_checkpoint_callback(
    record_times: List[int],
    recorded: Dict,
) -> callable:
    """
    Return a Gurobi callback that records the incumbent objective
    at each wall-clock checkpoint in ``record_times`` (seconds).

    Results are written into ``recorded``  {seconds -> obj_value}.
    """
    _idx = {"i": 0}

    def _callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        i = _idx["i"]
        if i >= len(record_times):
            return
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        if runtime >= record_times[i]:
            recorded[record_times[i]] = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            _idx["i"] += 1

    return _callback


def _fill_missing_checkpoints(
    recorded: Dict[int, float],
    record_times: List[int],
    final_obj: float,
) -> None:
    """Fill any checkpoints not yet reached with the final objective value."""
    for t in record_times:
        if t not in recorded:
            recorded[t] = final_obj


# ---------------------------------------------------------------------------
# Arc-flow MILP  (primary formulation)
# ---------------------------------------------------------------------------

def solve_milp_flow(
    instance:    IRPInstance,
    penalties:   PenaltyConfig,
    problem_cfg: ProblemConfig,
    solver_cfg:  SolverConfig,
    env:         gp.Env,
) -> MILPSolution:
    """
    Solve the IRP arc-flow MILP formulation.

    Decision variables
    ------------------
    X[v,t,i,j]  : binary — does truck v traverse arc (i,j) on day t?
    F[v,t,i,j]  : continuous — fuel flow on arc (i,j) for truck v on day t
    I[t,n]      : signed inventory at node n at end of period t
    Ip, Im      : positive/negative parts of I  (I = Ip − Im)
    B[t,n]      : shortfall below safety stock
    S[v,t,n]    : Miller-Tucker-Zemlin subtour-elimination variable
    """
    data = instance.data
    V, N = data.V, data.N
    T    = len(next(iter(instance.demand.values())))
    M    = problem_cfg.M

    Nodes     = data.Nodes
    Durations = data.Durations
    tank_cap  = data.tank_cap

    demand         = instance.demand
    initial_levels = instance.initial_levels

    phi1 = penalties.phi1
    phi2 = penalties.phi2
    daily_limit = problem_cfg.daily_driving_limit

    # ---- Derived index sets ----
    N_G    = [n.number for n in Nodes if n.location_type == "Station"]
    N_D    = [n.number for n in Nodes if n.location_type == "Depot"]
    o      = data.parking_index
    E      = build_edge_set(Nodes)
    E_set: Set[Tuple[int, int]] = set(E)


    # ---- Model ----
    model = gp.Model("IRP_flow", env=env)

    X  = model.addVars(V, T, E, vtype=GRB.BINARY,       lb=0,            name="Routes")
    F  = model.addVars(V, T, E,                          lb=0,            name="Flow")
    I  = model.addVars(T, N,                             lb=-GRB.INFINITY, name="Inventory")
    Ip = model.addVars(T, N,                             lb=0,            name="InventoryPlus")
    Im = model.addVars(T, N,                             lb=0,            name="InventoryMinus")
    B  = model.addVars(T, N,                             lb=0,            name="BelowSafety")
    S  = model.addVars(V, T, N,                          lb=0,            name="Subtour")

    # ---- Inventory constraints ----
    for node in Nodes:
        i = node.number
        if node.location_type != "Station":
            continue

        for t in range(T):
            model.addConstr(I[t, i] == Ip[t, i] - Im[t, i])

            inflow  = gp.quicksum(
                F[v, t, j, i] for v in range(V) for j in range(N) if (j, i) in E_set
            )
            outflow = gp.quicksum(
                F[v, t, i, j] for v in range(V) for j in range(N) if (i, j) in E_set
            )

            if t == 0:
                model.addConstr(
                    I[t, i] == initial_levels[node] - demand[node][t] + inflow - outflow
                )
            else:
                model.addConstr(
                    I[t, i] == Ip[t - 1, i] - demand[node][t] + inflow - outflow
                )

            model.addConstr(Ip[t, i] + B[t, i] >= node.safety)
            model.addConstr(I[t, i] <= node.capacity)

    # ---- Routing constraints ----
    for v in range(V):
        for t in range(T):
            # Each truck starts at parking and ends at parking
            model.addConstr(gp.quicksum(X[v, t, o, j] for j in N_D if (o, j) in E_set) == 1)
            model.addConstr(gp.quicksum(X[v, t, i, o] for i in N_G if (i, o) in E_set) == 1)

            # Flow conservation at every node
            for i in range(N):
                model.addConstr(
                    gp.quicksum(X[v, t, j, i] for j in range(N) if (j, i) in E_set)
                    == gp.quicksum(X[v, t, i, j] for j in range(N) if (i, j) in E_set)
                )

                #model.addConstr(gp.quicksum(X[v, t, i, j] for j in range(N) if (i, j) in E_set) <= 1)

            # Driving time limit
            model.addConstr(
                gp.quicksum(
                    X[v, t, i, j]
                    * (Durations[Nodes[i].location.name][Nodes[j].location.name] + 0.15)
                    for (i, j) in E
                ) <= daily_limit
            )

            # Subtour elimination (MTZ) — skip arcs returning to parking
            for (i, j) in E:
                if j != o:
                    model.addConstr(S[v, t, j] >= S[v, t, i] - M * (1 - X[v, t, i, j]) + 1)

    # ---- Flow / capacity constraints ----
    for v in range(V):
        for t in range(T):
            for (i, j) in E:
                # Arc flow cannot exceed truck capacity
                model.addConstr(F[v, t, i, j] <= tank_cap[v][0] * X[v, t, i, j])

            # Stations: trucks can only unload (net delivery >= 0)
            for i in N_G:
                model.addConstr(
                    gp.quicksum(F[v, t, j, i] for j in range(N) if (j, i) in E_set)
                    - gp.quicksum(F[v, t, i, j] for j in range(N) if (i, j) in E_set) >= 0
                )

            # Depots: trucks can only load (net pickup >= 0)
            for i in N_D:
                model.addConstr(
                    gp.quicksum(F[v, t, j, i] for j in range(N) if (j, i) in E_set)
                    - gp.quicksum(F[v, t, i, j] for j in range(N) if (i, j) in E_set) <= 0
                )

            # Trucks arrive at depots empty
            for j in N_D:
                model.addConstr(
                    gp.quicksum(F[v, t, i, j] for i in range(N) if (i, j) in E_set) == 0
                )

            # Trucks leave and return to parking empty
            model.addConstr(
                gp.quicksum(F[v, t, o, j] for j in range(N) if (o, j) in E_set) == 0
            )
            model.addConstr(
                gp.quicksum(F[v, t, j, o] for j in range(N) if (j, o) in E_set) == 0
            )

    # ---- Objective ----
    obj = gp.quicksum(
        X[v, t, i, j]
        * (Durations[Nodes[i].location.name][Nodes[j].location.name] + 0.15)
        for v in range(V) for t in range(T) for (i, j) in E
    )
    obj += phi1 * gp.quicksum(B[t, i]  for t in range(T) for i in range(N))
    obj += phi2 * gp.quicksum(Im[t, i] for t in range(T) for i in range(N))
    model.setObjective(obj, GRB.MINIMIZE)
    model.setParam("TimeLimit", 60 * solver_cfg.time_limit)

    model.setParam("TimeLimit", 0.2*60) ##remove later

    # ---- Solve with checkpoint callback ----
    record_times = solver_cfg.record_times_milp
    recorded: Dict[int, float] = {}
    model.optimize(make_checkpoint_callback(record_times, recorded))

    if model.SolCount == 0:
        return MILPSolution(
            obj_val=np.inf, gap=np.inf,
            routes={}, flows={}, loadings={}, unloadings={},
            levels=np.zeros((N, T)),
            objective_breakdown={}, objective_over_time={},
        )

    _fill_missing_checkpoints(recorded, record_times, model.ObjVal)

    # ---- Extract solution ----
    routes:     Dict = {}
    flows:      Dict = {}
    levels = np.zeros((N, T))

    for (v, t, i, j), var in X.items():
        if var.X > 0.5:
            routes.setdefault((v, t), []).append((i, j))

    for (v, t, i, j), var in F.items():
        if var.X > 1e-6:
            flows.setdefault((v, t), {})[(i, j)] = var.X

    for (t, i), var in I.items():
        levels[i, t] = var.X

    transport_cost   = sum(
        X[v, t, i, j].X
        * (Durations[Nodes[i].location.name][Nodes[j].location.name] + 0.15)
        for v in range(V) for t in range(T) for (i, j) in E
        if X[v, t, i, j].X > 1e-6
    )
    below_safety = phi1 * sum(B[t, i].X  for t in range(T) for i in range(N) if B[t, i].X  > 1e-6)
    below_zero   = phi2 * sum(Im[t, i].X for t in range(T) for i in range(N) if Im[t, i].X > 1e-6)

    return MILPSolution(
        obj_val=model.ObjVal,
        gap=model.MIPGap,
        routes=routes,
        flows=flows,
        loadings={},       # arc-flow formulation: derive from flows if needed
        unloadings={},
        levels=levels,
        objective_breakdown={
            "transport_cost":       transport_cost,
            "below_safety_penalty": below_safety,
            "below_zero_penalty":   below_zero,
            "total":                transport_cost + below_safety + below_zero,
        },
        objective_over_time=recorded,
    )


# ---------------------------------------------------------------------------
# Compartment-based MILP  (original formulation)
# ---------------------------------------------------------------------------

def solve_milp_compartment(
    instance:    IRPInstance,
    penalties:   PenaltyConfig,
    problem_cfg: ProblemConfig,
    solver_cfg:  SolverConfig,
    env:         gp.Env,
) -> MILPSolution:
    """
    Solve the IRP compartment-based MILP formulation.

    This is the original formulation. Use this when one_compartment=False
    to enforce that each compartment carries only one product type.

    Additional decision variables over solve_milp_flow
    ---------------------------------------------------
    U[v,t,n,c,p]  : amount unloaded at node n from compartment c (product p)
    L[v,t,n,c,p]  : amount loaded  at node n into compartment c (product p)
    Q[v,t,n,c,p]  : quantity in compartment c after leaving node n
    Y[v,t,n]      : binary — does truck v visit node n on day t?
    Z[v,t,n,c,p]  : binary — is compartment c allocated to product p at node n?
    """
    data = instance.data
    V, N = data.V, data.N
    T    = len(next(iter(instance.demand.values())))
    C    = data.C
    P    = data.P
    M    = problem_cfg.M

    Nodes     = data.Nodes
    Durations = data.Durations
    tank_cap  = data.tank_cap

    demand         = instance.demand
    initial_levels = instance.initial_levels

    daily_limit = problem_cfg.daily_driving_limit
    parking     = data.parking_index

    # Note: this formulation uses a full N×N arc matrix for routing
    # (not the restricted edge set E), relying on Y and structural
    # constraints to enforce product compatibility implicitly.

    model = gp.Model("IRP_compartment", env=env)

    # ---- Variables ----
    U = model.addVars(V, T, N, C, P,                    lb=0,            name="Unloadings")
    L = model.addVars(V, T, N, C, P,                    lb=0,            name="Loadings")
    Q = model.addVars(V, T, N, C, P,                    lb=0,            name="Quantities")
    X = model.addVars(V, T, N, N, vtype=GRB.BINARY,     lb=0, ub=1,      name="Routes")
    Y = model.addVars(V, T, N,    vtype=GRB.BINARY,     lb=0, ub=1,      name="Locations")
    Z = model.addVars(V, T, N, C, P, vtype=GRB.BINARY,  lb=0, ub=1,      name="Compartments")
    S = model.addVars(V, T, N,                                            name="Subtours")
    I = model.addVars(T, N,                              lb=-GRB.INFINITY, name="Levels")
    MS= model.addVars(T, N,                              lb=0,            name="Missing")

    # ---- Inventory constraints ----
    for node in Nodes:
        n = node.number
        if node.location_type != "Station":
            continue
        p = node.product.number
        for t in range(T):
            deliveries = gp.quicksum(U[v, t, n, c, p] for v in range(V) for c in range(C))
            if t == 0:
                model.addConstr(I[t, n] == initial_levels[node] - demand[node][t] + deliveries)
            else:
                model.addConstr(I[t, n] == I[t - 1, n] - demand[node][t] + deliveries)

            model.addConstr(I[t, n] + MS[t, n] >= node.safety)
            model.addConstr(I[t, n] <= node.capacity)

    # ---- Routing constraints ----
    for t in range(T):
        for v in range(V):
            # Route begins and ends at parking
            model.addConstr(gp.quicksum(X[v, t, parking, n] for n in range(N)) == 1)
            model.addConstr(gp.quicksum(X[v, t, n, parking] for n in range(N)) == 1)

            for node in Nodes:
                n = node.number
                out = gp.quicksum(X[v, t, n, m] for m in range(N))
                inc = gp.quicksum(X[v, t, m, n] for m in range(N))

                # Flow conservation
                model.addConstr(out == inc)
                # Y[v,t,n] = 1 iff truck v visits node n on day t
                model.addConstr(out + inc == 2 * Y[v, t, n])

                # Structural arc restrictions
                if node.location_type != "Station":
                    # Non-stations cannot connect back to parking
                    # (only stations are valid last stops)
                    model.addConstr(X[v, t, n, parking] == 0)
                else:
                    # Stations cannot be reached directly from parking
                    # (truck must load at a depot first)
                    model.addConstr(X[v, t, parking, n] == 0)

            # No self-loops
            model.addConstr(gp.quicksum(X[v, t, n, n] for n in range(N)) == 0)

            # Driving time limit
            model.addConstr(
                gp.quicksum(
                    X[v, t, n, m]
                    * (Durations[Nodes[n].location.name][Nodes[m].location.name] + 0.15)
                    for n in range(N) for m in range(N)
                ) <= daily_limit
            )

            # Subtour elimination (MTZ, excluding arcs to parking)
            model.addConstrs(
                S[v, t, n] >= S[v, t, m] - M * (1 - X[v, t, m, n]) + 1
                for n in range(N) if n != parking
                for m in range(N)
            )

    # ---- Compartment / capacity constraints ----
    for v in range(V):
        for t in range(T):
            # Each compartment carries at most one product
            model.addConstrs(
                gp.quicksum(Z[v, t, n, c, p] for p in range(P)) <= 1
                for n in range(N) for c in range(C)
            )
            # Quantity in compartment bounded by compartment capacity
            model.addConstrs(
                Q[v, t, n, c, p] <= tank_cap[v, c] * Z[v, t, n, c, p]
                for n in range(N) for c in range(C) for p in range(P)
            )
            # Total loaded == total unloaded per compartment per product
            model.addConstrs(
                gp.quicksum(L[v, t, n, c, p] for n in range(N))
                == gp.quicksum(U[v, t, n, c, p] for n in range(N))
                for c in range(C) for p in range(P)
            )

            for node in Nodes:
                n = node.number
                # U, L, Q can only be non-zero if truck visits the node
                model.addConstrs(U[v, t, n, c, p] <= M * Y[v, t, n] for c in range(C) for p in range(P))
                model.addConstrs(L[v, t, n, c, p] <= M * Y[v, t, n] for c in range(C) for p in range(P))
                model.addConstrs(Q[v, t, n, c, p] <= M * Y[v, t, n] for c in range(C) for p in range(P))

                if node.location_type == "Station":
                    # Q decreases by unloading amount after visiting a station
                    model.addConstrs(
                        Q[v, t, n, c, p]
                        <= Q[v, t, m, c, p] - U[v, t, n, c, p] + M * (1 - X[v, t, m, n])
                        for m in range(N) for c in range(C) for p in range(P)
                    )
                    # No loading at stations
                    model.addConstrs(L[v, t, n, c, p] == 0 for c in range(C) for p in range(P))

                elif node.location_type == "Depot":
                    # Q increases by loading amount after visiting a depot
                    model.addConstrs(
                        Q[v, t, n, c, p]
                        <= Q[v, t, m, c, p] + L[v, t, n, c, p] + M * (1 - X[v, t, m, n])
                        for m in range(N) for c in range(C) for p in range(P)
                    )
                    # No unloading at depots
                    model.addConstrs(U[v, t, n, c, p] == 0 for c in range(C) for p in range(P))
                    # Can only load the product this depot carries
                    depot_p  = node.product.number
                    other_ps = [p for p in range(P) if p != depot_p]
                    model.addConstrs(L[v, t, n, c, p] == 0 for c in range(C) for p in other_ps)

                elif node.location_type == "Parking":
                    # No activity at parking
                    model.addConstrs(U[v, t, n, c, p] == 0 for c in range(C) for p in range(P))
                    model.addConstrs(L[v, t, n, c, p] == 0 for c in range(C) for p in range(P))
                    model.addConstrs(Q[v, t, n, c, p] == 0 for c in range(C) for p in range(P))

    # Auxiliary: total unloaded == total loaded across all nodes (redundant but tightens)
    model.addConstrs(
        gp.quicksum(U[v, t, n, c, p] for n in range(N))
        == gp.quicksum(L[v, t, n, c, p] for n in range(N))
        for v in range(V) for t in range(T) for c in range(C) for p in range(P)
    )

    # ---- Objective ----
    obj = gp.quicksum(
        X[v, t, n, m]
        * (Durations[Nodes[n].location.name][Nodes[m].location.name] + 0.15)
        for v in range(V) for t in range(T) for n in range(N) for m in range(N)
    )
    obj += 1000 * gp.quicksum(MS[t, n] for t in range(T) for n in range(N))
    model.setObjective(obj, GRB.MINIMIZE)
    # Use 30% of the time limit (as in the original code)
    model.setParam("TimeLimit", 60 * solver_cfg.time_limit * 0.3)

    model.optimize()

    if model.SolCount == 0:
        return MILPSolution(
            obj_val=np.inf, gap=np.inf,
            routes={}, flows={}, loadings={}, unloadings={},
            levels=np.zeros((N, T)),
            objective_breakdown={}, objective_over_time={},
        )

    # ---- Extract solution ----
    routes:     Dict = {}
    loadings:   Dict = {}
    unloadings: Dict = {}
    levels = np.zeros((N, T))

    for (v, t, i, j), var in X.items():
        if var.X > 0.5:
            if (v, t) not in routes:
                routes[(v, t)] = np.zeros((N, N))
            routes[(v, t)][i, j] = 1

    for (v, t, i, c, p), var in L.items():
        if var.X > 0.01:
            loadings.setdefault((v, t), np.zeros(N))[i] += var.X

    for (v, t, i, c, p), var in U.items():
        if var.X > 0.01:
            unloadings.setdefault((v, t), np.zeros(N))[i] += var.X

    for (t, n), var in I.items():
        levels[n, t] = var.X

    return MILPSolution(
        obj_val=model.ObjVal,
        gap=model.MIPGap,
        routes=routes,
        flows={},          # compartment formulation doesn't use arc flows
        loadings=loadings,
        unloadings=unloadings,
        levels=levels,
        objective_breakdown={
            "transport_cost": model.ObjVal,   # breakdown not separately tracked
            "total":          model.ObjVal,
        },
        objective_over_time={},
    )
