"""
solvers/master.py
=================
Restricted Master Problem (RMP) for column generation.

Public API
----------
    build_rmp_model(...)  ->  (model, RMPVars)
    solve_rmp(model, vars, ...)  ->  RMPSolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import PenaltyConfig
from data.loader import IRPData
from instance import IRPInstance


# ---------------------------------------------------------------------------
# Return-type containers
# ---------------------------------------------------------------------------

@dataclass
class RMPVars:
    W:  gp.tupledict
    F:  gp.tupledict
    I:  gp.tupledict
    Ip: gp.tupledict
    Im: gp.tupledict
    B:  gp.tupledict


@dataclass
class RMPSolution:
    obj_val:         float
    levels:          np.ndarray
    routes_selected: np.ndarray
    edges_selected:  Dict[Tuple[int, int], np.ndarray]
    sigma:           np.ndarray
    iota:            np.ndarray
    W_sol:           Dict[Tuple[int, int, int], float]
    F_sol:           Dict[Tuple[int, int, int, int], float]
    I_sol:           Dict[Tuple[int, int], float]
    Ip_sol:          Dict[Tuple[int, int], float]
    Im_sol:          Dict[Tuple[int, int], float]
    B_sol:           Dict[Tuple[int, int], float]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_rmp_model(
    instance:   IRPInstance,
    penalties:  PenaltyConfig,
    env:        gp.Env,
    continuous: bool = True,
) -> Tuple[gp.Model, RMPVars]:
    """
    Construct the RMP Gurobi model and return it together with variable refs.

    Performance improvements vs the naive formulation
    --------------------------------------------------
    1. W is created only for feasible (v,t,r) triples — no zero-fixing
       constraints needed for infeasible routes.
    2. F is created only for edges in the feasible edge set E_all
       (~450 edges for the big network vs N²=1681).  Edges not yet used
       by any route get ub=0 so no extra constraints are needed.
    3. cap_v (truck capacity) is precomputed once outside the loops.
    """
    data   = instance.data
    routes = instance.routes

    V  = data.V
    N  = data.N
    T  = len(next(iter(instance.demand.values())))
    R  = routes.R

    demand          = instance.demand
    initial_levels  = instance.initial_levels
    route_durations = routes.route_durations
    route_edges     = routes.route_edges
    edge_routes     = routes.edge_routes
    feasible_routes = routes.feasible_routes
    tank_cap        = data.tank_cap
    Nodes           = data.Nodes

    phi1 = penalties.phi1
    phi2 = penalties.phi2

    # ------------------------------------------------------------------
    # Precompute truck capacities and feasible edge set
    # ------------------------------------------------------------------
    cap_v = {v: float(np.sum(tank_cap[v])) for v in range(V)}

    E_all = data.feasible_edges      # computed once in IRPData.__post_init__

    # Adjacency lists for fast flow-conservation constraint building
    in_E:  Dict[int, List[int]] = {n: [] for n in range(N)}
    out_E: Dict[int, List[int]] = {n: [] for n in range(N)}
    for (i, j) in E_all:
        out_E[i].append(j)
        in_E[j].append(i)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model_name = "RMP" if continuous else "RMP_integer"
    model = gp.Model(model_name, env=env)

    w_type = GRB.CONTINUOUS if continuous else GRB.BINARY

    # ---- W: only for feasible (v,t,r) combinations ----
    # This eliminates O(V*T*(R - |feasible|)) zero-fixing constraints.
    W_keys = [
        (v, t, r)
        for v in range(V)
        for t in range(T)
        for r in feasible_routes[(v, t)]
    ]
    W = model.addVars(W_keys, vtype=w_type, lb=0, ub=1, name="Routes")

    # ---- F: only over feasible edges ----
    # Edges not yet in any route get ub=0 (no flow possible until a route
    # uses them).  This avoids both the N² variable count and the need for
    # explicit "flow=0" constraints on unused edges.
    F_keys = [
        (v, t, i, j)
        for v in range(V)
        for t in range(T)
        for (i, j) in E_all
    ]
    
    F = model.addVars(
        F_keys,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=GRB.INFINITY,
        name="Flow",
    )

    I  = model.addVars(T, N, lb=-GRB.INFINITY, name="Inventory")
    Ip = model.addVars(T, N, lb=0,             name="InventPlus")
    Im = model.addVars(T, N, lb=0,             name="InventMinus")
    B  = model.addVars(T, N, lb=0,             name="BelowSafety")

    # ------------------------------------------------------------------
    # Inventory constraints  (station nodes only)
    # ------------------------------------------------------------------
    for node in Nodes:
        n = node.number
        if node.location_type != "Station":
            continue

        for t in range(T):
            inflow  = gp.quicksum(
                F[v, t, j, n] for v in range(V) for j in in_E[n]
                if (v, t, j, n) in F
            )
            outflow = gp.quicksum(
                F[v, t, n, j] for v in range(V) for j in out_E[n]
                if (v, t, n, j) in F
            )

            model.addConstr(I[t, n] == Ip[t, n] - Im[t, n])

            if t == 0:
                model.addConstr(
                    I[t, n] == initial_levels[node] - demand[node][t] + inflow - outflow,
                    name=f"Pi {t},{n}",
                )
            else:
                model.addConstr(
                    I[t, n] == Ip[t - 1, n] - demand[node][t] + inflow - outflow,
                    name=f"Pi {t},{n}",
                )

            model.addConstr(Ip[t, n] + B[t, n] >= node.safety,  name=f"Alpha {t},{n}")
            model.addConstr(I[t, n] <= node.capacity,             name=f"Beta {t},{n}")

    # ------------------------------------------------------------------
    # Route-selection constraints (Iota)
    # ------------------------------------------------------------------
    for v in range(V):
        for t in range(T):
            model.addConstr(
                gp.quicksum(W[v, t, r] for r in feasible_routes[(v, t)]) == 1,
                name=f"Iota {v},{t}",
            )
            # No zero-fixing loop needed: W only exists for feasible routes

    # ------------------------------------------------------------------
    # Flow constraints per truck per day
    # ------------------------------------------------------------------
    for v in range(V):
        for t in range(T):
            for node in Nodes:
                n = node.number
                inflow  = gp.quicksum(
                    F[v, t, j, n] for j in in_E[n] if (v, t, j, n) in F
                )
                outflow = gp.quicksum(
                    F[v, t, n, j] for j in out_E[n] if (v, t, n, j) in F
                )

                if node.location_type == "Depot":
                    model.addConstr(inflow == 0,                name=f"Gamma {v},{t},{n}")
                    model.addConstr(-inflow + outflow >= 0,     name=f"Delta {v},{t},{n}")
                elif node.location_type == "Station":
                    model.addConstr(inflow - outflow >= 0,      name=f"Delta {v},{t},{n}")
                else:  # Parking
                    if in_E[n]:
                        model.addConstr(inflow == 0)
                    if out_E[n]:
                        model.addConstr(outflow == 0)

                # Sigma constraints: link F to W for edges with known routes.
                # Edges not in edge_routes already have F ub=0 so no constraint needed.
                for j in out_E[n]:
                    if (n, j) in edge_routes:
                        model.addConstr(
                            -F[v, t, n, j]
                            + cap_v[v]
                            * gp.quicksum(
                                W[v, t, r]
                                for r in edge_routes[(n, j)]
                                if (v, t, r) in W
                            ) >= 0,
                            name=f"Sigma {v},{t},{n},{j}",
                        )
                    else:
                        model.addConstr(
                            -F[v, t, n, j]
                             >= 0,
                            name=f"Sigma {v},{t},{n},{j}",
                        )
    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    obj = gp.quicksum(
        W[v, t, r] * route_durations[r]
        for (v, t, r) in W_keys
    )
    obj += phi1 * gp.quicksum(B[t, n]  for t in range(T) for n in range(N))
    obj += phi2 * gp.quicksum(Im[t, n] for t in range(T) for n in range(N))

    model.setObjective(obj, GRB.MINIMIZE)

    return model, RMPVars(W=W, F=F, I=I, Ip=Ip, Im=Im, B=B)


# ---------------------------------------------------------------------------
# Solution extractor
# ---------------------------------------------------------------------------

def solve_rmp(
    model:    gp.Model,
    vars:     RMPVars,
    instance: IRPInstance,
) -> RMPSolution:
    model.optimize()

    data   = instance.data
    routes = instance.routes
    V, N   = data.V, data.N
    T      = len(next(iter(instance.demand.values())))
    R      = routes.R

    route_edges = routes.route_edges
    W, F, I, Ip, Im, B = vars.W, vars.F, vars.I, vars.Ip, vars.Im, vars.B

    print(f"[RMP] objective = {model.ObjVal:.8f}")

    # ------------------------------------------------------------------
    # Primal: routes selected and edge usage
    # ------------------------------------------------------------------
    routes_selected = np.zeros((V, T, R))
    edges_selected:  Dict[Tuple[int, int], np.ndarray] = {}
    levels = np.zeros((N, T))

    for (v, t, r), var in W.items():
        val = var.X
        if abs(val) > 1e-6:
            routes_selected[v, t, r] = val
            if (v, t) in edges_selected:
                edges_selected[(v, t)] += val * route_edges[r]
            else:
                edges_selected[(v, t)] = val * route_edges[r].copy()

    for t in range(T):
        for n in range(N):
            levels[n, t] = I[t, n].X

    # ------------------------------------------------------------------
    # Duals: sigma and iota
    # ------------------------------------------------------------------
    sigma = np.zeros((V, T, N, N))
    iota  = np.zeros((V, T))

    for constr in model.getConstrs():
        name = constr.ConstrName
        pi   = constr.Pi

        if abs(pi) > 1e-5 and name.startswith("Sigma "):
            parts = name[6:].split(",")
            v, t, n1, n2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            sigma[v, t, n1, n2] = pi

        elif abs(pi) > 1e-5 and name.startswith("Iota "):
            parts = name[5:].split(",")
            v, t = int(parts[0]), int(parts[1])
            iota[v, t] = pi

    # ------------------------------------------------------------------
    # Sparse solution dicts
    # ------------------------------------------------------------------
    W_sol = {
        (v, t, r): var.X
        for (v, t, r), var in W.items()
        if var.X > 1e-6
    }

    F_sol = {
        (v, t, i, j): var.X
        for (v, t, i, j), var in F.items()
        if var.X > 1e-6
    }

    I_sol  = {(t, n): I[t, n].X  for t in range(T) for n in range(N)}
    Ip_sol = {(t, n): Ip[t, n].X for t in range(T) for n in range(N)}
    Im_sol = {(t, n): Im[t, n].X for t in range(T) for n in range(N)}
    B_sol  = {(t, n): B[t, n].X  for t in range(T) for n in range(N)}

    return RMPSolution(
        obj_val=model.ObjVal,
        levels=levels,
        routes_selected=routes_selected,
        edges_selected=edges_selected,
        sigma=sigma,
        iota=iota,
        W_sol=W_sol,
        F_sol=F_sol,
        I_sol=I_sol,
        Ip_sol=Ip_sol,
        Im_sol=Im_sol,
        B_sol=B_sol,
    )