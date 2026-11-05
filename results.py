"""
results.py
==========
Dataclasses for storing experiment results.

Kept in a dedicated module so that pickle files produced by main.py
can be loaded in any other script or notebook simply by doing:

    from results import InstanceResult, MethodResult, NodeStat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NodeStat:
    """
    Per-node solution statistics, computed after each integer solve.

    Station fields
    --------------
    name              : location name
    node_idx          : index in Nodes list
    initial_level     : inventory at start of horizon
    demand            : list of demand per period (length T)
    delivered         : list of total delivered quantity per period (length T)
    levels            : list of end-of-period inventory levels (length T)
    n_visits          : total (truck, day) pairs that visited this station
    days_below_safety : number of periods where level < safety stock
    days_below_zero   : number of periods where level < 0

    Depot fields  (station fields not applicable are left as None / 0)
    ------------
    name              : location name
    node_idx          : index in Nodes list
    n_visits          : total (truck, day) pairs that departed from this depot
    qty_loaded        : total quantity loaded across all trucks and days
    """
    name:               str
    node_idx:           int

    # Station-specific (None for depots)
    initial_level:      Optional[float]       = None
    demand:             Optional[List[float]] = None
    delivered:          Optional[List[float]] = None
    levels:             Optional[List[float]] = None
    safety_stock:       Optional[float]       = None
    days_below_safety:  int                   = 0
    days_below_zero:    int                   = 0

    # Both station and depot
    n_visits:           int                   = 0

    # Depot-specific
    qty_loaded:         float                 = 0.0


@dataclass
class MethodResult:
    """
    All outputs from one solver method on one instance.

    Attributes
    ----------
    Objectives
        obj_val               : best integer objective (np.inf if no solution)
        lp_obj                : LP relaxation objective — CG/BP only, else None
        transport_cost        : routing time component of the objective
        below_safety_penalty  : phi1 * sum(B)
        below_zero_penalty    : phi2 * sum(Im)
        mip_gap               : final MIP gap reported by Gurobi

    Timing  (seconds)
        generation_time       : column generation / B&P tree search time
        solving_time          : final integer solve time
        total_time            : generation_time + solving_time

    Column pool
        n_routes              : total routes in pool at termination

    B&P specifics
        bp_nodes_explored     : tree nodes processed (None for non-BP methods)
        bp_best_obj_progress  : list of (wall_clock_s, obj_val) tuples

    Objective over time
        obj_over_time         : {checkpoint_seconds: obj_val} — always floats

    Full solution arrays
        levels                : np.ndarray shape (N, T)
        routes                : dict (v,t) -> edge matrix
        loadings              : dict (v,t) -> array length N
        unloadings            : dict (v,t) -> array length N
        flows                 : dict (v,t) -> {(i,j): flow}

    Node-level solution statistics
        station_stats         : dict node_idx -> NodeStat for each station
        depot_stats           : dict node_idx -> NodeStat for each depot
    """
    # Objectives
    obj_val:               float
    lp_obj:                Optional[float]            = None
    transport_cost:        float                       = np.nan
    below_safety_penalty:  float                       = np.nan
    below_zero_penalty:    float                       = np.nan
    mip_gap:               float                       = np.nan
    obj_bound:             float                       = np.nan

    # Timing
    generation_time:       float                       = 0.0
    solving_time:          float                       = 0.0
    total_time:            float                       = 0.0

    # Column pool
    n_routes:              int                         = 0

    # B&P specifics
    bp_nodes_explored:     Optional[int]               = None
    bp_best_obj_progress:  List[Tuple[float, float]]   = field(default_factory=list)

    # Progression — always {seconds: float}
    obj_over_time:         Dict[int, float]            = field(default_factory=dict)

    # Solution arrays
    levels:                Optional[np.ndarray]        = None
    routes:                Dict                        = field(default_factory=dict)
    loadings:              Dict                        = field(default_factory=dict)
    unloadings:            Dict                        = field(default_factory=dict)
    flows:                 Dict                        = field(default_factory=dict)

    # Route pool analysis
    # selected_route_ids : (v,t) -> route_id selected in the integer solution
    # route_records      : snapshot of RoutePool.route_records at solve time
    selected_route_ids:    Dict                        = field(default_factory=dict)
    route_records:         Dict                        = field(default_factory=dict)
    iteration_log:         list                        = field(default_factory=list)  # CG coalition log

    # Node-level statistics
    station_stats:         Dict[int, NodeStat]         = field(default_factory=dict)
    depot_stats:           Dict[int, NodeStat]         = field(default_factory=dict)


@dataclass
class InstanceResult:
    """
    All results for one instance under one (demand_coeff, V, T) setting.
    """
    # Identity
    instance_id:            int
    V:                      int
    T:                      int
    demand_coefficient:     float
    network_size:           str

    # Instance characteristics
    total_demand:           float                = np.nan
    demand_capacity_ratio:  float                = np.nan

    # Per-method results
    milp:          Optional[MethodResult]  = None
    cg:            Optional[MethodResult]  = None
    bp:            Optional[MethodResult]  = None
    bp_fractions:  Dict                    = field(default_factory=dict)  # {fraction -> MethodResult}
    warmstart:     Optional[MethodResult]  = None
    warmstart_cg:  Optional[MethodResult]  = None
    warmstart_bp:  Optional[MethodResult]  = None