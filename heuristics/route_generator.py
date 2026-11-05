"""
heuristics/route_generator.py
==============================
Random heuristic route generator used to warm-start column generation.

Generates feasible routes by randomly combining depot visits and station
visits, biased toward nearby locations (inverse-distance weighting).
Only routes that satisfy the driving-time limit and are not duplicates
are kept.

Public API
----------
    generate_routes(instance, data, max_routes) -> int
        Mutates instance.routes in-place and returns the new route count R.
"""

from __future__ import annotations

import numpy as np

from config import ProblemConfig
from data.loader import IRPData
from instance import IRPInstance, RouteRecord


def generate_routes(
    instance:    IRPInstance,
    problem_cfg: ProblemConfig,
    max_routes:  int = 100,
) -> int:
    """
    Populate ``instance.routes`` with up to ``max_routes`` randomly
    generated feasible routes and return the updated route count R.

    Each route is built by:
      1. Starting at the parking node.
      2. Randomly picking a depot (biased toward closer depots).
      3. Visiting one or more stations that carry the same product as
         the chosen depot (biased toward closer stations).
      4. Possibly repeating steps 2-3 (multi-depot sequences).
      5. Returning to the parking node.

    Routes that exceed the daily driving limit or are duplicates of
    already-known routes are discarded.

    Parameters
    ----------
    instance    : IRPInstance whose RoutePool will be mutated in-place
    problem_cfg : ProblemConfig providing daily_driving_limit
    max_routes  : target total number of routes (including route 0)

    Returns
    -------
    int : updated R (total routes in the pool after generation)
    """
    data   = instance.data
    routes = instance.routes

    Nodes           = data.Nodes
    Durations       = data.Durations
    depot_indices   = data.depot_indices
    station_indices = data.station_indices
    node_indices_to_product_dict = data.node_indices_to_product_dict
    V               = data.V
    T_horizon       = len(next(iter(instance.demand.values())))
    N               = data.N
    parking         = data.parking_index
    daily_limit     = problem_cfg.daily_driving_limit

    R = routes.R   # current route count (starts at 1 after create_instance)

    while R <= max_routes:
        visited_depots   = []
        visited_stations = []
        route            = [parking]
        route_duration   = 0.0
        infeasible       = False

        routes.route_edges[R] = np.zeros((N, N))

        # Each route can have 1 or 2 depot-station sequences
        n_sequences = np.random.choice([1, 2], p=[0.25, 0.75])

        for _ in range(n_sequences):
            # Cannot start a new sequence if we're already at a depot
            if route[-1] in depot_indices:
                infeasible = True
                break

            # ---- Pick a depot ----
            available_depots = [d for d in depot_indices if d not in visited_depots]
            if not available_depots:
                infeasible = True
                break

            # Inverse-distance weighting toward closer depots
            prob_depot = np.array([
                1.0 / max(Durations[Nodes[route[-1]].location.name][Nodes[d].location.name], 1e-6)
                for d in available_depots
            ])
            prob_depot /= prob_depot.sum()

            if _ == 0:
                prob_depot = np.ones(len(available_depots))/len(available_depots)

            chosen_depot     = int(np.random.choice(available_depots, p=prob_depot))
            depot_node       = Nodes[chosen_depot]
            depot_product    = depot_node.product

            route.append(chosen_depot)
            visited_depots.append(chosen_depot)
            leg = (
                Durations[Nodes[route[-2]].location.name][Nodes[route[-1]].location.name]
                + 0.15
            )
            route_duration                          += leg
            routes.route_edges[R][route[-2], route[-1]] = 1

            # ---- Visit compatible stations ----
            attempt = 0
            while True:
                available_stations = [s for s in station_indices if (s not in visited_stations and node_indices_to_product_dict[s]==depot_product)]
                if not available_stations:
                    break

                # Inverse-distance weighting toward closer stations
                prob_stn = np.array([
                    1.0 / (Durations[Nodes[route[-1]].location.name][Nodes[s].location.name] + 0.01)
                    for s in available_stations
                ])
                prob_stn /= prob_stn.sum()

                chosen_stn  = int(np.random.choice(available_stations, p=prob_stn))
                station_node = Nodes[chosen_stn]

                if station_node.product == depot_product:
                    # Compatible — add to route
                    attempt = 0
                    route.append(chosen_stn)
                    visited_stations.append(chosen_stn)
                    leg = (
                        Durations[Nodes[route[-2]].location.name][Nodes[route[-1]].location.name]
                        + 0.15
                    )
                    route_duration                              += leg
                    routes.route_edges[R][route[-2], route[-1]] = 1

                    # With 30% probability, stop visiting stations for this depot
                    if (n_sequences==2):
                        if np.random.uniform() < 0.3:
                            break
                    elif (n_sequences==1):
                        if np.random.uniform() < 0.2:
                            break
                    

            if infeasible:
                break
        
        if route[-1] in depot_indices:
            infeasible = True

        if infeasible:
            # Discard partial route and try again
            del routes.route_edges[R]
            continue

        # ---- Close the route back to parking ----
        route.append(parking)
        leg = (
            Durations[Nodes[route[-2]].location.name][Nodes[route[-1]].location.name]
            + 0.15
        )
        route_duration                          += leg
        routes.route_edges[R][route[-2], route[-1]] = 1

        # ---- Accept only feasible, non-duplicate routes ----
        if route_duration > daily_limit:
            del routes.route_edges[R]
            continue

        if route in routes.new_routes.values():
            del routes.route_edges[R]
            continue

        # ---- Register the new route ----
        routes.new_routes[R]      = route
        routes.route_durations.append(float(route_duration))

        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            if edge in routes.edge_routes:
                routes.edge_routes[edge].append(R)
            else:
                routes.edge_routes[edge] = [R]

        for v in range(V):
            for t in range(T_horizon):
                routes.feasible_routes[(v, t)].append(R)

        # ---- Build and store route metadata ----
        interior   = route[1:-1]   # exclude the opening/closing parking stops
        n_stations = sum(1 for s in interior if s in data.station_indices)
        n_depots   = sum(1 for s in interior if s in data.depot_indices)
        routes.route_records[R] = RouteRecord(
            route_id   = R,
            origin     = "warmstart",
            stops      = list(route),
            duration   = float(route_duration),
            n_stations = n_stations,
            n_depots   = n_depots,
            tree_level = None,      # warmstart routes are not from B&P tree
            reduced_cost = None,
        )

        print(list(route), n_stations, n_depots)

        R += 1

    routes.R = R
    return R