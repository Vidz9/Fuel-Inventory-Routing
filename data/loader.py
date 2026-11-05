"""
data/loader.py
==============
Loads all problem data from Excel input files and returns a single
``IRPData`` dataclass that the rest of the codebase can pass around.

Usage
-----
    from data.loader import load_data

    data = load_data(network_size="small", data_dir="path/to/Data")
    # data.Nodes, data.Trucks, data.Durations, data.tank_cap, ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from data.models import Location, Node, Product, Truck


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class IRPData:
    """
    All static problem data, fully loaded and ready for solvers.

    Attributes
    ----------
    Products        : list of all Product objects
    Locations       : list of all Location objects
    Nodes           : list of all Node objects (stations + depots + 1 parking)
    Trucks          : list of all Truck objects
    Durations       : nested dict  Durations[loc1_name][loc2_name] -> float (hours)
    tank_cap        : 2D numpy array  (n_trucks × n_compartments), capacities in kl
    station_indices : node indices whose location_type == 'Station'
    depot_indices   : node indices whose location_type == 'Depot'
    parking_index   : node index of the single Parking node
    n_stations      : number of distinct station locations
    n_depots        : number of distinct depot locations
    V               : number of trucks
    N               : number of nodes
    P               : number of products
    C               : number of compartments per truck
    """

    Products: List[Product]
    Locations: List[Location]
    Nodes: List[Node]
    Trucks: List[Truck]
    Durations: Dict[str, Dict[str, float]]
    tank_cap: np.ndarray

    station_indices: List[int]
    depot_indices: List[int]
    parking_index: int

    n_stations: int
    n_depots: int

    # Derived counts and structures (populated automatically after __post_init__)
    V: int = field(init=False)
    N: int = field(init=False)
    P: int = field(init=False)
    C: int = field(init=False)
    feasible_edges: list = field(init=False)  # product-compatible arcs, computed once
    node_indices_to_product_dict: dict = field(init=False)  # node_index -> product object

    def __post_init__(self):
        self.V = len(self.Trucks)
        self.N = len(self.Nodes)
        self.P = len(self.Products)
        self.C = self.Trucks[0].compartments if self.Trucks else 0
        # Import here to avoid circular imports (milp imports loader)
        from solvers.milp import build_edge_set
        self.feasible_edges = build_edge_set(self.Nodes)
        self.node_indices_to_product_dict = {
            node.number: node.product
            for node in self.Nodes
            if node.location_type in ("Station", "Depot")
        }


# ---------------------------------------------------------------------------
# Helper: resolve the data file path
# ---------------------------------------------------------------------------

def _data_path(data_dir: str, network_size: str) -> str:
    """Return the full path to the Excel file for a given network size."""
    filename = f"Input_data_{network_size}.xlsx"
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Expected a file named '{filename}' inside '{data_dir}'."
        )
    return path


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_data(
    network_size: str = "small",
    data_dir: str = "Data",
    one_compartment: bool = True,
) -> IRPData:
    """
    Read all input Excel sheets and construct an ``IRPData`` object.

    Parameters
    ----------
    network_size    : 'small', 'medium', 'big' or 'company'
                      (reads Input_data_<network_size>.xlsx)
    data_dir        : path to the folder that contains the Excel files
    one_compartment : if True, merge all tank compartments into one
                      (i.e. treat each truck as single-compartment)

    Returns
    -------
    IRPData
    """
    path = _data_path(data_dir, network_size)

    # -----------------------------------------------------------------------
    # Read sheets
    # -----------------------------------------------------------------------
    vehicle_info    = pd.read_excel(path, sheet_name="Trucks")
    duration_matrix = pd.read_excel(path, sheet_name="Duration matrix")
    locations_df    = pd.read_excel(path, sheet_name="Locations")
    stations_df     = pd.read_excel(path, sheet_name="Gas stations")
    depot_df        = pd.read_excel(path, sheet_name="Depots")

    # -----------------------------------------------------------------------
    # Products  (inferred from depot sheet — every product stored at depots)
    # -----------------------------------------------------------------------
    Products: List[Product] = [
        Product(number=p, name=name)
        for p, name in enumerate(depot_df["Product"].unique())
    ]
    product_by_name: Dict[str, Product] = {prod.name: prod for prod in Products}

    # -----------------------------------------------------------------------
    # Locations and Nodes
    # -----------------------------------------------------------------------
    Locations: List[Location] = []
    Nodes: List[Node] = []
    station_indices: List[int] = []
    depot_indices: List[int] = []

    for i, row in locations_df.iterrows():
        loc_name    = row["Location ID"]
        loc_type    = row["Type"]
        latitude    = row["Latitude"]
        longitude   = row["Longitude"]

        loc = Location(len(Locations), loc_name, latitude, longitude)
        Locations.append(loc)

        if loc_type == "Gas station":
            station_products = stations_df[stations_df["Location ID"] == loc_name]
            for _, sp_row in station_products.iterrows():
                product = product_by_name[sp_row["Product"]]
                capacity    = sp_row["Capacity"] / 1000        # litres → kl
                safety      = sp_row["Safety stock"] / 1000
                node = Node(len(Nodes), "Station", loc, product, capacity, safety)
                Nodes.append(node)
                station_indices.append(node.number)

        elif loc_type == "Depot":
            depot_products = depot_df[depot_df["Location ID"] == loc_name]
            for _, dp_row in depot_products.iterrows():
                product = product_by_name[dp_row["Product"]]
                node = Node(len(Nodes), "Depot", loc, product, np.inf, 0.0)
                Nodes.append(node)
                depot_indices.append(node.number)

        else:
            # Parking / home base — one dummy node, no real product
            dummy_product = Product(len(Products), "Nothing")
            node = Node(len(Nodes), "Parking", loc, dummy_product, np.inf, 0.0)
            Nodes.append(node)

    parking_index = next(
        n.number for n in Nodes if n.location_type == "Parking"
    )

    # -----------------------------------------------------------------------
    # Duration matrix   Durations[loc1_name][loc2_name] -> float hours
    # -----------------------------------------------------------------------
    Durations: Dict[str, Dict[str, float]] = {}
    for loc1 in Locations:
        Durations[loc1.name] = {}
        row = duration_matrix[duration_matrix["Location ID"] == loc1.name]
        for loc2 in Locations:
            Durations[loc1.name][loc2.name] = float(row[loc2.name].iloc[0])

    # -----------------------------------------------------------------------
    # Trucks and tank capacity matrix
    # -----------------------------------------------------------------------
    Trucks: List[Truck] = []
    n_raw_tanks = len(vehicle_info.columns) - 3   # first 3 cols are metadata

    for _, vrow in vehicle_info.iterrows():
        name = vrow["Truck"]
        if one_compartment:
            # Sum all compartment capacities into one
            total = sum(vrow.iloc[j + 3] for j in range(n_raw_tanks)) / 1000
            compartment_caps = np.array([total])
            n_tanks = 1
        else:
            compartment_caps = np.array(
                [vrow.iloc[j + 3] / 1000 for j in range(n_raw_tanks)]
            )
            n_tanks = n_raw_tanks

        Trucks.append(Truck(name=name, compartments=n_tanks, compartment_capacities=compartment_caps))

    # tank_cap[v, c] = capacity of compartment c of truck v
    tank_cap = np.zeros((len(Trucks), Trucks[0].compartments))
    for v, truck in enumerate(Trucks):
        for c, cap in enumerate(truck.compartment_capacities):
            tank_cap[v, c] = cap

    # -----------------------------------------------------------------------
    # Location-level counts (distinct physical locations, not nodes)
    # -----------------------------------------------------------------------
    n_stations = stations_df["Location ID"].nunique()
    n_depots   = depot_df["Location ID"].nunique()

    return IRPData(
        Products=Products,
        Locations=Locations,
        Nodes=Nodes,
        Trucks=Trucks,
        Durations=Durations,
        tank_cap=tank_cap,
        station_indices=station_indices,
        depot_indices=depot_indices,
        parking_index=parking_index,
        n_stations=n_stations,
        n_depots=n_depots,
    )