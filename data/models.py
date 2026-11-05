"""
data/models.py
==============
Core domain objects for the Inventory Routing Problem.

These are pure data classes with no I/O or solver dependencies,
so they can be imported anywhere without side effects.
"""

from __future__ import annotations
from typing import List
import numpy as np


class Product:
    """A fuel product (e.g. diesel, petrol) identified by an integer index."""

    def __init__(self, number: int, name: str):
        self.number = number
        self.name = name

    # Equality and hashing are needed because Product instances are used
    # as dictionary keys and compared with == throughout the codebase.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.number == other.number

    def __hash__(self) -> int:
        return hash(self.number)

    def __repr__(self) -> str:
        return f"Product({self.number}: {self.name})"


class Location:
    """A physical place on the map (depot, gas station, or parking lot)."""

    def __init__(
        self,
        number: int,
        name: str,
        latitude: float,
        longitude: float,
    ):
        self.number = number
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self) -> str:
        return f"Location({self.name})"


class Node:
    """
    A (location, product) pair — the basic unit of the network.

    location_type is one of: 'Station', 'Depot', 'Parking'.

    Station nodes have finite capacity and a safety stock level.
    Depot nodes have infinite capacity and zero safety stock.
    The single Parking node is the truck home base (start/end of every route).
    """

    def __init__(
        self,
        number: int,
        location_type: str,
        location: Location,
        product: Product,
        capacity: float,
        safety: float,
    ):
        if location_type not in ("Station", "Depot", "Parking"):
            raise ValueError(
                f"location_type must be 'Station', 'Depot', or 'Parking', got '{location_type}'"
            )
        self.number = number
        self.location_type = location_type
        self.location = location
        self.product = product
        self.capacity = capacity   # in thousands of litres (klitres)
        self.safety = safety       # minimum desired inventory level

    def __repr__(self) -> str:
        return f"Node({self.location_type} @ {self.location.name} | {self.product.name})"


class Truck:
    """
    A delivery vehicle with one or more product compartments.

    Parameters
    ----------
    name:
        Identifier string from the input data.
    compartments:
        Number of compartments (tanks).
    compartment_capacities:
        Capacity of each compartment in klitres.
        Must have length equal to ``compartments``.
    """

    def __init__(
        self,
        name: str,
        compartments: int,
        compartment_capacities: List[float],
    ):
        if compartments != len(compartment_capacities):
            raise ValueError(
                f"compartments={compartments} does not match "
                f"len(compartment_capacities)={len(compartment_capacities)}"
            )
        self.name = name
        self.compartments = compartments
        self.compartment_capacities = compartment_capacities

    @property
    def total_capacity(self) -> float:
        """Sum of all compartment capacities."""
        return float(np.sum(self.compartment_capacities))

    def __repr__(self) -> str:
        return (
            f"Truck({self.name}, compartments={self.compartments}, "
            f"total_cap={self.total_capacity:.2f} kl)"
        )
