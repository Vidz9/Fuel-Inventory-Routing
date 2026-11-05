"""
config.py
=========
Global solver and problem parameters.

All constants that were previously scattered as module-level variables in
preprocessing.py live here. Import this module wherever you need them:

    from config import SolverConfig, ProblemConfig

You can also override values at runtime before passing them to solvers:

    cfg = SolverConfig()
    cfg.time_limit = 30
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ProblemConfig:
    """
    Parameters that define the structure of the planning problem.

    Attributes
    ----------
    T               : planning horizon in days
    daily_driving_limit : maximum hours a truck may drive in one day
    one_compartment : if True, treat each truck as a single-compartment vehicle
                      (all tanks merged into one)
    M               : big-M constant for MILP formulations
    """
    T: int                    = 5
    daily_driving_limit: float = 5.0
    one_compartment: bool      = True
    M: float                   = 10_000.0


@dataclass
class PenaltyConfig:
    """
    Penalty weights used in the soft-constraint objective.

    Attributes
    ----------
    phi1 : penalty per unit of inventory below the safety stock level
    phi2 : penalty per unit of inventory below zero (stock-out)

    phi2 should be much larger than phi1 to make stock-outs very costly
    relative to merely falling below the safety threshold.
    """
    phi1: float = 100.0
    phi2: float = 10_000.0


@dataclass
class SolverConfig:
    """
    Gurobi and column-generation parameters.

    Attributes
    ----------
    time_limit          : wall-clock time limit for the main solve (minutes)
    subproblem_time_limit: time limit for each pricing subproblem (seconds)
    subproblem_early_stop_time : stop the subproblem early if a negative
                                 reduced cost is found after this many seconds
    subproblem_early_stop_threshold : reduced-cost threshold for early stop
    cg_optimality_tol   : column-generation convergence tolerance
                          (stop when best reduced cost >= -tol)
    record_times_milp   : checkpoints (seconds) at which the MILP objective
                          is recorded during branch-and-bound
    record_times_bp     : checkpoints (seconds) at which the branch-and-price
                          objective is recorded
    """
    time_limit: float                    = 10.0        # minutes
    subproblem_time_limit: float         = 10.0        # seconds
    subproblem_early_stop_time: float    = 1.0         # seconds
    subproblem_early_stop_threshold: float = -0.01
    cg_optimality_tol: float             = 1e-4
    n_stochastic_samples: int            = 5    # stochastic subproblem attempts per solve

    record_times_milp: list = field(
        default_factory=lambda: [60, 120, 300, 600]    # seconds
    )
    record_times_bp: list = field(
        default_factory=lambda: [10, 30, 60, 180, 300] # seconds
    )


@dataclass
class DataConfig:
    """
    File-system settings for input data.

    Attributes
    ----------
    data_dir     : path to the folder containing 'Input data <size>.xlsx' files
    network_size : which instance size to load ('small', 'medium', or 'large')
    """
    data_dir: str     = "Data"
    network_size: str = "small"

    def __post_init__(self):
        valid = {"small", "medium", "large"}
        if self.network_size not in valid:
            raise ValueError(
                f"network_size must be one of {valid}, got '{self.network_size}'"
            )