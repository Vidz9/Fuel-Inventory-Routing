"""
branching/branch_and_price.py
==============================
Branch-and-Price algorithm for the IRP.

The tree is explored best-first (lowest LP bound first) using a min-heap.
Each node runs column generation to solve its LP relaxation.  When a
node's LP solution is fractional in the route-selection variables W,
two children are created by forcing one arc to be mandatory or forbidden
for a particular (truck, day) pair.

After the B&P tree is exhausted (or the time limit hit), a final integer
solve is performed over the full route pool to recover the best integer
solution.

Public API
----------
    BranchingNode      — dataclass representing one tree node
    BranchAndPrice     — the B&P solver class
"""

from __future__ import annotations

import copy
import gc
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PenaltyConfig, ProblemConfig, SolverConfig
from instance import IRPInstance, RoutePool
from solvers.column_gen import CGResult, run_column_generation


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class BranchingNode:
    """
    One node in the branch-and-price tree.

    ``priority`` drives the heap ordering (min-heap = best-first by LP bound).
    All other fields are excluded from comparison so the heap only uses
    the priority.
    """
    priority: float
    level:    int

    # Per-node branching constraints (not compared)
    mandatory_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(
        compare=False, default_factory=dict
    )
    forbidden_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(
        compare=False, default_factory=dict
    )

    lower_bound:  float       = field(compare=False, default=0.0)
    is_integral:  bool        = field(compare=False, default=False)
    cg_result:    object      = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Branch-and-Price
# ---------------------------------------------------------------------------

class BranchAndPrice:
    """
    Branch-and-Price solver.

    The shared route pool lives inside ``instance.routes`` and grows
    monotonically as new columns are discovered across all tree nodes.

    Parameters
    ----------
    instance    : IRPInstance (route pool is mutated in-place during the run)
    penalties   : phi1 / phi2 penalty weights
    problem_cfg : daily driving limit, etc.
    solver_cfg  : time limits, CG tolerance
    env         : silent Gurobi environment
    lower_bound : LP lower bound from a preceding CG run (used to initialise
                  the root node and prune branches)
    """

    def __init__(
        self,
        instance:    IRPInstance,
        penalties:   PenaltyConfig,
        problem_cfg: ProblemConfig,
        solver_cfg:  SolverConfig,
        env,
        lower_bound: float = 0.0,
    ):
        self.instance    = instance
        self.penalties   = penalties
        self.problem_cfg = problem_cfg
        self.solver_cfg  = solver_cfg
        self.env         = env
        self.lower_bound = lower_bound

        self.best_obj:      float            = np.inf
        self.final_mip_gap: float            = np.nan
        self.iteration_log: list             = []   # accumulated across all B&P nodes
        self.best_solution: Optional[Dict]   = None
        self.nodes_heap:    list             = []
        self.nodes_explored: int             = 0

        self.generation_time_bp: float = 0.0
        self.solving_time_bp:    float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_multi_fraction(
        self,
        total_time_limit: float,
        gen_fractions:    list,
        method_start:     float,
        gen_time:         float,
        root_R:           int,
        lp_obj:           float = -np.inf,
        svm_filter_fn           = None,   # callable(instance) -> None
        random_filter_fn        = None,   # callable(instance) -> None
        lr_filter_fn            = None,   # callable(instance) -> None
    ) -> Dict[int | float, tuple]:
        """
        Two-phase B&P with multiple generation checkpoints.

        Phase 1 — tree exploration:
            Explore the B&P tree from right after root CG generation.
            Checkpoints are anchored to method_start so root-solve time does
            not shift their deadlines.  At each checkpoint, record the current
            pool size R but do NOT solve any integer MILP yet.

        Phase 2 — integer solves (all after tree exploration):
            0  (root node): pool = root_R, budget = time_limit - gen_time
            α=0.5:          pool = R_0.5,  budget = (1-0.5) * time_limit
            α=0.7:          pool = R_0.7,  budget = (1-0.7) * time_limit
            Each solve is capped by overall remaining time.

        Returns
        -------
        Dict: key 0 = root result, key frac = BP fraction result.
              Each value is (solution, cg_result, best_obj, gap).
        """
        import copy

        fractions      = sorted(gen_fractions)
        total_deadline = method_start + total_time_limit * 60   # hard wall-clock cap

        # Checkpoint deadlines anchored to method_start (not to now)
        # Deadline = method_start + frac * time_limit (absolute)
        # We subtract gen_time so tree exploration starts immediately after generation
        checkpoints = [
            (frac, method_start + frac * total_time_limit * 60)
            for frac in fractions
        ]

        # ---- Phase 1: tree exploration, snapshot R at each checkpoint ----
        start = time.perf_counter()
        root  = self._make_root_node()
        heapq.heappush(self.nodes_heap, root)

        remaining_cps      = list(checkpoints)
        pool_snapshots     = {}   # frac -> R at that checkpoint
        gen_time_snapshots = {}   # frac -> seconds of tree exploration at that checkpoint

        while True:
            # Snapshot all checkpoints whose deadline has passed
            while remaining_cps and time.perf_counter() >= remaining_cps[0][1]:
                frac_snap, _ = remaining_cps.pop(0)
                pool_snapshots[frac_snap] = self.instance.routes.R
                gen_time_snapshots[frac_snap] = time.perf_counter() - start
                print(f"[B&P] Snapshot α={frac_snap}: R={self.instance.routes.R}  "
                      f"nodes={self.nodes_explored}")

            if not remaining_cps:
                break

            if not self.nodes_heap:
                _t_exhausted = time.perf_counter() - start
                for frac_r, _ in remaining_cps:
                    pool_snapshots[frac_r] = self.instance.routes.R
                    gen_time_snapshots[frac_r] = _t_exhausted
                    print(f"[B&P] Snapshot α={frac_r} (tree exhausted): "
                          f"R={self.instance.routes.R}")
                break

            node = heapq.heappop(self.nodes_heap)
            self._process_node(node, gen_deadline=remaining_cps[0][1])
            self.nodes_explored += 1
            del node
            gc.collect()

        self.generation_time_bp = time.perf_counter() - start
        print(f"[B&P] Tree done. Nodes={self.nodes_explored}  "
              f"Total routes={self.instance.routes.R}  "
              f"Best obj so far={self.best_obj:.4f}")

        # ---- Phase 2: integer solves interleaved with tree exploration ----
        # Each solve runs at its checkpoint and gets budget =
        #   total_deadline - checkpoint_deadline
        # i.e. whatever time remains from that checkpoint to the end of the
        # total budget.  This is the only sequential design that gives each
        # MILP its intended budget regardless of how long prior solves took.
        #
        # Solve order:
        #   - Root: immediately after tree snapshots are done
        #   - Then each fraction in ascending order, each capped at time_left
        #
        # Budget derivation:
        #   checkpoint_time[root] = gen_time (minutes from method_start)
        #   checkpoint_time[frac] = max(gen_time, frac * total_time_limit)
        #     (if gen_time > frac*limit, the checkpoint was already past when
        #      tree started, so effective checkpoint = gen_time)
        #   budget[key] = total_time_limit - checkpoint_time[key]

        self._reset_instance_constraints()
        results = {}

        def _checkpoint_min(key):
            if key == 0:
                return gen_time
            return max(gen_time, key * total_time_limit)

        # gen_time_snapshots[0] = 0 (root pool existed before tree started)
        gen_time_snapshots[0] = 0.0
        solves = (
            [(0, root_R)]
            + [(frac, pool_snapshots.get(frac, self.instance.routes.R))
               for frac in fractions]
        )

        for key, R_snap in solves:
            cp_time   = _checkpoint_min(key)
            budget    = total_time_limit - cp_time          # intended budget (minutes)
            solve_limit = budget   # each MILP gets its full intended budget
            print(f"[B&P] Integer MILP key={key}: pool={R_snap} routes  "
                  f"checkpoint={cp_time:.1f} min  budget={budget:.1f} min  "
                  f"actual={solve_limit:.1f} min")

            orig_R = self.instance.routes.R
            self.instance.routes.R = R_snap

            cg_result = run_column_generation(
                instance=self.instance,
                penalties=self.penalties,
                problem_cfg=self.problem_cfg,
                solver_cfg=SolverConfig(
                    time_limit=solve_limit,
                    **{k: v for k, v in vars(self.solver_cfg).items()
                       if k != "time_limit"},
                ),
                env=self.env,
                lower_bound=lp_obj,
                integer_solution=True,
                only_integer=True,
                tree_level=0,
            )

            self.instance.routes.R = orig_R

            if cg_result.integer_solution is not None:
                if cg_result.integer_obj < self.best_obj:
                    self.best_obj      = cg_result.integer_obj
                    self.best_solution = cg_result.integer_solution

            cp_obj = cg_result.integer_obj if cg_result.integer_solution else self.best_obj
            if not np.isnan(cg_result.obj_bound) and cp_obj < np.inf                     and abs(cp_obj) > 1e-10:
                cp_gap = abs(cp_obj - cg_result.obj_bound) / abs(cp_obj)
            elif not np.isnan(cg_result.mip_gap):
                cp_gap = cg_result.mip_gap
            else:
                cp_gap = np.nan
            self.final_mip_gap = cp_gap

            self.solving_time_bp += cg_result.solving_time
            cp_gen_time = gen_time_snapshots.get(key, self.generation_time_bp)
            results[key] = (copy.copy(self.best_solution), cg_result,
                            cp_obj, cp_gap, R_snap, cp_gen_time)
            print(f"[B&P] key={key} obj={cp_obj:.4f}  gap={cp_gap:.4f}  "
                  f"solve={cg_result.solving_time:.1f}s")

            # Snapshot full feasible_routes ONCE before any filter
            _orig_feasible_full = {vt: list(r)
                                   for vt, r in self.instance.routes.feasible_routes.items()}
            _orig_R_full = self.instance.routes.R

            # ---- SVM-filtered solve with same pool snapshot and time budget ----
            if svm_filter_fn is not None:
                orig_feasible = _orig_feasible_full
                orig_R_full   = self.instance.routes.R
                self.instance.routes.R = R_snap
                svm_filter_fn(self.instance)   # restricts feasible_routes in-place
                filt_cg = run_column_generation(
                    instance=self.instance,
                    penalties=self.penalties,
                    problem_cfg=self.problem_cfg,
                    solver_cfg=SolverConfig(
                        time_limit=solve_limit,
                        **{k_: v_ for k_, v_ in vars(self.solver_cfg).items()
                           if k_ != "time_limit"},
                    ),
                    env=self.env,
                    lower_bound=lp_obj,
                    integer_solution=True,
                    only_integer=True,
                    tree_level=0,
                )
                # Restore full pool
                self.instance.routes.R = _orig_R_full
                for vt_, r_ in _orig_feasible_full.items():
                    self.instance.routes.feasible_routes[vt_] = r_

                filt_obj = filt_cg.integer_obj if filt_cg.integer_solution else np.inf
                if not np.isnan(filt_cg.obj_bound) and filt_obj < np.inf \
                        and abs(filt_obj) > 1e-10:
                    filt_gap = abs(filt_obj - filt_cg.obj_bound) / abs(filt_obj)
                elif not np.isnan(filt_cg.mip_gap):
                    filt_gap = filt_cg.mip_gap
                else:
                    filt_gap = np.nan
                results[(key, "filtered")] = (
                    copy.copy(filt_cg.integer_solution),
                    filt_cg, filt_obj, filt_gap, R_snap, cp_gen_time
                )
                print(f"[B&P filtered] key={key} obj={filt_obj:.4f}  "
                      f"gap={filt_gap:.4f}")

            # ---- Random-filtered solve ----
            if random_filter_fn is not None:
                self.instance.routes.R = R_snap
                for vt_, r_ in _orig_feasible_full.items():
                    self.instance.routes.feasible_routes[vt_] = r_
                random_filter_fn(self.instance)
                rand_cg = run_column_generation(
                    instance=self.instance,
                    penalties=self.penalties,
                    problem_cfg=self.problem_cfg,
                    solver_cfg=SolverConfig(
                        time_limit=solve_limit,
                        **{k_: v_ for k_, v_ in vars(self.solver_cfg).items()
                           if k_ != "time_limit"},
                    ),
                    env=self.env,
                    lower_bound=lp_obj,
                    integer_solution=True,
                    only_integer=True,
                    tree_level=0,
                )
                self.instance.routes.R = _orig_R_full
                for vt_, r_ in _orig_feasible_full.items():
                    self.instance.routes.feasible_routes[vt_] = r_
                rand_obj = rand_cg.integer_obj if rand_cg.integer_solution else np.inf
                if not np.isnan(rand_cg.obj_bound) and rand_obj < np.inf \
                        and abs(rand_obj) > 1e-10:
                    rand_gap = abs(rand_obj - rand_cg.obj_bound) / abs(rand_obj)
                elif not np.isnan(rand_cg.mip_gap):
                    rand_gap = rand_cg.mip_gap
                else:
                    rand_gap = np.nan
                results[(key, "random_filtered")] = (
                    copy.copy(rand_cg.integer_solution),
                    rand_cg, rand_obj, rand_gap, R_snap, cp_gen_time
                )
                print(f"[B&P random_filtered] key={key} obj={rand_obj:.4f}  "
                      f"gap={rand_gap:.4f}")

            # ---- LR-filtered solve ----
            if lr_filter_fn is not None:
                self.instance.routes.R = R_snap
                for vt_, r_ in _orig_feasible_full.items():
                    self.instance.routes.feasible_routes[vt_] = r_
                lr_filter_fn(self.instance)
                lr_cg = run_column_generation(
                    instance=self.instance,
                    penalties=self.penalties,
                    problem_cfg=self.problem_cfg,
                    solver_cfg=SolverConfig(
                        time_limit=solve_limit,
                        **{k_: v_ for k_, v_ in vars(self.solver_cfg).items()
                           if k_ != "time_limit"},
                    ),
                    env=self.env,
                    lower_bound=lp_obj,
                    integer_solution=True,
                    only_integer=True,
                    tree_level=0,
                )
                self.instance.routes.R = _orig_R_full
                for vt_, r_ in _orig_feasible_full.items():
                    self.instance.routes.feasible_routes[vt_] = r_
                lr_obj = lr_cg.integer_obj if lr_cg.integer_solution else np.inf
                if not np.isnan(lr_cg.obj_bound) and lr_obj < np.inf \
                        and abs(lr_obj) > 1e-10:
                    lr_gap = abs(lr_obj - lr_cg.obj_bound) / abs(lr_obj)
                elif not np.isnan(lr_cg.mip_gap):
                    lr_gap = lr_cg.mip_gap
                else:
                    lr_gap = np.nan
                results[(key, "lr_filtered")] = (
                    copy.copy(lr_cg.integer_solution),
                    lr_cg, lr_obj, lr_gap, R_snap, cp_gen_time
                )
                print(f"[B&P lr_filtered] key={key} obj={lr_obj:.4f}  "
                      f"gap={lr_gap:.4f}")

        return results

    def run(self, *args, **kwargs):
        """Deprecated — use run_multi_fraction() directly."""
        raise NotImplementedError(
            "BranchAndPrice.run() is deprecated. "
            "Call run_multi_fraction() from run_instance() instead."
        )

    # ------------------------------------------------------------------
    # Node processing
    # ------------------------------------------------------------------

    def _make_root_node(self) -> BranchingNode:
        """Create the root node with no branching constraints."""
        V = self.instance.data.V
        T = len(next(iter(self.instance.demand.values())))
        mandatory: Dict = {}
        forbidden:  Dict = {}
        for v in range(V):
            for t in range(T):
                mandatory[(v, t)] = []
                forbidden[(v, t)]  = []
        return BranchingNode(
            priority=0.0,
            level=0,
            mandatory_edges=mandatory,
            forbidden_edges=forbidden,
            lower_bound=self.lower_bound,
        )

    def _apply_node_constraints(self, node: BranchingNode) -> None:
        """
        Push this node's branching constraints into instance.routes so that
        the CG subproblem and RMP use them correctly.
        """
        routes = self.instance.routes
        V      = self.instance.data.V
        T      = len(next(iter(self.instance.demand.values())))

        routes.mandatory_edges = node.mandatory_edges
        routes.forbidden_edges  = node.forbidden_edges

        # Recompute feasible routes: start from all routes, then intersect/
        # remove based on mandatory and forbidden edge constraints.
        for v in range(V):
            for t in range(T):
                feasible = set(range(routes.R))

                for edge in node.mandatory_edges[(v, t)]:
                    if edge in routes.edge_routes:
                        feasible &= set(routes.edge_routes[edge])
                    else:
                        feasible = set()

                for edge in node.forbidden_edges[(v, t)]:
                    if edge in routes.edge_routes:
                        feasible -= set(routes.edge_routes[edge])

                # Always keep the dummy no-op route (id=0)
                feasible.add(0)
                routes.feasible_routes[(v, t)] = sorted(feasible)

    def _reset_instance_constraints(self) -> None:
        """Remove branching constraints so the final solve uses all routes."""
        routes = self.instance.routes
        V      = self.instance.data.V
        T      = len(next(iter(self.instance.demand.values())))
        for v in range(V):
            for t in range(T):
                routes.mandatory_edges[(v, t)] = []
                routes.forbidden_edges[(v, t)]  = []
                routes.feasible_routes[(v, t)]  = list(range(routes.R))

    def _process_node(self, node: BranchingNode,
                      gen_deadline: float = float("inf")) -> None:
        """
        Solve the LP relaxation at this B&P node via column generation,
        then either update the incumbent or branch.

        gen_deadline is a wall-clock time (time.perf_counter()) at which
        column generation should stop, even mid-node.  The per-node CG
        time limit is set to the remaining time until that deadline, so
        the node terminates as soon as generation time expires.
        """
        self._apply_node_constraints(node)

        remaining_min = max((gen_deadline - time.perf_counter()) / 60, 0.0)
        cg_result: CGResult = run_column_generation(
            instance=self.instance,
            penalties=self.penalties,
            problem_cfg=self.problem_cfg,
            solver_cfg=SolverConfig(
                time_limit=remaining_min,   # also limits individual subproblem solves
                **{k: v for k, v in vars(self.solver_cfg).items() if k != "time_limit"},
            ),
            env=self.env,
            lower_bound=node.lower_bound,
            integer_solution=False,
            only_integer=False,
            tree_level=node.level,
            deadline=gen_deadline,   # CG loop breaks as soon as this is reached
        )

        node.cg_result  = cg_result
        node.lower_bound = cg_result.lp_obj
        # Accumulate coalition log from this node, tagging each entry with node id
        if cg_result.iteration_log:
            for _entry in cg_result.iteration_log:
                self.iteration_log.append({**_entry, "node_id": self.nodes_explored})

        if cg_result.lp_obj == np.inf:
            # Infeasible node — prune
            return

        # Record route usage in the OPTIMAL LP solution of this node.
        # Called exactly once per node, on the final (converged) RMP solution —
        # not after every intermediate RMP solve inside the CG loop.
        # This lets us track how often each route appears in node optima,
        # which will later be used to prune rarely-used routes before the
        # final integer solve.
        if cg_result.lp_solution is not None:
            _V = self.instance.data.V
            _T = len(next(iter(self.instance.demand.values())))
            self.instance.routes.update_rmp_stats(
                cg_result.lp_solution.W_sol, _V, _T
            )

        # Update incumbent if CG found a better integer solution
        if (cg_result.integer_solution is not None
                and cg_result.integer_obj < self.best_obj):
            self.best_obj      = cg_result.integer_obj
            self.best_solution = cg_result.integer_solution

        # Prune by bound
        if node.lower_bound >= self.best_obj:
            return

        # Check integrality of LP solution
        routes_selected = cg_result.lp_solution.routes_selected   # (V, T, R)
        is_integral = all(
            np.count_nonzero(routes_selected[v, t]) <= 1
            for v in range(self.instance.data.V)
            for t in range(len(next(iter(self.instance.demand.values()))))
        )

        if is_integral:
            if node.lower_bound < self.best_obj:
                self.best_obj = node.lower_bound
        else:
            for child in self._branch(node, routes_selected):
                heapq.heappush(self.nodes_heap, child)

        print(f"[B&P] Heap size: {len(self.nodes_heap)}  |  Best obj: {self.best_obj:.4f}")

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def _branch(
        self,
        node:             BranchingNode,
        routes_selected:  np.ndarray,
    ) -> List[BranchingNode]:
        """
        Create two children by picking a random fractional (truck, day, arc)
        and branching on whether that arc is mandatory or forbidden.
        """
        routes  = self.instance.routes
        V       = self.instance.data.V
        T       = len(next(iter(self.instance.demand.values())))

        # Find a fractional (truck, day) and a random arc in its LP route
        while True:
            v = np.random.randint(0, V)
            t = np.random.randint(0, T)

            row = routes_selected[v, t]
            if row.sum() < 1e-9:
                continue

            # Sample a route proportional to its LP weight
            probs = row / row.sum()
            chosen_r = int(np.random.choice(len(row), p=probs))

            arc_indices = np.argwhere(routes.route_edges[chosen_r] == 1)
            if len(arc_indices) == 0:
                continue

            idx = arc_indices[np.random.choice(len(arc_indices))]
            arc = (int(idx[0]), int(idx[1]))

            if (arc not in node.mandatory_edges[(v, t)]
                    and arc not in node.forbidden_edges[(v, t)]):
                break

        print(f"[B&P] Branching on arc {arc} for (truck={v}, day={t})")

        level = node.level + 1

        # Child 1: arc is mandatory for (v, t)
        mand_1 = copy.deepcopy(node.mandatory_edges)
        mand_1[(v, t)].append(arc)
        forb_1 = copy.deepcopy(node.forbidden_edges)

        # Child 2: arc is forbidden for (v, t)
        mand_2 = copy.deepcopy(node.mandatory_edges)
        forb_2 = copy.deepcopy(node.forbidden_edges)
        forb_2[(v, t)].append(arc)

        child1 = BranchingNode(
            priority=node.lower_bound,
            level=level,
            mandatory_edges=mand_1,
            forbidden_edges=forb_1,
            lower_bound=node.lower_bound,
        )
        child2 = BranchingNode(
            priority=node.lower_bound,
            level=level,
            mandatory_edges=mand_2,
            forbidden_edges=forb_2,
            lower_bound=node.lower_bound,
        )
        return [child1, child2]