"""Network simplex implementation for minimum-cost flow."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from .basis import TreeBasis
from .data import FlowResult, NetworkProblem, ProgressCallback, ProgressInfo, SolverOptions
from .exceptions import (
    InvalidProblemError,
    SolverConfigurationError,
    UnboundedProblemError,
)

DEVEX_WEIGHT_MIN = 1e-12  # Prevent division by zero or runaway weights in Devex pricing.
DEVEX_WEIGHT_MAX = 1e12  # Cap the Devex weight to avoid catastrophic scaling.
PERTURB_EPS_BASE = 1e-10  # Base epsilon for lexicographic cost perturbation.
PERTURB_GROWTH = 1.00001  # Slight growth per arc to break ties deterministically.


@dataclass
class ArcState:
    tail: int
    head: int
    cost: float
    lower: float
    upper: float
    flow: float
    in_tree: bool
    artificial: bool
    key: tuple[str, str]
    shift: float = 0.0

    def forward_residual(self) -> float:
        if math.isinf(self.upper):
            return math.inf
        return self.upper - self.flow

    def backward_residual(self) -> float:
        return self.flow - self.lower


class NetworkSimplex:
    """Network simplex solver for the minimum-cost flow problem.

    This class implements the network simplex algorithm, a specialized version of the
    simplex method optimized for network flow problems. It maintains a spanning tree
    basis and iteratively improves the solution through pivots.

    The algorithm operates in two phases:
    - Phase 1: Find an initial feasible solution (minimize artificial arc usage)
    - Phase 2: Optimize from the feasible basis (minimize actual cost)

    Implementation Details:
        - Uses a spanning tree basis (|N|-1 basic arcs)
        - Maintains node potentials (dual variables) for reduced cost computation
        - Supports Devex and Dantzig pricing strategies for pivot selection
        - Uses Forrest-Tomlin updates for efficient basis factorization
        - Applies cost perturbation for degeneracy handling

    Attributes:
        problem: The NetworkProblem instance to solve.
        options: Solver configuration (pricing strategy, tolerances, etc.).
        node_ids: List of node IDs (including artificial root).
        arcs: Internal arc representation with flow state.
        basis: TreeBasis managing the spanning tree structure.

    See Also:
        - solve_min_cost_flow(): Public API wrapper
        - docs/algorithm.md: Detailed algorithm explanation
        - TreeBasis: Basis management and factorization

    Note:
        This class is internal to the solver. Use solve_min_cost_flow() instead
        of instantiating this class directly.
    """

    ROOT_NODE = "__network_simplex_root__"

    def __init__(self, problem: NetworkProblem, options: SolverOptions | None = None):
        self.problem = problem
        self.options = options if options is not None else SolverOptions()
        self.logger = logging.getLogger(__name__)
        self.ft_rebuilds = 0
        self.ft_updates_since_rebuild = 0
        # Store node ids in a dense index space so core routines can use list lookups.
        self.node_ids: list[str] = [self.ROOT_NODE] + sorted(problem.nodes.keys())
        self.node_index: dict[str, int] = {
            node_id: idx for idx, node_id in enumerate(self.node_ids)
        }
        self.root = 0
        self.tolerance = self.options.tolerance

        self.node_supply: list[float] = self._initial_supplies()
        self.arcs: list[ArcState] = []
        self._build_arcs()
        self.actual_arc_count = len(self.arcs)
        max_cost = max((abs(arc.cost) for arc in self.arcs), default=1.0)
        # Artificial arcs receive a large penalty so Phase 1 prefers genuine problem arcs.
        self.penalty_cost = max_cost * (len(self.node_ids) + 1)

        self.node_count = len(self.node_ids)
        self.basis = TreeBasis(self.node_count, self.root, self.tolerance)
        self.tree_adj: list[list[int]] = [[] for _ in range(self.node_count)]
        self.pricing_block = 0
        if self.options.block_size is not None:
            self.block_size = self.options.block_size
        else:
            self.block_size = max(1, self.actual_arc_count // 8)
        self.devex_weights: list[float] = [1.0] * len(self.arcs)
        self.original_costs = [arc.cost for arc in self.arcs]
        self.perturbed_costs = [arc.cost for arc in self.arcs]
        self._apply_cost_perturbation()
        self._initialize_tree()
        self._rebuild_tree_structure()
        self._reset_devex_weights()

    def _initial_supplies(self) -> list[float]:
        supplies = [0.0] * len(self.node_ids)
        for idx, node_id in enumerate(self.node_ids[1:], start=1):
            supplies[idx] = self.problem.nodes[node_id].supply
        total_supply = sum(supplies)
        if abs(total_supply) > self.tolerance:
            raise InvalidProblemError(
                f"Supplies do not balance after lower-bound adjustment: total supply "
                f"{total_supply:.6f} exceeds tolerance {self.tolerance}. The sum of all node "
                f"supplies must equal zero for a valid flow problem."
            )
        return supplies

    def _build_arcs(self) -> None:
        node_supply_adjusted = self.node_supply[:]
        expanded_arcs = list(self.problem.undirected_expansion())
        expanded_arcs.sort(key=lambda a: (a.tail, a.head))

        # Translate each problem arc into an internal state while accounting for lower bounds.
        for arc in expanded_arcs:
            tail_idx = self.node_index[arc.tail]
            head_idx = self.node_index[arc.head]
            lower = arc.lower
            capacity = arc.capacity
            if capacity is None:
                upper = math.inf
            else:
                upper = float(capacity) - lower
                if upper < -self.tolerance:
                    raise InvalidProblemError(
                        f"Arc capacity ({capacity}) is less than lower bound ({lower}) "
                        f"for arc {arc.tail} -> {arc.head}. Capacity must be >= lower bound."
                    )
                upper = max(0.0, upper)
            if lower:
                node_supply_adjusted[tail_idx] -= lower
                node_supply_adjusted[head_idx] += lower
            self.arcs.append(
                ArcState(
                    tail=tail_idx,
                    head=head_idx,
                    cost=arc.cost,
                    lower=0.0,
                    upper=upper,
                    flow=0.0,
                    in_tree=False,
                    artificial=False,
                    key=(arc.tail, arc.head),
                    shift=lower,
                )
            )

        # Update supplies after lower bound adjustments
        self.node_supply = node_supply_adjusted

    def _initialize_tree(self) -> None:
        """Create an initial spanning tree solution using artificial root arcs."""
        for node_idx, supply in enumerate(self.node_supply):
            if node_idx == self.root:
                continue
            node_id = self.node_ids[node_idx]
            if abs(supply) <= self.tolerance:
                # Zero-supply node: attach via penalized root->node arc with zero flow.
                arc = ArcState(
                    tail=self.root,
                    head=node_idx,
                    cost=self.penalty_cost,
                    lower=0.0,
                    upper=math.inf,
                    flow=0.0,
                    in_tree=True,
                    artificial=True,
                    key=(self.ROOT_NODE, node_id),
                )
            elif supply > 0:
                arc = ArcState(
                    tail=node_idx,
                    head=self.root,
                    cost=self.penalty_cost,
                    lower=0.0,
                    upper=supply,
                    flow=supply,
                    in_tree=True,
                    artificial=True,
                    key=(node_id, self.ROOT_NODE),
                )
            else:
                demand = -supply
                arc = ArcState(
                    tail=self.root,
                    head=node_idx,
                    cost=self.penalty_cost,
                    lower=0.0,
                    upper=demand,
                    flow=demand,
                    in_tree=True,
                    artificial=True,
                    key=(self.ROOT_NODE, node_id),
                )
            arc_idx = len(self.arcs)
            self.arcs.append(arc)
            self.original_costs.append(arc.cost)
            self.perturbed_costs.append(arc.cost)
            self.devex_weights.append(1.0)
            self.tree_adj[self.root].append(arc_idx)
            self.tree_adj[node_idx].append(arc_idx)

        # Ensure every node (except root) has at least one tree arc; add zero-flow arc if needed.
        for node_idx in range(1, self.node_count):
            if not self.tree_adj[node_idx]:
                arc = ArcState(
                    tail=self.root,
                    head=node_idx,
                    cost=self.penalty_cost,
                    lower=0.0,
                    upper=math.inf,
                    flow=0.0,
                    in_tree=True,
                    artificial=True,
                    key=(self.ROOT_NODE, self.node_ids[node_idx]),
                )
                arc_idx = len(self.arcs)
                self.arcs.append(arc)
                self.original_costs.append(arc.cost)
                self.perturbed_costs.append(arc.cost)
                self.devex_weights.append(1.0)
                self.tree_adj[self.root].append(arc_idx)
                self.tree_adj[node_idx].append(arc_idx)

    def _rebuild_tree_structure(self) -> None:
        """Recompute parent pointers and potentials based on active tree arcs."""
        self.basis.rebuild(self.tree_adj, self.arcs)

    def _find_entering_arc(self, allow_zero: bool) -> tuple[int, int] | None:
        """Return (arc_idx, direction) for entering arc, where direction is +1 or -1."""
        if self.options.pricing_strategy == "devex":
            return self._find_entering_arc_devex(allow_zero)
        elif self.options.pricing_strategy == "dantzig":
            return self._find_entering_arc_dantzig(allow_zero)
        else:
            raise SolverConfigurationError(
                f"Unknown pricing strategy '{self.options.pricing_strategy}'. "
                f"Valid options: 'devex', 'dantzig'."
            )

    def _update_devex_weight(self, arc_idx: int, arc: ArcState) -> float:
        """Update and return the Devex weight for the given arc."""
        weight = max(self.devex_weights[arc_idx], DEVEX_WEIGHT_MIN)
        projection = self.basis.project_column(arc)
        if projection is not None:
            # Recompute Devex weight using the latest basis solve to stabilise pricing.
            weight = float(np.dot(projection, projection))
            if not math.isfinite(weight) or weight <= DEVEX_WEIGHT_MIN:
                weight = DEVEX_WEIGHT_MIN
            elif weight > DEVEX_WEIGHT_MAX:
                weight = DEVEX_WEIGHT_MAX
            self.devex_weights[arc_idx] = weight
        return weight

    def _is_better_candidate(
        self, merit: float, idx: int, best_merit: float, best: tuple[int, int] | None
    ) -> bool:
        """Check if current candidate is better than the best found so far."""
        better = merit > best_merit + self.tolerance
        tie = not better and abs(merit - best_merit) <= self.tolerance
        return better or (tie and (best is None or idx < best[0]))

    def _find_entering_arc_devex(self, allow_zero: bool) -> tuple[int, int] | None:
        """Devex pricing: block-based search with normalized reduced costs."""
        zero_candidates: list[tuple[int, int]] = []
        best: tuple[int, int] | None = None
        best_merit = -math.inf
        block_count = max(1, (self.actual_arc_count + self.block_size - 1) // self.block_size)

        for _ in range(block_count):
            start = self.pricing_block * self.block_size
            if start >= self.actual_arc_count:
                self.pricing_block = 0
                start = 0
            end = min(start + self.block_size, self.actual_arc_count)
            best_merit = -math.inf
            best = None
            zero_candidates = []

            for idx in range(start, end):
                arc = self.arcs[idx]
                if arc.in_tree or arc.artificial:
                    continue
                rc = arc.cost + self.basis.potential[arc.tail] - self.basis.potential[arc.head]
                forward_res = arc.forward_residual()
                backward_res = arc.backward_residual()

                # Check forward direction
                if forward_res > self.tolerance and rc < -self.tolerance:
                    weight = self._update_devex_weight(idx, arc)
                    merit = (rc * rc) / weight
                    if self._is_better_candidate(merit, idx, best_merit, best):
                        best_merit = merit
                        best = (idx, 1)
                    continue

                # Check backward direction
                if backward_res > self.tolerance and rc > self.tolerance:
                    weight = self._update_devex_weight(idx, arc)
                    merit = (rc * rc) / weight
                    if self._is_better_candidate(merit, idx, best_merit, best):
                        best_merit = merit
                        best = (idx, -1)
                    continue

                # Collect zero-reduced-cost candidates if allowed
                if allow_zero and forward_res > self.tolerance and abs(rc) <= self.tolerance:
                    zero_candidates.append((idx, 1))
                elif allow_zero and backward_res > self.tolerance and abs(rc) <= self.tolerance:
                    zero_candidates.append((idx, -1))

            if best is not None:
                return best
            if allow_zero and zero_candidates:
                self.pricing_block = (self.pricing_block + 1) % block_count
                return zero_candidates[0]

            self.pricing_block = (self.pricing_block + 1) % block_count

        return None

    def _find_entering_arc_dantzig(self, allow_zero: bool) -> tuple[int, int] | None:
        """Dantzig pricing: simple first-eligible arc with most negative reduced cost."""
        best: tuple[int, int] | None = None
        best_rc = 0.0

        for idx in range(self.actual_arc_count):
            arc = self.arcs[idx]
            if arc.in_tree or arc.artificial:
                continue
            rc = arc.cost + self.basis.potential[arc.tail] - self.basis.potential[arc.head]
            forward_res = arc.forward_residual()
            backward_res = arc.backward_residual()

            if forward_res > self.tolerance and rc < -self.tolerance:
                if best is None or rc < best_rc:
                    best = (idx, 1)
                    best_rc = rc
            elif backward_res > self.tolerance and rc > self.tolerance:
                if best is None or -rc < best_rc:
                    best = (idx, -1)
                    best_rc = -rc
            elif (
                allow_zero
                and forward_res > self.tolerance
                and abs(rc) <= self.tolerance
                and best is None
            ):
                best = (idx, 1)
            elif (
                allow_zero
                and backward_res > self.tolerance
                and abs(rc) <= self.tolerance
                and best is None
            ):
                best = (idx, -1)

        return best

    def _update_tree_sets(self) -> None:
        self.tree_adj = [[] for _ in range(self.node_count)]
        for idx, arc in enumerate(self.arcs):
            if arc.in_tree:
                self.tree_adj[arc.tail].append(idx)
                self.tree_adj[arc.head].append(idx)

    def _run_simplex_iterations(
        self,
        max_iterations: int,
        allow_zero: bool,
        phase_one: bool = False,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 100,
        phase: int = 1,
        total_iterations_offset: int = 0,
        start_time: float | None = None,
    ) -> int:
        iterations = 0
        while iterations < max_iterations:
            entering = self._find_entering_arc(allow_zero=allow_zero)
            if entering is None:
                break
            arc_idx, direction = entering
            self._pivot(arc_idx, direction)
            iterations += 1

            # Call progress callback at specified interval
            if progress_callback is not None and iterations % progress_interval == 0:
                objective_estimate = self._compute_objective_estimate()
                elapsed = time.time() - start_time if start_time is not None else 0.0
                progress_info = ProgressInfo(
                    iteration=total_iterations_offset + iterations,
                    max_iterations=total_iterations_offset + max_iterations,
                    phase=phase,
                    phase_iterations=iterations,
                    objective_estimate=objective_estimate,
                    elapsed_time=elapsed,
                )
                progress_callback(progress_info)

            if phase_one and not any(
                arc.artificial and arc.flow > self.tolerance for arc in self.arcs
            ):
                # All artificial arcs are zero again, so Phase 1 is complete.
                break
        return iterations

    def _apply_phase_costs(self, phase: int) -> None:
        if phase == 1:
            for idx in range(self.actual_arc_count):
                self.arcs[idx].cost = self.perturbed_costs[idx] - 1.0 - 1e-6 * idx
        elif phase == 2:
            for idx, arc in enumerate(self.arcs):
                arc.cost = self.perturbed_costs[idx]
        else:
            raise SolverConfigurationError(f"Invalid phase {phase}. Phase must be 1 or 2.")

    def _pivot(self, arc_idx: int, direction: int) -> None:
        entering = self.arcs[arc_idx]
        tail = entering.tail if direction == 1 else entering.head
        head = entering.head if direction == 1 else entering.tail

        cycle = self.basis.collect_cycle(self.tree_adj, self.arcs, tail, head)
        cycle.append((arc_idx, direction))

        theta = math.inf
        leaving_idx = arc_idx
        best_residual = -math.inf

        for idx, sign in cycle:
            arc = self.arcs[idx]
            if idx == arc_idx:
                # The entering arc was appended to the cycle for flow updates only.
                continue
            if sign == 1:
                residual = arc.forward_residual()
                if residual < theta - self.tolerance:
                    theta = residual
                    leaving_idx = idx
                    best_residual = residual
                elif abs(residual - theta) <= self.tolerance:
                    if residual > best_residual + self.tolerance or (
                        abs(residual - best_residual) <= self.tolerance and idx < leaving_idx
                    ):
                        leaving_idx = idx
                        best_residual = residual
            else:
                residual = arc.backward_residual()
                if residual < theta - self.tolerance:
                    theta = residual
                    leaving_idx = idx
                    best_residual = residual
                elif abs(residual - theta) <= self.tolerance:
                    if residual > best_residual + self.tolerance or (
                        abs(residual - best_residual) <= self.tolerance and idx < leaving_idx
                    ):
                        leaving_idx = idx
                        best_residual = residual

        if theta is math.inf:
            # Compute reduced cost for diagnostic information
            entering_rc = (
                entering.cost
                + self.basis.potential[entering.tail]
                - self.basis.potential[entering.head]
            )
            if direction == -1:
                entering_rc = -entering_rc
            raise UnboundedProblemError(
                "Unbounded problem detected: entering arc can increase indefinitely without "
                "hitting any capacity constraint. This indicates a negative-cost cycle with "
                "infinite capacity.",
                entering_arc=entering.key,
                reduced_cost=entering_rc,
            )
        theta = max(0.0, theta)

        for idx, sign in cycle:
            arc = self.arcs[idx]
            arc.flow += sign * theta
            if arc.flow < arc.lower - self.tolerance:
                arc.flow = arc.lower
            if not math.isinf(arc.upper) and arc.flow > arc.upper + self.tolerance:
                arc.flow = arc.upper

        entering.in_tree = True
        projection = self.basis.project_column(entering)
        if projection is not None:
            entering_weight = float(np.dot(projection, projection))
            if not math.isfinite(entering_weight) or entering_weight <= DEVEX_WEIGHT_MIN:
                entering_weight = DEVEX_WEIGHT_MIN
            elif entering_weight > DEVEX_WEIGHT_MAX:
                entering_weight = DEVEX_WEIGHT_MAX
        else:
            entering_weight = 1.0
            for idx, _ in cycle:
                if idx == arc_idx:
                    continue
                entering_weight += self.devex_weights[idx]
        self.devex_weights[arc_idx] = max(DEVEX_WEIGHT_MIN, entering_weight)
        if leaving_idx == arc_idx:
            entering.in_tree = False
            # Degenerate pivot: tree unchanged but flows adjusted.
            return
        leaving_arc = self.arcs[leaving_idx]
        leaving_arc.in_tree = False
        self.devex_weights[leaving_idx] = 1.0

        # Check if we've hit the FT update limit and need a full rebuild
        force_rebuild = self.ft_updates_since_rebuild >= self.options.ft_update_limit

        basis_updated = self.basis.replace_arc(leaving_idx, arc_idx, self.arcs, self.tolerance)
        self._update_tree_sets()

        if force_rebuild:
            # Periodic rebuild to maintain numerical stability
            self.basis.rebuild(self.tree_adj, self.arcs, build_numeric=True)
            self.ft_rebuilds += 1
            self.ft_updates_since_rebuild = 0
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "FT update limit reached; rebuilding basis",
                    extra={
                        "leaving_idx": leaving_idx,
                        "entering_idx": arc_idx,
                        "ft_updates": self.options.ft_update_limit,
                    },
                )
            self._reset_devex_weights()
        elif not basis_updated:
            # FT update failed: refresh the numeric basis and reset Devex heuristics.
            self.basis.rebuild(self.tree_adj, self.arcs, build_numeric=True)
            self.ft_rebuilds += 1
            self.ft_updates_since_rebuild = 0
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Forrest–Tomlin update failed; rebuilding basis",
                    extra={"leaving_idx": leaving_idx, "entering_idx": arc_idx},
                )
            self._reset_devex_weights()
        else:
            # Successful FT update
            self.basis.rebuild(self.tree_adj, self.arcs, build_numeric=False)
            self.ft_updates_since_rebuild += 1

    def _apply_cost_perturbation(self) -> None:
        base_eps = PERTURB_EPS_BASE
        factor = 1.0
        for idx in range(self.actual_arc_count):
            perturb = base_eps * factor
            self.perturbed_costs[idx] = self.original_costs[idx] + perturb
            self.arcs[idx].cost = self.perturbed_costs[idx]
            factor *= PERTURB_GROWTH
        for idx in range(self.actual_arc_count, len(self.arcs)):
            self.perturbed_costs[idx] = self.original_costs[idx]

    def solve(
        self,
        max_iterations: int | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 100,
    ) -> FlowResult:
        """Solve the minimum-cost flow problem.

        Args:
            max_iterations: Maximum number of simplex iterations. If None, uses value from
                           SolverOptions (default: max(100, 5*num_arcs)).
            progress_callback: Optional callback function to receive progress updates.
            progress_interval: Number of iterations between progress callbacks (default: 100).

        Returns:
            FlowResult containing solution, dual values, and solver statistics.
        """
        if max_iterations is None:
            if self.options.max_iterations is not None:
                max_iterations = self.options.max_iterations
            else:
                max_iterations = max(100, 5 * len(self.arcs))

        total_iterations = 0
        start_time = time.time()

        # Phase 1: find feasible flow minimizing artificial usage.
        self._apply_phase_costs(phase=1)
        self._rebuild_tree_structure()
        iters = self._run_simplex_iterations(
            max_iterations,
            allow_zero=True,
            phase_one=True,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            phase=1,
            total_iterations_offset=0,
            start_time=start_time,
        )
        total_iterations += iters

        infeasible = any(arc.artificial and arc.flow > self.tolerance for arc in self.arcs)
        if infeasible:
            if total_iterations >= max_iterations:
                return FlowResult(
                    objective=0.0,
                    flows={},
                    status="iteration_limit",
                    iterations=total_iterations,
                    duals={},
                )
            return FlowResult(
                objective=0.0,
                flows={},
                status="infeasible",
                iterations=total_iterations,
                duals={},
            )

        remaining = max(0, max_iterations - total_iterations)
        # Phase 2 restores original costs and seeks an optimal solution given the feasible basis.
        self._apply_phase_costs(phase=2)
        self._rebuild_tree_structure()
        iters = self._run_simplex_iterations(
            remaining,
            allow_zero=False,
            phase_one=False,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            phase=2,
            total_iterations_offset=total_iterations,
            start_time=start_time,
        )
        total_iterations += iters

        status = "optimal" if total_iterations < max_iterations else "iteration_limit"

        flows: dict[tuple[str, str], float] = {}
        objective = 0.0
        for idx, arc in enumerate(self.arcs):
            if arc.artificial:
                continue
            flow_value = arc.flow + arc.shift
            if arc.key in flows:
                flows[arc.key] += flow_value
            else:
                flows[arc.key] = flow_value
            objective += flow_value * self.original_costs[idx]

        # Remove near-zero noise
        for key, value in list(flows.items()):
            if abs(value) <= self.tolerance:
                flows.pop(key)
            else:
                flows[key] = float(round(value, 12))

        # Extract dual values (node potentials) for sensitivity analysis
        # Skip the root node (index 0) as it's artificial
        duals: dict[str, float] = {}
        for idx in range(1, len(self.node_ids)):
            node_id = self.node_ids[idx]
            duals[node_id] = float(round(self.basis.potential[idx], 12))

        return FlowResult(
            objective=float(round(objective, 12)),
            flows=flows,
            status=status,
            iterations=total_iterations,
            duals=duals,
        )

    def _reset_devex_weights(self) -> None:
        for idx in range(len(self.devex_weights)):
            self.devex_weights[idx] = 1.0

    def _compute_objective_estimate(self) -> float:
        """Compute current objective value estimate based on current flows and costs."""
        objective = 0.0
        for idx, arc in enumerate(self.arcs):
            if arc.artificial:
                continue
            flow_value = arc.flow + arc.shift
            # Use original costs for meaningful estimate
            objective += flow_value * self.original_costs[idx]
        return objective
