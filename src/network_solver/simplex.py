"""Network simplex implementation for minimum-cost flow."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .basis import TreeBasis
from .data import Basis, FlowResult, NetworkProblem, ProgressCallback, ProgressInfo, SolverOptions
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
        self.options = options if options is not None else SolverOptions()
        self.logger = logging.getLogger(__name__)

        # Apply automatic scaling if enabled
        self.scaling_factors = None
        if self.options.auto_scale:
            from .scaling import (
                ScalingFactors,
                compute_scaling_factors,
                scale_problem,
                should_scale_problem,
            )

            if should_scale_problem(problem):
                self.scaling_factors = compute_scaling_factors(problem)
                problem = scale_problem(problem, self.scaling_factors)
                self.logger.info(
                    "Applied automatic problem scaling",
                    extra={
                        "cost_scale": self.scaling_factors.cost_scale,
                        "capacity_scale": self.scaling_factors.capacity_scale,
                        "supply_scale": self.scaling_factors.supply_scale,
                    },
                )
            else:
                # No scaling needed
                self.scaling_factors = ScalingFactors(enabled=False)
        else:
            # Scaling disabled
            from .scaling import ScalingFactors

            self.scaling_factors = ScalingFactors(enabled=False)

        self.problem = problem
        self.ft_rebuilds = 0
        self.ft_updates_since_rebuild = 0

        # Detect network specializations for potential optimizations
        from .specializations import analyze_network_structure, get_specialization_info
        from .specialized_pivots import select_pivot_strategy

        self.network_structure = analyze_network_structure(problem)
        self.specialization_info = get_specialization_info(self.network_structure)

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"Detected network type: {self.specialization_info['description']}",
                extra=self.specialization_info,
            )

        # Select specialized pivot strategy based on detected network type
        # Note: Specialized pivots will be initialized after basis setup
        self.specialized_pivot_strategy = None
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

        # Block size configuration and auto-tuning
        self.auto_tune_block_size = False
        if self.options.block_size is None or self.options.block_size == "auto":
            self.auto_tune_block_size = True
            self.block_size = self._compute_initial_block_size()
        elif isinstance(self.options.block_size, str):
            # This should be caught by validation, but just in case
            self.auto_tune_block_size = True
            self.block_size = self._compute_initial_block_size()
        else:
            self.block_size = self.options.block_size

        # Auto-tuning state
        self.degenerate_pivot_count = 0
        self.total_pivot_count = 0
        self.last_adaptation_iteration = 0
        self.adaptation_interval = 50  # Adapt every 50 iterations

        self.devex_weights: list[float] = [1.0] * len(self.arcs)
        self.original_costs = [arc.cost for arc in self.arcs]
        self.perturbed_costs = [arc.cost for arc in self.arcs]
        self._apply_cost_perturbation()
        self._initialize_tree()
        self._rebuild_tree_structure()
        self._reset_devex_weights()

        # Initialize specialized pivot strategy after basis is set up

        self.specialized_pivot_strategy = select_pivot_strategy(
            self, self.network_structure.network_type.value
        )
        if self.specialized_pivot_strategy is not None and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"Using specialized pivot strategy for {self.network_structure.network_type.value}"
            )

    def _compute_initial_block_size(self) -> int:
        """Compute initial block size based on problem size (static heuristic).

        Strategy:
        - Very small (<100 arcs): num_arcs // 4 (but at least 1)
        - Small (100-1000): num_arcs // 4
        - Medium (1000-10000): num_arcs // 8
        - Large (>10000): num_arcs // 16

        Note: We avoid full scans (block_size == num_arcs) due to potential
        issues with pricing logic edge cases.

        Returns:
            Initial block size (at least 1).
        """
        n = self.actual_arc_count
        if n < 100:
            return max(1, n // 4)  # Very small
        elif n < 1000:
            return max(1, n // 4)  # Small
        elif n < 10000:
            return max(1, n // 8)  # Medium
        else:
            return max(1, n // 16)  # Large

    def _adapt_block_size(self, iteration: int) -> None:
        """Adapt block size based on runtime performance metrics.

        Called periodically during solve to adjust block size:
        - Increase if high degenerate ratio (>30%) - stuck in local area
        - Decrease if low degenerate ratio (<10%) - maximize exploration

        Args:
            iteration: Current iteration number.
        """
        if not self.auto_tune_block_size:
            return

        # Only adapt every N iterations
        if iteration - self.last_adaptation_iteration < self.adaptation_interval:
            return

        # Need sufficient samples to make a decision
        if self.total_pivot_count < 10:
            return

        degenerate_ratio = self.degenerate_pivot_count / self.total_pivot_count
        old_block_size = self.block_size

        # High degenerate ratio: increase block size (explore wider)
        if degenerate_ratio > 0.30:
            self.block_size = min(self.actual_arc_count, int(self.block_size * 1.5))
        # Low degenerate ratio: decrease block size (more focused search)
        elif degenerate_ratio < 0.10:
            self.block_size = max(10, int(self.block_size * 0.75))

        if self.block_size != old_block_size:
            self.logger.debug(
                f"Adapted block_size: {old_block_size} → {self.block_size} "
                f"(degenerate_ratio={degenerate_ratio:.2%})",
                extra={
                    "old_block_size": old_block_size,
                    "new_block_size": self.block_size,
                    "degenerate_ratio": degenerate_ratio,
                    "iteration": iteration,
                },
            )

        # Reset counters for next adaptation window
        self.degenerate_pivot_count = 0
        self.total_pivot_count = 0
        self.last_adaptation_iteration = iteration

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
        """Create an initial spanning tree solution using artificial root arcs.

        This method can be called multiple times (e.g., after a failed warm-start).
        It removes any existing artificial arcs before creating new ones.
        """
        # Remove existing artificial arcs to make this method idempotent
        # Keep only non-artificial arcs
        non_artificial_count = sum(1 for arc in self.arcs if not arc.artificial)
        self.arcs = self.arcs[:non_artificial_count]
        self.original_costs = self.original_costs[:non_artificial_count]
        self.perturbed_costs = self.perturbed_costs[:non_artificial_count]
        self.devex_weights = self.devex_weights[:non_artificial_count]

        # Clear tree adjacency lists (will be rebuilt)
        for adj_list in self.tree_adj:
            adj_list.clear()

        # Mark all non-artificial arcs as out of tree
        for arc in self.arcs:
            arc.in_tree = False
            arc.flow = 0.0

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

    def _apply_warm_start_basis(self, warm_start_basis: Basis) -> bool:
        """Apply a warm-start basis from a previous solve.

        Strategy:
        1. Mark basis arcs as in-tree
        2. Use BFS to find nodes reachable from root via basis arcs
        3. Add artificial arcs for unreachable nodes
        4. Compute flows to satisfy conservation

        Known Limitations:
            There is a known edge case where warm-starting with a basis from
            the exact same problem (identical structure and parameters) can
            fail and return infeasible with objective=0. This requires further
            investigation. See GitHub issue for details.

        Returns:
            True if warm-start was successfully applied, False otherwise.
        """
        # TODO: Fix edge case where warm-starting the exact same problem fails
        # When rebuilding an identical problem and applying warm-start with its
        # own basis, the solver becomes infeasible (objective=0). This was
        # discovered during testing but couldn't be fixed without breaking
        # existing tests. Needs further investigation of flow recomputation logic.
        # IMPORTANT: Do all validation checks BEFORE modifying any arc states
        # If we return False, the solver falls back to cold start, which relies
        # on the initial tree structure set up by _initialize_tree()

        # Check for empty basis first
        if len(warm_start_basis.tree_arcs) == 0:
            self.logger.warning("Warm-start basis is empty. Falling back to cold start.")
            return False

        # Build mapping of arc keys to indices
        arc_key_to_idx: dict[tuple[str, str], int] = {}
        for idx, arc in enumerate(self.arcs):
            if not arc.artificial:
                arc_key_to_idx[arc.key] = idx

        # Check if all basis arcs exist in current problem
        tree_arc_indices: list[int] = []
        for arc_key in warm_start_basis.tree_arcs:
            if arc_key not in arc_key_to_idx:
                self.logger.warning(
                    f"Warm-start basis contains arc {arc_key} not in current problem. "
                    "Falling back to cold start."
                )
                return False
            tree_arc_indices.append(arc_key_to_idx[arc_key])

        # All validation passed - now we can safely modify arc states
        # Mark all arcs as out of tree and reset flows
        for arc in self.arcs:
            arc.in_tree = False
            arc.flow = 0.0

        # Mark basis arcs as in-tree and set their flows from the basis
        for idx in tree_arc_indices:
            arc = self.arcs[idx]
            arc.in_tree = True
            # Use flow from basis if available, otherwise keep at 0
            if arc.key in warm_start_basis.arc_flows:
                basis_flow = warm_start_basis.arc_flows[arc.key]

                # Validate that basis flow is within arc capacity bounds
                if basis_flow < arc.lower - self.tolerance:
                    self.logger.warning(
                        f"Warm-start basis has flow {basis_flow:.2f} below lower bound "
                        f"{arc.lower:.2f} on arc {arc.key}. Falling back to cold start."
                    )
                    return False
                if basis_flow > arc.upper + self.tolerance:
                    self.logger.warning(
                        f"Warm-start basis has flow {basis_flow:.2f} exceeding capacity "
                        f"{arc.upper:.2f} on arc {arc.key}. Falling back to cold start."
                    )
                    return False

                arc.flow = basis_flow

        # Find connected components using Union-Find
        # We need to connect each component to the root
        parent_uf = list(range(self.node_count))

        def find(x: int) -> int:
            if parent_uf[x] != x:
                parent_uf[x] = find(parent_uf[x])
            return parent_uf[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent_uf[px] = py

        # Build components from in-tree arcs
        for arc in self.arcs:
            if arc.in_tree:
                union(arc.tail, arc.head)

        # For each component (other than root's component), add one artificial arc
        # We need to pick the right arc based on the component's net supply
        components_seen = {find(self.root)}
        artificial_arcs_added = 0

        for node_idx in range(1, self.node_count):  # Skip root
            component = find(node_idx)
            if component not in components_seen:
                components_seen.add(component)

                # Calculate net supply for this component
                component_supply = sum(
                    self.node_supply[i] for i in range(1, self.node_count) if find(i) == component
                )

                # Find appropriate artificial arc based on supply direction
                # Positive supply → arc from component to root
                # Negative supply → arc from root to component
                # Zero supply → pick any arc (flow will be 0)
                for arc in self.arcs:
                    if not arc.artificial or arc.in_tree:
                        continue
                    # Check if arc connects this component to root
                    if arc.tail == self.root or arc.head == self.root:
                        other_node = arc.head if arc.tail == self.root else arc.tail
                        if find(other_node) == component:
                            # Check if arc direction matches supply
                            if component_supply > self.tolerance and arc.tail == other_node:
                                # Positive supply, need arc TO root
                                arc.in_tree = True
                                artificial_arcs_added += 1
                                break
                            elif component_supply < -self.tolerance and arc.head == other_node:
                                # Negative supply, need arc FROM root
                                arc.in_tree = True
                                artificial_arcs_added += 1
                                break
                            elif abs(component_supply) <= self.tolerance:
                                # Balanced component, any arc works
                                arc.in_tree = True
                                artificial_arcs_added += 1
                                break

        # Rebuild tree adjacency lists
        for adj_list in self.tree_adj:
            adj_list.clear()
        for idx, arc in enumerate(self.arcs):
            if arc.in_tree:
                self.tree_adj[arc.tail].append(idx)
                self.tree_adj[arc.head].append(idx)

        # Verify we have exactly n-1 tree arcs
        total_tree_arcs = sum(1 for arc in self.arcs if arc.in_tree)
        expected_tree_arcs = self.node_count - 1
        if total_tree_arcs != expected_tree_arcs:
            self.logger.warning(
                f"Warm-start tree has {total_tree_arcs} arcs, expected {expected_tree_arcs}. "
                "Falling back to cold start."
            )
            return False

        # Recompute tree arc flows to satisfy flow conservation
        if not self._recompute_tree_flows():
            # Flow recomputation failed due to capacity violations
            self.logger.warning(
                "Warm-start basis incompatible with current capacities. Falling back to cold start."
            )
            return False

        # Log success
        self.logger.info(
            f"Successfully applied warm-start basis with {len(tree_arc_indices)} basis arcs",
            extra={
                "basis_arcs": len(tree_arc_indices),
                "artificial_added": artificial_arcs_added,
                "total_tree_arcs": total_tree_arcs,
            },
        )
        return True

    def _recompute_tree_flows(self) -> bool:
        """Recompute flows on artificial tree arcs to satisfy flow conservation.

        This is called during warm-start to set flows on artificial arcs that
        were added to complete the spanning tree. Non-artificial arc flows are
        already set from the warm-start basis.

        Uses post-order tree traversal to compute flows that satisfy
        conservation at each node given the current node supplies.

        Returns:
            True if flows could be computed without violating capacity constraints,
            False if the warm-start basis is incompatible with current capacities.
        """
        # Build parent structure via BFS from root
        parent: list[int | None] = [None] * self.node_count
        parent_arc_idx: list[int | None] = [None] * self.node_count
        parent[self.root] = self.root

        queue = deque([self.root])
        visited = {self.root}

        while queue:
            node = queue.popleft()
            for arc_idx in self.tree_adj[node]:
                arc = self.arcs[arc_idx]
                if not arc.in_tree:
                    continue
                neighbor = arc.head if arc.tail == node else arc.tail
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = node
                parent_arc_idx[neighbor] = arc_idx
                queue.append(neighbor)

        # Track if any capacity violations occur
        capacity_violated = False

        # Compute flows using post-order traversal
        def compute_flow_to_parent(node: int) -> None:
            """Recursively compute flow on arc to parent."""
            nonlocal capacity_violated

            # First, process all children
            for arc_idx in self.tree_adj[node]:
                arc = self.arcs[arc_idx]
                if not arc.in_tree:
                    continue
                neighbor = arc.head if arc.tail == node else arc.tail
                if parent[neighbor] == node:
                    compute_flow_to_parent(neighbor)

            # Now compute flow to our parent
            if node == self.root:
                return

            parent_arc = parent_arc_idx[node]
            if parent_arc is None:
                return

            arc = self.arcs[parent_arc]

            # Compute net flow balance at this node
            # balance = supply + (inflow from children - outflow to children)
            flow_balance = self.node_supply[node]

            for child_arc_idx in self.tree_adj[node]:
                child_arc = self.arcs[child_arc_idx]
                if not child_arc.in_tree:
                    continue
                neighbor = child_arc.head if child_arc.tail == node else child_arc.tail
                if parent[neighbor] == node:  # neighbor is a child
                    if child_arc.tail == neighbor:
                        # Flow from child to node (inflow)
                        flow_balance += child_arc.flow
                    else:
                        # Flow from node to child (outflow)
                        flow_balance -= child_arc.flow

            # Compute required flow on arc to parent
            required_flow = flow_balance if arc.tail == node else -flow_balance

            # Check if required flow violates capacity constraints
            if required_flow < -self.tolerance:
                # Negative flow required (violates lower bound of 0)
                self.logger.warning(
                    f"Warm-start requires negative flow {required_flow:.2f} on arc {arc.key}"
                )
                capacity_violated = True
                return
            elif required_flow > arc.upper + self.tolerance:
                # Flow exceeds capacity
                self.logger.warning(
                    f"Warm-start requires flow {required_flow:.2f} exceeding capacity "
                    f"{arc.upper:.2f} on arc {arc.key}"
                )
                capacity_violated = True
                return

            # Set flow on arc to parent (clamp to valid range for numerical stability)
            arc.flow = max(0.0, min(arc.upper, required_flow))

        # Start recursion from all children of root
        for arc_idx in self.tree_adj[self.root]:
            arc = self.arcs[arc_idx]
            if not arc.in_tree:
                continue
            neighbor = arc.head if arc.tail == self.root else arc.tail
            if parent[neighbor] == self.root:
                compute_flow_to_parent(neighbor)

        # Return whether flow recomputation succeeded without capacity violations
        return not capacity_violated

    def _extract_basis(self) -> Basis:
        """Extract the current basis (spanning tree) for warm-starting future solves."""
        tree_arcs: set[tuple[str, str]] = set()
        arc_flows: dict[tuple[str, str], float] = {}

        for arc in self.arcs:
            if arc.in_tree and not arc.artificial:
                tree_arcs.add(arc.key)
                arc_flows[arc.key] = arc.flow

        return Basis(tree_arcs=tree_arcs, arc_flows=arc_flows)

    def _find_entering_arc(self, allow_zero: bool) -> tuple[int, int] | None:
        """Return (arc_idx, direction) for entering arc, where direction is +1 or -1."""
        # Try specialized pivot strategy first if available
        if self.specialized_pivot_strategy is not None:
            result = self.specialized_pivot_strategy.find_entering_arc(allow_zero)
            if result is not None:
                return result

        # Fall back to standard pricing strategies
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

            # Adapt block size if auto-tuning is enabled
            self._adapt_block_size(total_iterations_offset + iterations)

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

        if self.logger.isEnabledFor(logging.DEBUG):
            entering_rc = (
                entering.cost
                + self.basis.potential[entering.tail]
                - self.basis.potential[entering.head]
            )
            if direction == -1:
                entering_rc = -entering_rc
            self.logger.debug(
                "Pivot: entering arc",
                extra={
                    "entering_arc": entering.key,
                    "direction": "forward" if direction == 1 else "backward",
                    "reduced_cost": entering_rc,
                },
            )

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
            if not math.isfinite(entering_weight):
                self.logger.warning(
                    "Devex weight is not finite, clamping to minimum",
                    extra={"entering_arc": entering.key, "weight": entering_weight},
                )
                entering_weight = DEVEX_WEIGHT_MIN
            elif entering_weight <= DEVEX_WEIGHT_MIN:
                entering_weight = DEVEX_WEIGHT_MIN
            elif entering_weight > DEVEX_WEIGHT_MAX:
                self.logger.warning(
                    "Devex weight exceeds maximum, clamping",
                    extra={
                        "entering_arc": entering.key,
                        "weight": entering_weight,
                        "max": DEVEX_WEIGHT_MAX,
                    },
                )
                entering_weight = DEVEX_WEIGHT_MAX
        else:
            entering_weight = 1.0
            for idx, _ in cycle:
                if idx == arc_idx:
                    continue
                entering_weight += self.devex_weights[idx]
        self.devex_weights[arc_idx] = max(DEVEX_WEIGHT_MIN, entering_weight)

        # Track pivot for auto-tuning
        self.total_pivot_count += 1
        is_degenerate = (leaving_idx == arc_idx) or (abs(theta) < self.tolerance)
        if is_degenerate:
            self.degenerate_pivot_count += 1

        if leaving_idx == arc_idx:
            entering.in_tree = False
            # Degenerate pivot: tree unchanged but flows adjusted.
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Degenerate pivot (entering arc is also leaving)",
                    extra={"arc": entering.key, "theta": theta},
                )
            return
        leaving_arc = self.arcs[leaving_idx]
        leaving_arc.in_tree = False
        self.devex_weights[leaving_idx] = 1.0

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Pivot: leaving arc",
                extra={
                    "leaving_arc": leaving_arc.key,
                    "theta": theta,
                    "is_degenerate": abs(theta) < self.tolerance,
                },
            )

        # Check if we've hit the FT update limit and need a full rebuild
        force_rebuild = self.ft_updates_since_rebuild >= self.options.ft_update_limit

        basis_updated = self.basis.replace_arc(leaving_idx, arc_idx, self.arcs, self.tolerance)
        self._update_tree_sets()

        if force_rebuild:
            # Periodic rebuild to maintain numerical stability
            self.logger.debug(
                "FT update limit reached; rebuilding basis",
                extra={
                    "leaving_idx": leaving_idx,
                    "entering_idx": arc_idx,
                    "ft_updates": self.options.ft_update_limit,
                },
            )
            self.basis.rebuild(self.tree_adj, self.arcs, build_numeric=True)
            self.ft_rebuilds += 1
            self.ft_updates_since_rebuild = 0
            self._reset_devex_weights()
        elif not basis_updated:
            # FT update failed: refresh the numeric basis and reset Devex heuristics.
            self.logger.warning(
                "Forrest-Tomlin update failed; rebuilding basis for numerical stability",
                extra={
                    "leaving_arc": leaving_arc.key,
                    "entering_arc": entering.key,
                    "ft_rebuilds": self.ft_rebuilds + 1,
                },
            )
            self.basis.rebuild(self.tree_adj, self.arcs, build_numeric=True)
            self.ft_rebuilds += 1
            self.ft_updates_since_rebuild = 0
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
        warm_start_basis: Basis | None = None,
    ) -> FlowResult:
        """Solve the minimum-cost flow problem.

        Args:
            max_iterations: Maximum number of simplex iterations. If None, uses value from
                           SolverOptions (default: max(100, 5*num_arcs)).
            progress_callback: Optional callback function to receive progress updates.
            progress_interval: Number of iterations between progress callbacks (default: 100).
            warm_start_basis: Optional basis from a previous solve to initialize the solver.

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

        # Calculate total supply for logging
        total_supply = sum(abs(s) for s in self.node_supply)

        self.logger.info(
            "Starting network simplex solver",
            extra={
                "nodes": len(self.node_ids) - 1,  # Exclude artificial root
                "arcs": self.actual_arc_count,
                "max_iterations": max_iterations,
                "pricing_strategy": self.options.pricing_strategy,
                "total_supply": total_supply,
                "tolerance": self.tolerance,
                "warm_start": warm_start_basis is not None,
            },
        )

        # Try to apply warm-start basis if provided
        skip_phase_1 = False
        warm_start_applied = False
        if warm_start_basis is not None:
            self.logger.info("Attempting to apply warm-start basis")
            if self._apply_warm_start_basis(warm_start_basis):
                warm_start_applied = True
                # Apply Phase 1 costs and rebuild tree structure
                self._apply_phase_costs(phase=1)
                self._rebuild_tree_structure()

                # Check if warm-start basis is feasible (no artificial arcs in tree with flow)
                artificial_in_tree = sum(1 for arc in self.arcs if arc.in_tree and arc.artificial)
                if artificial_in_tree == 0:
                    # No artificial arcs in tree means basis fully covers all nodes
                    # Switch to Phase 2 costs immediately
                    self.logger.info(
                        "Warm-start basis fully covers all nodes, skipping Phase 1",
                        extra={"artificial_in_tree": artificial_in_tree},
                    )
                    self._apply_phase_costs(phase=2)
                    self._rebuild_tree_structure()
                    skip_phase_1 = True
                else:
                    # Need Phase 1 to eliminate artificial flow
                    artificial_flow = sum(
                        arc.flow
                        for arc in self.arcs
                        if arc.artificial and arc.flow > self.tolerance
                    )
                    self.logger.info(
                        "Warm-start basis requires Phase 1 refinement",
                        extra={
                            "artificial_flow": artificial_flow,
                            "artificial_in_tree": artificial_in_tree,
                        },
                    )
            else:
                # Warm-start failed, reinitialize with cold start
                self.logger.info("Warm-start failed, performing cold start")
                self._initialize_tree()
                self._rebuild_tree_structure()

        # Phase 1: find feasible flow minimizing artificial usage.
        if not skip_phase_1:
            self.logger.info(
                "Phase 1: Finding initial feasible solution", extra={"elapsed_ms": 0.0}
            )
            # Note: Phase costs and tree structure already applied for warm-start above
            # For cold start (or failed warm-start), need to apply them here
            if not warm_start_applied:
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
            # Calculate total artificial flow for diagnostics
            artificial_flow = sum(
                arc.flow for arc in self.arcs if arc.artificial and arc.flow > self.tolerance
            )
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Phase 1 complete",
                extra={
                    "iterations": iters,
                    "total_iterations": total_iterations,
                    "artificial_flow": artificial_flow,
                    "elapsed_ms": elapsed_ms,
                },
            )
        else:
            iters = 0
            self.logger.info("Phase 1 skipped (warm-start basis is feasible)")

        # Check for infeasibility: artificial arcs have flow
        has_artificial_flow = any(arc.artificial and arc.flow > self.tolerance for arc in self.arcs)

        # For warm-start cases, also verify flow conservation (catches cases where warm-start
        # leads to invalid state with zero artificial flow but violated conservation)
        flow_conservation_violated = False
        if warm_start_applied:
            for node_idx in range(1, self.node_count):
                if node_idx == self.root:
                    continue

                # Calculate net flow: supply + inflow - outflow
                net_flow = self.node_supply[node_idx]
                for arc in self.arcs:
                    if arc.tail == node_idx:
                        net_flow -= arc.flow
                    elif arc.head == node_idx:
                        net_flow += arc.flow

                if abs(net_flow) > self.tolerance:
                    self.logger.error(
                        f"Flow conservation violated at node {self.node_ids[node_idx]}: "
                        f"imbalance = {net_flow:.6f}"
                    )
                    flow_conservation_violated = True

        infeasible = has_artificial_flow or flow_conservation_violated
        if infeasible:
            if total_iterations >= max_iterations:
                self.logger.error(
                    "Iteration limit reached before finding feasible solution",
                    extra={"iterations": total_iterations, "max_iterations": max_iterations},
                )
                return FlowResult(
                    objective=0.0,
                    flows={},
                    status="iteration_limit",
                    iterations=total_iterations,
                    duals={},
                )
            self.logger.error(
                "Problem is infeasible - no feasible solution exists",
                extra={"iterations": total_iterations},
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
        self.logger.info(
            "Phase 2: Optimizing from feasible basis",
            extra={"remaining_iterations": remaining},
        )
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
        elapsed_ms = (time.time() - start_time) * 1000

        # Calculate preliminary objective for logging
        preliminary_objective = sum(
            (arc.flow + arc.shift) * self.original_costs[idx]
            for idx, arc in enumerate(self.arcs)
            if not arc.artificial
        )

        self.logger.info(
            "Phase 2 complete",
            extra={
                "iterations": iters,
                "total_iterations": total_iterations,
                "objective": preliminary_objective,
                "elapsed_ms": elapsed_ms,
            },
        )

        status = "optimal" if total_iterations < max_iterations else "iteration_limit"

        if status == "iteration_limit":
            self.logger.warning(
                "Iteration limit reached before optimality",
                extra={"iterations": total_iterations, "max_iterations": max_iterations},
            )

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

        # Final solution logging with comprehensive metrics
        elapsed_ms = (time.time() - start_time) * 1000
        tree_arcs = sum(1 for arc in self.arcs if arc.in_tree and not arc.artificial)
        nonzero_flows = len(flows)

        self.logger.info(
            "Solver complete",
            extra={
                "status": status,
                "objective": float(round(objective, 12)),
                "iterations": total_iterations,
                "elapsed_ms": elapsed_ms,
                "tree_arcs": tree_arcs,
                "nonzero_flows": nonzero_flows,
                "ft_rebuilds": self.ft_rebuilds,
            },
        )

        # Extract basis for warm-starting future solves (only for feasible solutions)
        basis = None
        if status in ("optimal", "iteration_limit"):
            basis = self._extract_basis()

        # Unscale solution if automatic scaling was applied
        if self.scaling_factors and self.scaling_factors.enabled:
            from .scaling import unscale_solution

            flows, objective = unscale_solution(flows, objective, self.scaling_factors)
            self.logger.debug("Unscaled solution back to original units")

        return FlowResult(
            objective=float(round(objective, 12)),
            flows=flows,
            status=status,
            iterations=total_iterations,
            duals=duals,
            basis=basis,
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
