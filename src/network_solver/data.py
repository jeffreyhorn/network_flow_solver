"""Core data structures for linear network programming problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .exceptions import InvalidProblemError


@dataclass(frozen=True)
class Node:
    """Represents a network node with a supply (positive) or demand (negative).

    Attributes:
        id: Unique identifier for the node.
        supply: Supply (positive), demand (negative), or transshipment (0.0).
                Default is 0.0 (transshipment node).

    Examples:
        >>> # Supply node (e.g., factory producing 100 units)
        >>> factory = Node(id="factory_a", supply=100.0)

        >>> # Demand node (e.g., warehouse needing 50 units)
        >>> warehouse = Node(id="warehouse_1", supply=-50.0)

        >>> # Transshipment node (intermediate node with no supply/demand)
        >>> hub = Node(id="distribution_hub", supply=0.0)

    Note:
        For a feasible problem, the sum of all node supplies must equal zero
        (total supply = total demand). This is enforced during problem validation.
    """

    id: str
    supply: float = 0.0


@dataclass(frozen=True)
class Arc:
    """Represents a directed arc with a capacity, cost, and optional lower bound.

    Attributes:
        tail: Source node ID.
        head: Destination node ID.
        capacity: Upper bound on flow. Use None for infinite capacity.
        cost: Cost per unit of flow on this arc.
        lower: Lower bound on flow (default: 0.0). Must be <= capacity.

    Examples:
        >>> # Basic arc from factory to warehouse
        >>> arc1 = Arc(tail="factory", head="warehouse", capacity=100.0, cost=2.5)

        >>> # Arc with infinite capacity (e.g., internal transfer)
        >>> arc2 = Arc(tail="node_a", head="node_b", capacity=None, cost=0.0)

        >>> # Arc with lower bound (e.g., minimum contract requirement)
        >>> arc3 = Arc(tail="supplier", head="customer",
        ...            capacity=500.0, cost=10.0, lower=100.0)

    Raises:
        InvalidProblemError: If tail == head (self-loops not supported).
        InvalidProblemError: If capacity < lower bound.

    Note:
        For undirected graphs, use NetworkProblem with directed=False. The edge
        will be automatically transformed to allow bidirectional flow.
        See docs/api.md#working-with-undirected-graphs for details.
    """

    tail: str
    head: str
    capacity: float | None
    cost: float
    lower: float = 0.0

    def __post_init__(self) -> None:
        if self.tail == self.head:
            raise InvalidProblemError(
                f"Self-loop detected on node '{self.tail}'. Self-loops are not supported "
                f"in network simplex."
            )
        if self.capacity is not None and self.capacity < self.lower:
            raise InvalidProblemError(
                f"Arc {self.tail} -> {self.head} has capacity ({self.capacity}) less than "
                f"lower bound ({self.lower}). Capacity must be >= lower bound."
            )


@dataclass
class NetworkProblem:
    """Encapsulates a minimum-cost flow problem.

    This class represents a complete network flow problem instance with nodes,
    arcs, and configuration. Use solve_min_cost_flow() to find an optimal solution.

    Attributes:
        directed: True for directed graph, False for undirected.
        nodes: Dictionary mapping node IDs to Node objects.
        arcs: List of Arc objects defining the network structure.
        tolerance: Numerical tolerance for feasibility checks (default: 1e-3).

    Examples:
        >>> # Directed transportation problem
        >>> nodes = {
        ...     "factory": Node(id="factory", supply=100.0),
        ...     "warehouse": Node(id="warehouse", supply=-100.0),
        ... }
        >>> arcs = [
        ...     Arc(tail="factory", head="warehouse", capacity=150.0, cost=5.0)
        ... ]
        >>> problem = NetworkProblem(directed=True, nodes=nodes, arcs=arcs)
        >>> problem.validate()  # Check problem is well-formed

        >>> # Undirected network (e.g., bidirectional cables)
        >>> nodes_undirected = {
        ...     "A": Node(id="A", supply=50.0),
        ...     "B": Node(id="B", supply=-50.0),
        ... }
        >>> edges = [
        ...     Arc(tail="A", head="B", capacity=100.0, cost=2.0)  # Bidirectional
        ... ]
        >>> problem = NetworkProblem(directed=False, nodes=nodes_undirected, arcs=edges)

    Methods:
        validate(): Check problem validity (supply balance, arc endpoints).
        undirected_expansion(): Transform undirected edges to directed arcs.

    See Also:
        - build_problem(): Construct from dictionaries (useful for JSON input).
        - solve_min_cost_flow(): Solve the problem using network simplex.
        - docs/algorithm.md: Mathematical formulation and algorithm details.
    """

    directed: bool
    nodes: dict[str, Node]
    arcs: list[Arc]
    tolerance: float = 1e-3

    def validate(self) -> None:
        # Enforce flow conservation before solving so later phases can assume balance.
        total_supply = sum(node.supply for node in self.nodes.values())
        if abs(total_supply) > self.tolerance:
            raise InvalidProblemError(
                f"Problem is unbalanced: total supply {total_supply:.6f} exceeds tolerance "
                f"{self.tolerance}. For a valid flow problem, the sum of all node supplies "
                f"must equal zero (supplies balance demands)."
            )
        for arc in self.arcs:
            if arc.tail not in self.nodes:
                raise InvalidProblemError(
                    f"Arc tail '{arc.tail}' not found in node set. All arc endpoints must "
                    f"reference existing nodes."
                )
            if arc.head not in self.nodes:
                raise InvalidProblemError(
                    f"Arc head '{arc.head}' not found in node set. All arc endpoints must "
                    f"reference existing nodes."
                )

    def undirected_expansion(self) -> Sequence[Arc]:
        """Return arcs expanded to directed equivalents when graph is undirected.

        For undirected graphs, each edge {u, v} with capacity C and cost c is transformed
        into a single directed arc (u, v) with:
        - capacity: C
        - lower bound: -C (allowing flow in either direction)
        - cost: c (same cost regardless of direction)

        This transformation allows bidirectional flow while maintaining the network simplex
        structure. A positive flow value means flow goes tail→head, while a negative value
        means flow goes head→tail.

        Requirements for undirected edges:
        - Must have finite capacity (no infinite capacity edges)
        - Cannot specify custom lower bounds (automatically set to -capacity)
        - Costs are symmetric (same cost in both directions)

        Returns:
            Tuple of Arc objects representing the directed transformation.

        Raises:
            InvalidProblemError: If any edge has infinite capacity or custom lower bound.
        """
        if self.directed:
            return tuple(self.arcs)
        expanded: list[Arc] = []
        # Translate each undirected edge into a directed representation that carries
        # the same capacity while respecting the simplex solver's sign convention.
        for arc in self.arcs:
            if arc.capacity is None:
                raise InvalidProblemError(
                    f"Undirected edge {arc.tail} -- {arc.head} has infinite capacity. "
                    f"Undirected graphs require finite capacity on all edges. "
                    f"This is necessary to enable bidirectional flow representation where "
                    f"the edge is transformed to a directed arc with lower bound -capacity. "
                    f"Please specify a finite capacity value for this edge."
                )
            cap = float(arc.capacity)
            if abs(arc.lower) > 1e-12 and not math.isclose(
                arc.lower, -cap, rel_tol=0.0, abs_tol=1e-12
            ):
                raise InvalidProblemError(
                    f"Undirected edge {arc.tail} -- {arc.head} has custom lower bound "
                    f"({arc.lower}). Undirected edges do not support custom lower bounds because "
                    f"the lower bound is automatically set to -capacity to enable bidirectional flow. "
                    f"For undirected graphs, leave the lower bound at 0.0 (default) and it will be "
                    f"automatically transformed to -{cap} during preprocessing."
                )
            # Encode the backwards arc implicitly via a negative lower bound on the forward arc.
            # This allows flow from -capacity to +capacity on the edge.
            lower_bound = -float(arc.capacity)
            expanded.append(
                Arc(
                    tail=arc.tail,
                    head=arc.head,
                    capacity=cap,
                    cost=arc.cost,
                    lower=lower_bound,
                )
            )
        return tuple(expanded)


@dataclass
class Basis:
    """Represents a basis (spanning tree) for warm-starting the solver.

    A basis consists of the set of basic arcs that form a spanning tree in the
    network. Warm-starting allows the solver to begin from a previous solution's
    basis rather than constructing an initial basis from scratch, which can
    significantly reduce solve time for similar problems.

    Attributes:
        tree_arcs: Set of arc (tail, head) tuples that form the spanning tree basis.
                   These arcs are "in the basis" (basic variables in simplex terminology).
        arc_flows: Dictionary mapping arc (tail, head) tuples to their flow values.
                   Used to initialize the flows when warm-starting.

    Examples:
        >>> # Solve once and extract basis
        >>> result1 = solve_min_cost_flow(problem1)
        >>> basis = result1.basis
        >>> print(f"Basis has {len(basis.tree_arcs)} tree arcs")
        Basis has 5 tree arcs
        >>>
        >>> # Use basis to warm-start a similar problem
        >>> result2 = solve_min_cost_flow(problem2, warm_start_basis=basis)
        >>> print(f"Warm-start used {result2.iterations} iterations")
        Warm-start used 3 iterations

    See Also:
        - FlowResult.basis: Extract basis from a solution
        - solve_min_cost_flow(warm_start_basis=...): Warm-start solver
        - docs/examples.md#warm-starting: Warm-start examples

    Note:
        The basis is problem-specific. Warm-starting works best when:
        - The network structure is similar (same nodes and arcs)
        - Supply/demand or costs have changed slightly
        - Capacities have been adjusted
    """

    tree_arcs: set[tuple[str, str]] = field(default_factory=set)
    arc_flows: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass
class FlowResult:
    """Represents the output of a minimum-cost flow computation.

    Attributes:
        objective: Total cost of the solution (∑ cost_ij * flow_ij).
        flows: Dictionary mapping arc (tail, head) tuples to flow values.
               For undirected graphs, positive = tail→head, negative = head→tail.
        status: Solution status:
                - 'optimal': Optimal solution found
                - 'infeasible': No feasible solution exists
                - 'unbounded': Objective can decrease without bound
                - 'iteration_limit': Reached max iterations before optimality
        iterations: Number of simplex iterations performed.
        duals: Dictionary mapping node IDs to dual values (node potentials).
               For optimal solutions, these represent shadow prices for supply/demand.
               Useful for sensitivity analysis and what-if scenarios.
        basis: Basis (spanning tree) information for warm-starting subsequent solves.
               Can be passed to solve_min_cost_flow(warm_start_basis=...) to initialize
               from this solution's basis. None if solution is infeasible/unbounded.

    Examples:
        >>> from network_solver import solve_min_cost_flow, build_problem
        >>> nodes = [
        ...     {"id": "A", "supply": 10.0},
        ...     {"id": "B", "supply": -10.0}
        ... ]
        >>> arcs = [{"tail": "A", "head": "B", "capacity": 20.0, "cost": 3.0}]
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>> result = solve_min_cost_flow(problem)
        >>> print(f"Status: {result.status}")
        Status: optimal
        >>> print(f"Total cost: ${result.objective:.2f}")
        Total cost: $30.00
        >>> print(f"Flow A→B: {result.flows[('A', 'B')]:.1f}")
        Flow A→B: 10.0
        >>> print(f"Shadow price at A: {result.duals['A']:.2f}")
        Shadow price at A: -3.00
        >>> print(f"Basis arcs: {len(result.basis.tree_arcs)}")
        Basis arcs: 1

    See Also:
        - solve_min_cost_flow(): Main solver function.
        - docs/examples.md#sensitivity-analysis: Using dual values.
        - docs/examples.md#warm-starting: Using basis for warm-starts.
        - save_result(): Persist results to JSON.
    """

    objective: float
    flows: dict[tuple[str, str], float] = field(default_factory=dict)
    status: str = "optimal"
    iterations: int = 0
    duals: dict[str, float] = field(default_factory=dict)
    basis: Basis | None = None


@dataclass(frozen=True)
class ProgressInfo:
    """Progress information provided during solver execution.

    Attributes:
        iteration: Current iteration number.
        max_iterations: Maximum allowed iterations.
        phase: Current phase (1 for feasibility, 2 for optimality).
        phase_iterations: Iterations completed in current phase.
        objective_estimate: Current estimate of objective value (may be inaccurate during phase 1).
        elapsed_time: Elapsed time in seconds since solve started.
    """

    iteration: int
    max_iterations: int
    phase: int
    phase_iterations: int
    objective_estimate: float
    elapsed_time: float


# Type alias for progress callback function
ProgressCallback = Callable[[ProgressInfo], None]


@dataclass
class SolverOptions:
    """Configuration options for the network simplex solver.

    Attributes:
        max_iterations: Maximum number of simplex iterations.
                       If None, defaults to max(100, 5*num_arcs).
                       Typical range: 100-10000 depending on problem size.
        tolerance: Numerical tolerance for feasibility and optimality checks (default: 1e-6).
                  Lower values (1e-8, 1e-10) give higher precision but may increase iterations.
                  Higher values (1e-4, 1e-3) are faster but less precise.
        pricing_strategy: Arc pricing strategy:
                         - "devex" (default): Devex normalized pricing (usually faster)
                         - "dantzig": Most negative reduced cost (simpler, sometimes better for dense problems)
        block_size: Number of arcs to examine per pricing block.
                   - None or "auto": Auto-tune based on problem size with runtime adaptation
                   - int: Fixed block size (no adaptation)
                   Smaller blocks (10-50) = more pivots, larger blocks = fewer pivots.
        ft_update_limit: Maximum Forrest-Tomlin basis updates before full rebuild (default: 64).
                        Lower values (20-40) = more stable but slower.
                        Higher values (100-200) = faster but may lose numerical stability.
        projection_cache_size: Cache size for basis projections (default: 100).
                              Optimized cache provides 14% speedup on medium problems (70+ nodes).
                              Cache is invalidated on basis changes (no LRU overhead).
                              Recommended: 100 for most problems, 0 to disable for very small problems.
                              Memory usage: ~800 bytes per cached projection.
        auto_scale: Enable automatic problem scaling for numerical stability (default: True).
                   - True: Automatically detect and scale problems with wide value ranges
                   - False: Use original problem values without scaling
                   Scaling is applied when costs, capacities, or supplies differ by >6 orders of magnitude.
                   The solution is automatically unscaled back to original units.
        adaptive_refactorization: Enable adaptive basis refactorization (default: True).
                                 - True: Monitor condition number and trigger rebuilds adaptively
                                 - False: Use fixed ft_update_limit only
                                 When enabled, rebuilds are triggered by either:
                                 1. Condition number exceeds threshold
                                 2. Update count exceeds current ft_update_limit
        condition_number_threshold: Condition number limit for triggering rebuild (default: 1e12).
                                   Lower values (1e10) = more frequent rebuilds, more stable
                                   Higher values (1e14) = fewer rebuilds, faster but less stable
                                   Only used when adaptive_refactorization=True.
        adaptive_ft_min: Minimum value for adaptive ft_update_limit (default: 20).
                        Prevents limit from becoming too small.
        adaptive_ft_max: Maximum value for adaptive ft_update_limit (default: 200).
                        Prevents limit from becoming too large.
        use_dense_inverse: Compute and maintain dense basis inverse (default: None = auto).
                          - None (default): Auto-detect based on sparse LU availability
                            - If scipy available: False (use sparse LU only)
                            - If scipy unavailable: True (fall back to dense inverse)
                          - False: Force sparse LU only (requires scipy, raises error if unavailable)
                          - True: Always compute dense inverse with np.linalg.inv (O(n³) memory and time)
                          Dense inverse enables Sherman-Morrison rank-1 updates but is memory-intensive.
                          For large problems (>1000 nodes), sparse LU is recommended when available.
        use_vectorized_pricing: Enable vectorized pricing operations (default: True).
                               - True: Use NumPy vectorized operations for 2-3x speedup (recommended)
                               - False: Use traditional loop-based pricing (for debugging/comparison)
                               Only applies to Devex pricing strategy. Dantzig pricing is always loop-based.
                               Vectorization provides significant performance improvements on medium/large problems.

    Examples:
        >>> # Default options (auto-tuning enabled)
        >>> options = SolverOptions()

        >>> # Explicit auto-tuning
        >>> options = SolverOptions(block_size="auto")

        >>> # High-precision solve
        >>> options = SolverOptions(tolerance=1e-10, ft_update_limit=32)

        >>> # Fast solve with fixed block size (no auto-tuning)
        >>> options = SolverOptions(
        ...     tolerance=1e-4,
        ...     pricing_strategy="devex",
        ...     block_size=100,
        ...     ft_update_limit=128
        ... )

        >>> # Stable solve for ill-conditioned problems
        >>> options = SolverOptions(
        ...     tolerance=1e-8,
        ...     ft_update_limit=20
        ... )

    See Also:
        - solve_min_cost_flow(): Pass options to control solver behavior.
        - docs/benchmarks.md: Performance tuning guidelines.
    """

    max_iterations: int | None = None
    tolerance: float = 1e-6
    pricing_strategy: str = "devex"
    block_size: int | str | None = None
    ft_update_limit: int = 64
    projection_cache_size: int = 100  # Optimized cache provides 14% speedup on medium problems
    auto_scale: bool = True
    adaptive_refactorization: bool = True
    condition_number_threshold: float = 1e12
    adaptive_ft_min: int = 20
    adaptive_ft_max: int = 200
    use_dense_inverse: bool | None = None
    use_vectorized_pricing: bool = True  # Re-enabled with cycling fix (see #TODO)

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise InvalidProblemError(
                f"Tolerance must be positive, got {self.tolerance}. "
                f"Tolerance controls numerical precision for feasibility and optimality checks."
            )
        if self.pricing_strategy not in ("devex", "dantzig"):
            raise InvalidProblemError(
                f"Invalid pricing strategy '{self.pricing_strategy}'. Must be 'devex' or 'dantzig'."
            )
        if self.block_size is not None:
            if isinstance(self.block_size, str):
                if self.block_size != "auto":
                    raise InvalidProblemError(
                        f"Invalid block_size '{self.block_size}'. Must be a positive integer, 'auto', or None."
                    )
            elif self.block_size <= 0:
                raise InvalidProblemError(
                    f"Block size must be positive, got {self.block_size}. "
                    f"Block size controls how many arcs are examined per pricing iteration."
                )
        if self.ft_update_limit <= 0:
            raise InvalidProblemError(
                f"FT update limit must be positive, got {self.ft_update_limit}. "
                f"This controls how often the basis factorization is rebuilt."
            )
        if self.condition_number_threshold <= 1:
            raise InvalidProblemError(
                f"Condition number threshold must be > 1, got {self.condition_number_threshold}. "
                f"Typical values are 1e10 to 1e14."
            )
        if self.adaptive_ft_min <= 0 or self.adaptive_ft_min > self.adaptive_ft_max:
            raise InvalidProblemError(
                f"Adaptive FT min must be positive and <= max, got min={self.adaptive_ft_min}, max={self.adaptive_ft_max}."
            )

        # Resolve use_dense_inverse default based on sparse LU availability
        if self.use_dense_inverse is None:
            from .basis_lu import has_sparse_lu

            # Auto-detect: use sparse LU if available, otherwise fall back to dense
            object.__setattr__(self, "use_dense_inverse", not has_sparse_lu())
        elif self.use_dense_inverse is False:
            # Validate that sparse LU is actually available
            from .basis_lu import has_sparse_lu

            if not has_sparse_lu():
                raise InvalidProblemError(
                    "use_dense_inverse=False requires scipy for sparse LU factorization. "
                    "Either install scipy or set use_dense_inverse=True to use dense inverse."
                )


def build_problem(
    nodes: Iterable[dict[str, float]],
    arcs: Iterable[dict[str, float]],
    directed: bool,
    tolerance: float,
) -> NetworkProblem:
    """Factory helper used by IO layer to assemble a NetworkProblem."""
    node_map: dict[str, Node] = {}
    for node in nodes:
        node_id = str(node["id"])
        # Deduplicate nodes here so downstream code can index directly.
        if node_id in node_map:
            raise InvalidProblemError(
                f"Duplicate node id '{node_id}'. Each node must have a unique identifier."
            )
        supply = float(node.get("supply", 0.0))
        node_map[node_id] = Node(id=node_id, supply=supply)

    arc_objs: list[Arc] = []
    for arc in arcs:
        tail = str(arc["tail"])
        head = str(arc["head"])
        capacity_val = arc.get("capacity")
        capacity = float(capacity_val) if capacity_val is not None else None
        cost = float(arc.get("cost", 0.0))
        lower = float(arc.get("lower", 0.0))
        arc_objs.append(Arc(tail=tail, head=head, capacity=capacity, cost=cost, lower=lower))

    problem = NetworkProblem(
        directed=directed,
        nodes=node_map,
        arcs=arc_objs,
        tolerance=float(tolerance),
    )
    problem.validate()
    return problem
