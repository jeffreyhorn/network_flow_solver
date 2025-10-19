"""Core data structures for linear network programming problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .exceptions import InvalidProblemError


@dataclass(frozen=True)
class Node:
    """Represents a network node with a supply (positive) or demand (negative)."""

    id: str
    supply: float = 0.0


@dataclass(frozen=True)
class Arc:
    """Represents a directed arc with a capacity, cost, and optional lower bound."""

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
    """Encapsulates a minimum cost flow problem."""

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
class FlowResult:
    """Represents the output of a flow computation.

    Attributes:
        objective: The objective function value (total cost).
        flows: Dictionary mapping arc (tail, head) tuples to flow values.
        status: Solution status ('optimal', 'infeasible', 'unbounded', 'iteration_limit').
        iterations: Number of simplex iterations performed.
        duals: Dictionary mapping node IDs to dual values (node potentials).
               For optimal solutions, these represent shadow prices for supply/demand constraints.
               Useful for sensitivity analysis.
    """

    objective: float
    flows: dict[tuple[str, str], float] = field(default_factory=dict)
    status: str = "optimal"
    iterations: int = 0
    duals: dict[str, float] = field(default_factory=dict)


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
        max_iterations: Maximum number of simplex iterations. If None, defaults to max(100, 5*num_arcs).
        tolerance: Numerical tolerance for feasibility and optimality checks (default: 1e-6).
        pricing_strategy: Arc pricing strategy - "devex" (default) or "dantzig" (first eligible).
        block_size: Number of arcs to examine per pricing block. If None, defaults to num_arcs/8.
        ft_update_limit: Maximum number of Forrest-Tomlin basis updates before rebuild (default: 64).
    """

    max_iterations: int | None = None
    tolerance: float = 1e-6
    pricing_strategy: str = "devex"
    block_size: int | None = None
    ft_update_limit: int = 64

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
        if self.block_size is not None and self.block_size <= 0:
            raise InvalidProblemError(
                f"Block size must be positive, got {self.block_size}. "
                f"Block size controls how many arcs are examined per pricing iteration."
            )
        if self.ft_update_limit <= 0:
            raise InvalidProblemError(
                f"FT update limit must be positive, got {self.ft_update_limit}. "
                f"This controls how often the basis factorization is rebuilt."
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
