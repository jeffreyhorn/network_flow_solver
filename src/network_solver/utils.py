"""Utility functions for analyzing and validating network flow solutions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .data import FlowResult, NetworkProblem


@dataclass
class FlowPath:
    """Represents a path from source to target with flow.

    Attributes:
        nodes: Sequence of node IDs from source to target.
        arcs: Sequence of (tail, head) arc tuples along the path.
        flow: Flow value along the path (minimum flow on any arc).
        cost: Total cost along the path (sum of arc costs times flow).
    """

    nodes: list[str]
    arcs: list[tuple[str, str]]
    flow: float
    cost: float


@dataclass
class ValidationResult:
    """Results from validating a flow solution.

    Attributes:
        is_valid: True if the flow satisfies all constraints.
        errors: List of validation error messages (empty if valid).
        flow_balance: Dict mapping node IDs to net flow (supply - demand - outflow + inflow).
        capacity_violations: List of arcs exceeding capacity.
        lower_bound_violations: List of arcs below lower bound.
    """

    is_valid: bool
    errors: list[str]
    flow_balance: dict[str, float]
    capacity_violations: list[tuple[str, str]]
    lower_bound_violations: list[tuple[str, str]]


@dataclass
class BottleneckArc:
    """Represents an arc at or near capacity that limits flow.

    Attributes:
        tail: Source node ID.
        head: Destination node ID.
        flow: Current flow on the arc.
        capacity: Arc capacity (None if infinite).
        utilization: Flow / capacity (1.0 = at capacity, None for infinite capacity).
        cost: Cost per unit of flow.
        slack: Remaining capacity (capacity - flow).
    """

    tail: str
    head: str
    flow: float
    capacity: float | None
    utilization: float | None
    cost: float
    slack: float


def extract_path(
    result: FlowResult,
    problem: NetworkProblem,
    source: str,
    target: str,
    tolerance: float = 1e-6,
) -> FlowPath | None:
    """Extract a flow-carrying path from source to target.

    Uses breadth-first search to find a path from source to target that carries
    positive flow in the solution. If multiple paths exist, returns one of them
    (not necessarily the one with maximum flow).

    Args:
        result: Solution containing flows on arcs.
        problem: Original problem definition with arc costs.
        source: Starting node ID.
        target: Ending node ID.
        tolerance: Minimum flow value to consider an arc active (default: 1e-6).

    Returns:
        FlowPath object if a path exists, None otherwise.

    Raises:
        ValueError: If source or target node doesn't exist in the problem.
    """
    if source not in problem.nodes:
        raise ValueError(f"Source node '{source}' not found in problem")
    if target not in problem.nodes:
        raise ValueError(f"Target node '{target}' not found in problem")

    if source == target:
        return FlowPath(nodes=[source], arcs=[], flow=0.0, cost=0.0)

    # Build adjacency list from flows
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for (tail, head), flow in result.flows.items():
        if flow > tolerance:
            if tail not in adjacency:
                adjacency[tail] = []
            adjacency[tail].append((head, flow))

    # BFS to find path
    queue: deque[str] = deque([source])
    parent: dict[str, tuple[str, float]] = {source: (source, 0.0)}

    while queue:
        node = queue.popleft()
        if node == target:
            break
        if node not in adjacency:
            continue
        for neighbor, flow in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = (node, flow)
                queue.append(neighbor)

    if target not in parent:
        return None

    # Reconstruct path
    path_nodes: list[str] = []
    path_arcs: list[tuple[str, str]] = []
    current = target

    while current != source:
        path_nodes.append(current)
        prev, _ = parent[current]
        path_arcs.append((prev, current))
        current = prev

    path_nodes.append(source)
    path_nodes.reverse()
    path_arcs.reverse()

    # Find minimum flow along path
    min_flow = float("inf")
    for tail, head in path_arcs:
        flow = result.flows.get((tail, head), 0.0)
        min_flow = min(min_flow, flow)

    # Calculate total cost
    arc_costs: dict[tuple[str, str], float] = {}
    for arc in problem.arcs:
        arc_costs[(arc.tail, arc.head)] = arc.cost

    total_cost = 0.0
    for tail, head in path_arcs:
        cost = arc_costs.get((tail, head), 0.0)
        total_cost += cost * min_flow

    return FlowPath(
        nodes=path_nodes,
        arcs=path_arcs,
        flow=min_flow,
        cost=total_cost,
    )


def validate_flow(
    problem: NetworkProblem,
    result: FlowResult,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate that a flow solution satisfies all problem constraints.

    Checks:
    - Flow conservation at each node (inflow - outflow = supply)
    - Capacity constraints (flow <= capacity for each arc)
    - Lower bound constraints (flow >= lower for each arc)

    Args:
        problem: Problem definition with nodes, arcs, and constraints.
        result: Solution to validate.
        tolerance: Numerical tolerance for constraint violations (default: 1e-6).

    Returns:
        ValidationResult with detailed information about any violations.
    """
    errors: list[str] = []
    flow_balance: dict[str, float] = {}
    capacity_violations: list[tuple[str, str]] = []
    lower_bound_violations: list[tuple[str, str]] = []

    # Initialize flow balance with node supplies
    for node_id, node in problem.nodes.items():
        flow_balance[node_id] = node.supply

    # Build arc map for quick lookup - use expanded arcs for undirected problems
    arc_map: dict[tuple[str, str], tuple[float | None, float]] = {}
    expanded_arcs = problem.undirected_expansion()
    for arc in expanded_arcs:
        arc_map[(arc.tail, arc.head)] = (arc.capacity, arc.lower)

    # Process each arc in the solution
    for (tail, head), flow in result.flows.items():
        # Update flow balance
        if tail in flow_balance:
            flow_balance[tail] -= flow
        else:
            flow_balance[tail] = -flow

        if head in flow_balance:
            flow_balance[head] += flow
        else:
            flow_balance[head] = flow

        # Check capacity constraint
        if (tail, head) in arc_map:
            capacity, lower = arc_map[(tail, head)]

            if capacity is not None and flow > capacity + tolerance:
                capacity_violations.append((tail, head))
                errors.append(
                    f"Arc ({tail}, {head}): flow {flow:.6f} exceeds capacity {capacity:.6f}"
                )

            # Check lower bound constraint
            if flow < lower - tolerance:
                lower_bound_violations.append((tail, head))
                errors.append(
                    f"Arc ({tail}, {head}): flow {flow:.6f} below lower bound {lower:.6f}"
                )

    # Check flow conservation at each node
    for node_id, balance in flow_balance.items():
        if abs(balance) > tolerance:
            errors.append(f"Node {node_id}: flow imbalance {balance:.6f} (should be zero)")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        flow_balance=flow_balance,
        capacity_violations=capacity_violations,
        lower_bound_violations=lower_bound_violations,
    )


def compute_bottleneck_arcs(
    problem: NetworkProblem,
    result: FlowResult,
    threshold: float = 0.95,
    tolerance: float = 1e-6,
) -> list[BottleneckArc]:
    """Identify arcs that are at or near capacity (bottlenecks).

    Bottleneck arcs limit the amount of flow through the network. Increasing
    their capacity would allow more flow and potentially reduce costs.

    Args:
        problem: Problem definition with arc capacities and costs.
        result: Solution with flow values.
        threshold: Minimum utilization to consider an arc a bottleneck (default: 0.95 = 95%).
        tolerance: Minimum flow to consider an arc active (default: 1e-6).

    Returns:
        List of BottleneckArc objects sorted by utilization (descending).
        Arcs with infinite capacity are excluded.
    """
    # Build arc map for quick lookup - use expanded arcs for undirected problems
    arc_map: dict[tuple[str, str], tuple[float | None, float]] = {}
    expanded_arcs = problem.undirected_expansion()
    for arc in expanded_arcs:
        arc_map[(arc.tail, arc.head)] = (arc.capacity, arc.cost)

    bottlenecks: list[BottleneckArc] = []

    for (tail, head), flow in result.flows.items():
        if flow < tolerance:
            continue

        if (tail, head) not in arc_map:
            continue

        capacity, cost = arc_map[(tail, head)]

        # Skip infinite capacity arcs
        if capacity is None:
            continue

        utilization = flow / capacity if capacity > 0 else 0.0

        if utilization >= threshold:
            slack = capacity - flow
            bottlenecks.append(
                BottleneckArc(
                    tail=tail,
                    head=head,
                    flow=flow,
                    capacity=capacity,
                    utilization=utilization,
                    cost=cost,
                    slack=slack,
                )
            )

    # Sort by utilization (highest first), then by slack (lowest first)
    bottlenecks.sort(key=lambda x: (-x.utilization, x.slack))

    return bottlenecks
