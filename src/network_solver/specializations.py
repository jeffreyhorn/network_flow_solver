"""Network problem specialization detection and optimization.

This module provides utilities to detect special network structures
(transportation problems, bipartite matching, assignment problems, etc.)
and apply specialized solution strategies for improved performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import NetworkProblem


class NetworkType(Enum):
    """Types of specialized network flow problems."""

    GENERAL = "general"
    TRANSPORTATION = "transportation"
    ASSIGNMENT = "assignment"
    BIPARTITE_MATCHING = "bipartite_matching"
    MAX_FLOW = "max_flow"
    SHORTEST_PATH = "shortest_path"


@dataclass
class NetworkStructure:
    """Detected structure and properties of a network problem.

    Attributes:
        network_type: The specialized type detected
        is_bipartite: Whether the graph is bipartite
        source_nodes: Set of source node IDs (supply > 0)
        sink_nodes: Set of sink node IDs (supply < 0)
        transshipment_nodes: Set of transshipment node IDs (supply = 0)
        partitions: For bipartite graphs, the two partitions
        total_supply: Total supply in the network
        total_demand: Total demand (absolute value)
        is_balanced: Whether supply equals demand
        has_lower_bounds: Whether any arcs have non-zero lower bounds
        has_finite_capacities: Whether all capacities are finite
    """

    network_type: NetworkType
    is_bipartite: bool
    source_nodes: set[str]
    sink_nodes: set[str]
    transshipment_nodes: set[str]
    partitions: tuple[set[str], set[str]] | None
    total_supply: float
    total_demand: float
    is_balanced: bool
    has_lower_bounds: bool
    has_finite_capacities: bool


def analyze_network_structure(problem: NetworkProblem) -> NetworkStructure:
    """Analyze a network problem to detect its structure and type.

    Args:
        problem: The network flow problem to analyze

    Returns:
        NetworkStructure describing the detected properties

    Examples:
        >>> problem = build_problem(...)
        >>> structure = analyze_network_structure(problem)
        >>> if structure.network_type == NetworkType.TRANSPORTATION:
        ...     print("Detected transportation problem!")
    """
    # Categorize nodes by supply/demand
    source_nodes = set()
    sink_nodes = set()
    transshipment_nodes = set()
    total_supply = 0.0
    total_demand = 0.0

    for node_id, node in problem.nodes.items():
        if node.supply > problem.tolerance:
            source_nodes.add(node_id)
            total_supply += node.supply
        elif node.supply < -problem.tolerance:
            sink_nodes.add(node_id)
            total_demand += abs(node.supply)
        else:
            transshipment_nodes.add(node_id)

    is_balanced = abs(total_supply - total_demand) <= problem.tolerance

    # Check for lower bounds and capacity constraints
    has_lower_bounds = any(arc.lower > problem.tolerance for arc in problem.arcs)
    has_finite_capacities = all(
        arc.capacity is not None and arc.capacity < float("inf") for arc in problem.arcs
    )

    # Check if graph is bipartite
    is_bipartite, partitions = _check_bipartite(problem)

    # Detect specialized problem type
    network_type = _detect_network_type(
        problem,
        source_nodes,
        sink_nodes,
        transshipment_nodes,
        is_bipartite,
        has_lower_bounds,
        is_balanced,
    )

    return NetworkStructure(
        network_type=network_type,
        is_bipartite=is_bipartite,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        transshipment_nodes=transshipment_nodes,
        partitions=partitions,
        total_supply=total_supply,
        total_demand=total_demand,
        is_balanced=is_balanced,
        has_lower_bounds=has_lower_bounds,
        has_finite_capacities=has_finite_capacities,
    )


def _check_bipartite(problem: NetworkProblem) -> tuple[bool, tuple[set[str], set[str]] | None]:
    """Check if the network graph is bipartite using BFS coloring.

    Args:
        problem: The network problem

    Returns:
        Tuple of (is_bipartite, partitions) where partitions is None if not bipartite
    """
    if not problem.nodes:
        return False, None

    # Build adjacency list (undirected for bipartite check)
    adj: dict[str, set[str]] = {node_id: set() for node_id in problem.nodes}

    for arc in problem.arcs:
        adj[arc.tail].add(arc.head)
        if not problem.directed:
            adj[arc.head].add(arc.tail)
        # For directed graphs, also add reverse edges for bipartite check
        else:
            adj[arc.head].add(arc.tail)

    # Try to 2-color the graph using BFS
    color: dict[str, int] = {}
    partition_0: set[str] = set()
    partition_1: set[str] = set()

    # Handle disconnected components
    for start_node in problem.nodes:
        if start_node in color:
            continue

        # BFS from this node
        queue = [start_node]
        color[start_node] = 0
        partition_0.add(start_node)

        while queue:
            node = queue.pop(0)
            node_color = color[node]
            next_color = 1 - node_color

            for neighbor in adj[node]:
                if neighbor not in color:
                    color[neighbor] = next_color
                    if next_color == 0:
                        partition_0.add(neighbor)
                    else:
                        partition_1.add(neighbor)
                    queue.append(neighbor)
                elif color[neighbor] != next_color:
                    # Found odd cycle - not bipartite
                    return False, None

    return True, (partition_0, partition_1)


def _detect_network_type(
    problem: NetworkProblem,
    source_nodes: set[str],
    sink_nodes: set[str],
    transshipment_nodes: set[str],
    is_bipartite: bool,
    has_lower_bounds: bool,
    is_balanced: bool,
) -> NetworkType:
    """Detect the specialized type of network problem.

    Args:
        problem: The network problem
        source_nodes: Set of source nodes
        sink_nodes: Set of sink nodes
        transshipment_nodes: Set of transshipment nodes
        is_bipartite: Whether graph is bipartite
        has_lower_bounds: Whether problem has lower bounds
        is_balanced: Whether supply equals demand

    Returns:
        The detected NetworkType
    """
    num_sources = len(source_nodes)
    num_sinks = len(sink_nodes)
    num_transship = len(transshipment_nodes)

    # Check for transportation problem:
    # - Only sources and sinks (no transshipment nodes)
    # - Bipartite graph
    # - All arcs from sources to sinks
    # - No lower bounds
    if (
        num_transship == 0
        and num_sources > 0
        and num_sinks > 0
        and is_bipartite
        and not has_lower_bounds
    ):
        # Verify all arcs go from source to sink
        all_source_to_sink = all(
            arc.tail in source_nodes and arc.head in sink_nodes for arc in problem.arcs
        )
        if all_source_to_sink:
            # Check if it's an assignment problem (special case of transportation)
            # Assignment: each source/sink has supply/demand of 1, and it's balanced
            if is_balanced:
                all_unit_supply = all(
                    abs(problem.nodes[node].supply - 1.0) <= problem.tolerance
                    for node in source_nodes
                )
                all_unit_demand = all(
                    abs(problem.nodes[node].supply + 1.0) <= problem.tolerance
                    for node in sink_nodes
                )
                if all_unit_supply and all_unit_demand and num_sources == num_sinks:
                    return NetworkType.ASSIGNMENT

            return NetworkType.TRANSPORTATION

    # Check for shortest path FIRST (more specific than bipartite matching):
    # - Single source with supply = 1, single sink with demand = -1
    # - All other nodes are transshipment
    # - Unit flow problem
    if num_sources == 1 and num_sinks == 1:
        source_node = next(iter(source_nodes))
        sink_node = next(iter(sink_nodes))
        if (
            abs(problem.nodes[source_node].supply - 1.0) <= problem.tolerance
            and abs(problem.nodes[sink_node].supply + 1.0) <= problem.tolerance
        ):
            return NetworkType.SHORTEST_PATH

    # Check for bipartite matching:
    # - Bipartite graph
    # - All supplies/demands are -1, 0, or 1
    # - Typically for finding maximum matchings
    if is_bipartite and not has_lower_bounds:
        all_unit_values = all(
            abs(abs(node.supply) - 1.0) <= problem.tolerance
            or abs(node.supply) <= problem.tolerance
            for node in problem.nodes.values()
        )
        if all_unit_values:
            return NetworkType.BIPARTITE_MATCHING

    # Check for max flow problem:
    # - Single source, single sink
    # - All other nodes are transshipment
    # - No costs matter (or all zero/unit costs)
    # - Only if NOT already detected as transportation/shortest path
    # - With varying costs and transshipment, it's just a general problem
    if num_sources == 1 and num_sinks == 1 and not has_lower_bounds:
        all_zero_cost = all(abs(arc.cost) <= problem.tolerance for arc in problem.arcs)
        all_unit_cost = all(abs(arc.cost - 1.0) <= problem.tolerance for arc in problem.arcs)
        # Only classify as max flow if costs are clearly uniform (all 0 or all 1)
        # and we don't have transshipment nodes with varying costs
        if all_zero_cost or all_unit_cost:
            return NetworkType.MAX_FLOW

    # Default: general network flow
    return NetworkType.GENERAL


def get_specialization_info(structure: NetworkStructure) -> dict[str, Any]:
    """Get human-readable information about detected specialization.

    Args:
        structure: The analyzed network structure

    Returns:
        Dictionary with specialization details
    """
    info = {
        "type": structure.network_type.value,
        "is_bipartite": structure.is_bipartite,
        "num_sources": len(structure.source_nodes),
        "num_sinks": len(structure.sink_nodes),
        "num_transshipment": len(structure.transshipment_nodes),
        "is_balanced": structure.is_balanced,
        "total_supply": structure.total_supply,
        "total_demand": structure.total_demand,
        "has_lower_bounds": structure.has_lower_bounds,
        "has_finite_capacities": structure.has_finite_capacities,
    }

    # Add type-specific information
    if structure.network_type == NetworkType.TRANSPORTATION:
        info["description"] = (
            f"Transportation problem: {len(structure.source_nodes)} sources → "
            f"{len(structure.sink_nodes)} sinks"
        )
        info["optimization_hint"] = "Can use specialized transportation simplex algorithm"

    elif structure.network_type == NetworkType.ASSIGNMENT:
        info["description"] = (
            f"Assignment problem: {len(structure.source_nodes)} workers ↔ "
            f"{len(structure.sink_nodes)} jobs"
        )
        info["optimization_hint"] = "Can use Hungarian algorithm or specialized simplex"

    elif structure.network_type == NetworkType.BIPARTITE_MATCHING:
        if structure.partitions:
            p0, p1 = structure.partitions
            info["description"] = f"Bipartite matching: |U|={len(p0)}, |V|={len(p1)}"
        else:
            info["description"] = "Bipartite matching problem"
        info["optimization_hint"] = "Can use Hopcroft-Karp or augmenting path algorithms"

    elif structure.network_type == NetworkType.MAX_FLOW:
        info["description"] = "Maximum flow problem (single source/sink)"
        info["optimization_hint"] = "Can use push-relabel or Dinic's algorithm"

    elif structure.network_type == NetworkType.SHORTEST_PATH:
        info["description"] = "Shortest path problem (unit flow)"
        info["optimization_hint"] = "Can use Dijkstra or Bellman-Ford algorithm"

    else:
        info["description"] = "General minimum-cost flow problem"
        info["optimization_hint"] = "Using network simplex (optimal for general case)"

    return info
