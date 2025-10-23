"""Problem preprocessing utilities for network flow optimization.

This module provides preprocessing techniques to simplify network flow problems
before solving, which can significantly improve performance:

- Remove redundant parallel arcs with identical costs
- Detect disconnected components
- Simplify series arcs (merge consecutive arcs)
- Remove zero-supply nodes with single incident arc

Preprocessing preserves problem semantics while reducing problem size.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import FlowResult, NetworkProblem

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing a network flow problem.

    Attributes:
        problem: The preprocessed problem instance
        removed_arcs: Number of arcs removed during preprocessing
        removed_nodes: Number of nodes removed during preprocessing
        merged_arcs: Number of arc series that were merged
        redundant_arcs: Number of redundant parallel arcs removed
        disconnected_components: Number of disconnected components detected
        preprocessing_time_ms: Time spent preprocessing in milliseconds
        optimizations: Dictionary mapping optimization names to counts
        arc_mapping: Maps original arc keys (tail, head) to preprocessed arc keys.
                     If an arc was merged or removed, it may map to None or a different arc.
        node_mapping: Maps original node IDs to preprocessed node IDs.
                      If a node was removed, it maps to None.
    """

    problem: NetworkProblem
    removed_arcs: int = 0
    removed_nodes: int = 0
    merged_arcs: int = 0
    redundant_arcs: int = 0
    disconnected_components: int = 0
    preprocessing_time_ms: float = 0.0
    optimizations: dict[str, int] = field(default_factory=dict)
    arc_mapping: dict[tuple[str, str], tuple[str, str] | None] = field(default_factory=dict)
    node_mapping: dict[str, str | None] = field(default_factory=dict)


def preprocess_problem(
    problem: NetworkProblem,
    remove_redundant: bool = True,
    detect_disconnected: bool = True,
    simplify_series: bool = True,
    remove_zero_supply: bool = True,
) -> PreprocessingResult:
    """Preprocess a network flow problem to reduce size and improve solving performance.

    Applies a series of optimizations that preserve problem semantics while reducing
    the number of nodes and arcs:

    1. **Remove redundant arcs**: Parallel arcs with identical costs are merged
    2. **Detect disconnected components**: Warns if components can't exchange flow
    3. **Simplify series arcs**: Consecutive arcs through zero-supply nodes are merged
    4. **Remove zero-supply nodes**: Transshipment nodes with single arc are eliminated

    Args:
        problem: The network flow problem to preprocess
        remove_redundant: Remove redundant parallel arcs (default: True)
        detect_disconnected: Detect disconnected components (default: True)
        simplify_series: Simplify series arcs (default: True)
        remove_zero_supply: Remove zero-supply single-arc nodes (default: True)

    Returns:
        PreprocessingResult containing the preprocessed problem and statistics

    Examples:
        >>> from network_solver import build_problem, preprocess_problem
        >>>
        >>> # Problem with redundant arcs
        >>> nodes = [
        ...     {"id": "A", "supply": 100.0},
        ...     {"id": "B", "supply": -100.0},
        ... ]
        >>> arcs = [
        ...     {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
        ...     {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Redundant
        ... ]
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>>
        >>> result = preprocess_problem(problem)
        >>> print(f"Removed {result.redundant_arcs} redundant arcs")
        Removed 1 redundant arcs
        >>> print(f"Reduced from {len(arcs)} to {len(result.problem.arcs)} arcs")
        Reduced from 2 to 1 arcs

    Note:
        - Preprocessing is safe: it preserves optimal solutions
        - Original problem is not modified (creates new problem instance)
        - For large problems, preprocessing can reduce solve time by 50%+
        - Solutions map back to original problem automatically

    See Also:
        - preprocess_and_solve(): Convenience function that preprocesses and solves
        - docs/examples.md#preprocessing: Detailed examples and benchmarks
    """
    import time

    start_time = time.time()

    # Start with a copy of the original problem
    from .data import Arc, NetworkProblem, Node

    nodes = {nid: Node(id=nid, supply=node.supply) for nid, node in problem.nodes.items()}
    arcs = [
        Arc(
            tail=arc.tail,
            head=arc.head,
            capacity=arc.capacity,
            cost=arc.cost,
            lower=arc.lower,
        )
        for arc in problem.arcs
    ]

    preprocessed = NetworkProblem(
        directed=problem.directed,
        nodes=nodes,
        arcs=arcs,
        tolerance=problem.tolerance,
    )

    result = PreprocessingResult(problem=preprocessed)

    # Initialize identity mappings for all original nodes and arcs
    for node_id in problem.nodes:
        result.node_mapping[node_id] = node_id
    for arc in problem.arcs:
        result.arc_mapping[(arc.tail, arc.head)] = (arc.tail, arc.head)

    # Apply optimizations in order
    if remove_redundant:
        count = _remove_redundant_arcs(preprocessed, result.arc_mapping)
        result.redundant_arcs = count
        result.removed_arcs += count
        result.optimizations["redundant_arcs_removed"] = count
        if count > 0:
            logger.info(f"Removed {count} redundant parallel arcs")

    if detect_disconnected:
        count = _detect_disconnected_components(preprocessed)
        result.disconnected_components = count
        result.optimizations["disconnected_components"] = count
        if count > 1:
            logger.warning(f"Detected {count} disconnected components - problem may be infeasible")

    if simplify_series:
        nodes_removed, arcs_merged = _simplify_series_arcs(
            preprocessed, result.arc_mapping, result.node_mapping
        )
        result.removed_nodes += nodes_removed
        result.merged_arcs = arcs_merged
        result.removed_arcs += arcs_merged
        result.optimizations["series_arcs_merged"] = arcs_merged
        result.optimizations["series_nodes_removed"] = nodes_removed
        if nodes_removed > 0:
            logger.info(f"Simplified {arcs_merged} series arcs, removed {nodes_removed} nodes")

    if remove_zero_supply:
        count = _remove_zero_supply_nodes(preprocessed, result.arc_mapping, result.node_mapping)
        result.removed_nodes += count
        result.removed_arcs += count
        result.optimizations["zero_supply_nodes_removed"] = count
        if count > 0:
            logger.info(f"Removed {count} zero-supply transshipment nodes")

    result.preprocessing_time_ms = (time.time() - start_time) * 1000

    # Log summary
    if result.removed_arcs > 0 or result.removed_nodes > 0:
        logger.info(
            f"Preprocessing complete: removed {result.removed_arcs} arcs, "
            f"{result.removed_nodes} nodes in {result.preprocessing_time_ms:.2f}ms"
        )

    return result


def _remove_redundant_arcs(
    problem: NetworkProblem,
    arc_mapping: dict[tuple[str, str], tuple[str, str] | None] | None = None,
) -> int:
    """Remove redundant parallel arcs with identical costs.

    When multiple arcs exist between the same pair of nodes with the same cost,
    they can be merged into a single arc with combined capacity.

    Args:
        problem: Problem to modify in-place
        arc_mapping: Optional mapping to populate with arc transformations

    Returns:
        Number of redundant arcs removed
    """
    # Group arcs by (tail, head, cost, lower)
    arc_groups: dict[tuple[str, str, float, float], list[int]] = defaultdict(list)

    for idx, arc in enumerate(problem.arcs):
        key = (arc.tail, arc.head, arc.cost, arc.lower)
        arc_groups[key].append(idx)

    # Find groups with multiple arcs (redundant)
    redundant_indices = set()
    merged_arcs = []

    for key, indices in arc_groups.items():
        if len(indices) > 1:
            # Merge capacities
            tail, head, cost, lower = key
            total_capacity: float | None

            # If any arc has infinite capacity, result is infinite
            if any(problem.arcs[idx].capacity is None for idx in indices):
                total_capacity = None
            else:
                total_capacity = sum(
                    arc.capacity
                    for arc in (problem.arcs[idx] for idx in indices)
                    if arc.capacity is not None
                )

            # Keep first arc with merged capacity
            from .data import Arc

            merged_arcs.append(
                Arc(tail=tail, head=head, capacity=total_capacity, cost=cost, lower=lower)
            )

            # All redundant arcs map to the same merged arc
            if arc_mapping is not None:
                for _idx in indices:
                    arc_mapping[(tail, head)] = (tail, head)

            # Mark others as redundant
            redundant_indices.update(indices[1:])
        else:
            # Keep single arc as-is
            merged_arcs.append(problem.arcs[indices[0]])

            # Identity mapping for unchanged arcs
            if arc_mapping is not None:
                tail, head = key[0], key[1]
                arc_mapping[(tail, head)] = (tail, head)

    # Update problem arcs
    problem.arcs = merged_arcs

    return len(redundant_indices)


def _detect_disconnected_components(problem: NetworkProblem) -> int:
    """Detect disconnected components in the network graph.

    Uses BFS to find all connected components. If there are multiple components,
    the problem may be infeasible (unless each component is balanced).

    Args:
        problem: Problem to analyze

    Returns:
        Number of disconnected components
    """
    from collections import deque

    # Build adjacency list (undirected for connectivity)
    adj: dict[str, set[str]] = defaultdict(set)
    for arc in problem.arcs:
        adj[arc.tail].add(arc.head)
        adj[arc.head].add(arc.tail)

    # BFS to find components
    visited = set()
    components = []

    for node_id in problem.nodes:
        if node_id in visited:
            continue

        # Start new component
        component = set()
        queue = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()
            component.add(current)

            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return len(components)


def _simplify_series_arcs(
    problem: NetworkProblem,
    arc_mapping: dict[tuple[str, str], tuple[str, str] | None] | None = None,
    node_mapping: dict[str, str | None] | None = None,
) -> tuple[int, int]:
    """Simplify series arcs by merging consecutive arcs through zero-supply nodes.

    If a zero-supply node has exactly one incoming and one outgoing arc,
    they can be merged into a single arc (series reduction).

    This is done iteratively to handle chains of transshipment nodes.

    Args:
        problem: Problem to modify in-place
        arc_mapping: Optional mapping to populate with arc transformations
        node_mapping: Optional mapping to populate with node transformations

    Returns:
        Tuple of (nodes_removed, arcs_merged)
    """
    total_nodes_removed = 0
    total_arcs_merged = 0

    # Iterate until no more simplifications are possible
    while True:
        # Build adjacency lists for current arcs
        incoming: dict[str, list[int]] = defaultdict(list)
        outgoing: dict[str, list[int]] = defaultdict(list)

        for idx, arc in enumerate(problem.arcs):
            outgoing[arc.tail].append(idx)
            incoming[arc.head].append(idx)

        # Find zero-supply nodes with exactly 1 in, 1 out
        # First pass: identify candidates
        candidates = []

        for node_id, node in problem.nodes.items():
            if abs(node.supply) > problem.tolerance:
                continue

            in_arcs = incoming.get(node_id, [])
            out_arcs = outgoing.get(node_id, [])

            if len(in_arcs) == 1 and len(out_arcs) == 1:
                in_idx = in_arcs[0]
                out_idx = out_arcs[0]

                in_arc = problem.arcs[in_idx]
                out_arc = problem.arcs[out_idx]

                # Skip if this would create a self-loop
                if in_arc.tail == out_arc.head:
                    continue

                candidates.append((node_id, in_idx, out_idx, in_arc, out_arc))

        # Second pass: greedily select nodes to merge, ensuring we don't create
        # arcs that reference nodes being removed in THIS iteration
        nodes_being_merged = set()
        nodes_to_remove = []
        arcs_to_remove = set()
        merged_arcs = []

        for node_id, in_idx, out_idx, in_arc, out_arc in candidates:
            # Skip if predecessor or successor is being merged in THIS iteration
            if in_arc.tail in nodes_being_merged or out_arc.head in nodes_being_merged:
                continue

            # Merge: create arc from in_arc.tail to out_arc.head
            # Cost: sum of costs
            # Capacity: min of capacities
            # Lower: max of lower bounds (conservative)

            if in_arc.capacity is None:
                merged_capacity = out_arc.capacity
            elif out_arc.capacity is None:
                merged_capacity = in_arc.capacity
            else:
                merged_capacity = min(in_arc.capacity, out_arc.capacity)

            merged_cost = in_arc.cost + out_arc.cost
            merged_lower = max(in_arc.lower, out_arc.lower)

            from .data import Arc

            merged_arcs.append(
                Arc(
                    tail=in_arc.tail,
                    head=out_arc.head,
                    capacity=merged_capacity,
                    cost=merged_cost,
                    lower=merged_lower,
                )
            )

            # Track mappings: both original arcs map to the merged arc
            if arc_mapping is not None:
                arc_mapping[(in_arc.tail, in_arc.head)] = (in_arc.tail, out_arc.head)
                arc_mapping[(out_arc.tail, out_arc.head)] = (in_arc.tail, out_arc.head)

            # Track node removal
            if node_mapping is not None:
                node_mapping[node_id] = None

            nodes_to_remove.append(node_id)
            nodes_being_merged.add(node_id)
            arcs_to_remove.add(in_idx)
            arcs_to_remove.add(out_idx)

        # If no nodes to remove, we're done
        if not nodes_to_remove:
            break

        # Remove nodes
        nodes_removed_set = set(nodes_to_remove)
        for node_id in nodes_to_remove:
            del problem.nodes[node_id]

        # Keep only arcs that don't reference removed nodes
        # This ensures we don't keep stray arcs that reference deleted nodes
        kept_arcs = []
        for idx, arc in enumerate(problem.arcs):
            # Skip arcs we're explicitly removing (will be replaced by merged arcs)
            if idx in arcs_to_remove:
                continue
            # Skip arcs that reference nodes we're removing
            if arc.tail in nodes_removed_set or arc.head in nodes_removed_set:
                continue
            kept_arcs.append(arc)

        # Add merged arcs
        problem.arcs = kept_arcs + merged_arcs

        total_nodes_removed += len(nodes_to_remove)
        total_arcs_merged += len(nodes_to_remove)

    return total_nodes_removed, total_arcs_merged


def _remove_zero_supply_nodes(
    problem: NetworkProblem,
    arc_mapping: dict[tuple[str, str], tuple[str, str] | None] | None = None,
    node_mapping: dict[str, str | None] | None = None,
) -> int:
    """Remove zero-supply nodes with exactly one incident arc.

    If a zero-supply node has only one arc (either incoming or outgoing),
    it can be removed along with the arc (no flow can pass through).

    Args:
        problem: Problem to modify in-place
        arc_mapping: Optional mapping to populate with arc transformations
        node_mapping: Optional mapping to populate with node transformations

    Returns:
        Number of nodes removed
    """
    # Build adjacency lists
    incident: dict[str, list[int]] = defaultdict(list)

    for idx, arc in enumerate(problem.arcs):
        incident[arc.tail].append(idx)
        incident[arc.head].append(idx)

    # Find zero-supply nodes with exactly 1 incident arc
    nodes_to_remove = []
    arcs_to_remove = set()

    for node_id, node in problem.nodes.items():
        if abs(node.supply) > problem.tolerance:
            continue

        inc_arcs = incident[node_id]

        if len(inc_arcs) == 1:
            nodes_to_remove.append(node_id)
            arcs_to_remove.add(inc_arcs[0])

    if not nodes_to_remove:
        return 0

    # Track mappings before removing
    if node_mapping is not None:
        for node_id in nodes_to_remove:
            node_mapping[node_id] = None

    if arc_mapping is not None:
        for idx in arcs_to_remove:
            arc = problem.arcs[idx]
            arc_mapping[(arc.tail, arc.head)] = None

    # Remove nodes
    for node_id in nodes_to_remove:
        del problem.nodes[node_id]

    # Remove arcs
    problem.arcs = [arc for idx, arc in enumerate(problem.arcs) if idx not in arcs_to_remove]

    return len(nodes_to_remove)


def translate_result(
    flow_result: FlowResult,
    preproc_result: PreprocessingResult,
    original_problem: NetworkProblem,
) -> FlowResult:
    """Translate solution from preprocessed problem back to original problem.

    Takes a solution on the preprocessed problem and maps it back to the original
    problem's node and arc structure using the arc_mapping and node_mapping from
    preprocessing.

    Args:
        flow_result: FlowResult from solving the preprocessed problem
        preproc_result: PreprocessingResult containing the mappings
        original_problem: The original problem before preprocessing

    Returns:
        FlowResult mapped to the original problem's structure

    Note:
        - Removed arcs will have flow = 0
        - Merged arcs will share the flow from the merged arc (distributed by capacity for redundant arcs)
        - Removed nodes will have duals computed from adjacent preserved nodes using dual feasibility
        - The basis cannot be translated back and will be set to None
    """
    from .data import FlowResult

    # Create reverse mapping: preprocessed arc -> list of original arcs
    reverse_arc_mapping: dict[tuple[str, str] | None, list[tuple[str, str]]] = defaultdict(list)
    for orig_arc, preprocessed_arc in preproc_result.arc_mapping.items():
        reverse_arc_mapping[preprocessed_arc].append(orig_arc)

    # Translate flows: map from preprocessed arcs to original arcs
    translated_flows: dict[tuple[str, str], float] = {}

    for orig_arc in original_problem.arcs:
        arc_key = (orig_arc.tail, orig_arc.head)
        preprocessed_arc_key = preproc_result.arc_mapping.get(arc_key)

        if preprocessed_arc_key is None:
            # Arc was removed - no flow
            translated_flows[arc_key] = 0.0
        else:
            # Get flow from preprocessed arc
            preprocessed_flow = flow_result.flows.get(preprocessed_arc_key, 0.0)

            # For redundant arcs that were merged, we need to distribute the flow
            # Get all original arcs that map to this preprocessed arc
            original_arcs_sharing_flow = reverse_arc_mapping[preprocessed_arc_key]

            # Check if these are redundant parallel arcs (same tail and head as preprocessed arc)
            # vs series arcs that were merged into a different arc
            is_redundant_merge = all(
                oa == preprocessed_arc_key for oa in original_arcs_sharing_flow
            )

            if len(original_arcs_sharing_flow) > 1 and is_redundant_merge:
                # Multiple REDUNDANT arcs merged - distribute flow proportionally by capacity
                # Get capacities of all arcs that map to this preprocessed arc
                capacities = []
                for oa in original_arcs_sharing_flow:
                    orig_arc_obj = next(
                        (a for a in original_problem.arcs if (a.tail, a.head) == oa),
                        None,
                    )
                    if orig_arc_obj:
                        capacities.append(
                            orig_arc_obj.capacity
                            if orig_arc_obj.capacity is not None
                            else float("inf")
                        )

                # If all have infinite capacity, distribute equally
                if all(c == float("inf") for c in capacities):
                    translated_flows[arc_key] = preprocessed_flow / len(original_arcs_sharing_flow)
                else:
                    # Distribute proportionally by capacity (excluding infinite)
                    finite_capacities = [c if c != float("inf") else 0 for c in capacities]
                    total_capacity = sum(finite_capacities)
                    if total_capacity > 0:
                        my_capacity = capacities[original_arcs_sharing_flow.index(arc_key)]
                        if my_capacity == float("inf"):
                            my_capacity = 0
                        translated_flows[arc_key] = preprocessed_flow * (
                            my_capacity / total_capacity
                        )
                    else:
                        translated_flows[arc_key] = 0.0
            else:
                # Single arc OR series arc merged - use the same flow
                # For series arcs, all arcs in the series carry the same flow
                translated_flows[arc_key] = preprocessed_flow

    # Translate duals: include preserved nodes and compute duals for removed nodes
    translated_duals: dict[str, float] = {}

    # First, populate duals for preserved nodes
    for node_id in original_problem.nodes:
        preprocessed_node_id = preproc_result.node_mapping.get(node_id)

        if preprocessed_node_id is not None:
            # Node was preserved - use its dual
            translated_duals[node_id] = flow_result.duals.get(preprocessed_node_id, 0.0)

    # Second, compute duals for removed nodes based on adjacent preserved nodes
    # Using dual feasibility: dual_j - dual_i <= c_ij (for arc i->j)
    for node_id in original_problem.nodes:
        preprocessed_node_id = preproc_result.node_mapping.get(node_id)

        if preprocessed_node_id is None:
            # Node was removed - compute its dual from adjacent nodes
            # Find all arcs incident to this node in the original problem
            incoming_arcs = [a for a in original_problem.arcs if a.head == node_id]
            outgoing_arcs = [a for a in original_problem.arcs if a.tail == node_id]

            # Collect constraints from adjacent preserved nodes
            dual_constraints = []

            for arc in incoming_arcs:
                if arc.tail in translated_duals:
                    # For arc tail->node_id with cost c:
                    # dual[node_id] - dual[tail] <= c
                    # Therefore: dual[node_id] <= dual[tail] + c
                    dual_constraints.append(translated_duals[arc.tail] + arc.cost)

            for arc in outgoing_arcs:
                if arc.head in translated_duals:
                    # For arc node_id->head with cost c:
                    # dual[head] - dual[node_id] <= c
                    # Therefore: dual[node_id] >= dual[head] - c
                    dual_constraints.append(translated_duals[arc.head] - arc.cost)

            # If we have constraints, use the average to satisfy them approximately
            # This is a heuristic that tries to be "centered" among the constraints
            if dual_constraints:
                translated_duals[node_id] = sum(dual_constraints) / len(dual_constraints)
            else:
                # No adjacent preserved nodes - default to 0
                translated_duals[node_id] = 0.0

    # Create translated result
    # Note: basis cannot be translated back, so we set it to None
    return FlowResult(
        objective=flow_result.objective,
        flows=translated_flows,
        status=flow_result.status,
        iterations=flow_result.iterations,
        duals=translated_duals,
        basis=None,  # Cannot translate basis back to original problem
    )


def preprocess_and_solve(
    problem: NetworkProblem, **solve_kwargs: Any
) -> tuple[PreprocessingResult, FlowResult]:
    """Convenience function to preprocess and solve in one call.

    Args:
        problem: Network flow problem to solve
        **solve_kwargs: Additional arguments passed to solve_min_cost_flow()

    Returns:
        Tuple of (preprocessing_result, flow_result)

        The flow_result is automatically translated back to the original problem's
        structure, so arc flows and node duals correspond to the original problem.

    Warning:
        If preprocessing modifies the problem structure (removes/merges arcs or nodes),
        warm_start_basis from the original problem will be incompatible with the
        preprocessed problem. This function will automatically drop warm_start_basis
        if preprocessing made structural changes.

    Note:
        The returned flow_result will have basis=None because the basis cannot be
        reliably translated back to the original problem structure.

    Example:
        >>> from network_solver import build_problem, preprocess_and_solve
        >>>
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>> preproc_result, flow_result = preprocess_and_solve(problem)
        >>>
        >>> print(f"Removed {preproc_result.removed_arcs} arcs")
        >>> print(f"Optimal cost: ${flow_result.objective:.2f}")
        >>> # flow_result.flows keys correspond to original problem arcs
    """
    from .solver import solve_min_cost_flow

    preproc_result = preprocess_problem(problem)

    # Check if preprocessing made structural changes
    structural_changes = (
        preproc_result.removed_arcs > 0
        or preproc_result.removed_nodes > 0
        or preproc_result.merged_arcs > 0
    )

    # Guard against incompatible warm_start_basis
    if structural_changes and "warm_start_basis" in solve_kwargs:
        logger.warning(
            "Dropping warm_start_basis: preprocessing made structural changes "
            f"(removed {preproc_result.removed_arcs} arcs, "
            f"{preproc_result.removed_nodes} nodes, "
            f"merged {preproc_result.merged_arcs} arc series). "
            "Basis from original problem is incompatible with preprocessed problem."
        )
        # Intentionally create a new dictionary to avoid mutating the caller's solve_kwargs.
        solve_kwargs = {k: v for k, v in solve_kwargs.items() if k != "warm_start_basis"}

    flow_result = solve_min_cost_flow(preproc_result.problem, **solve_kwargs)

    # Translate result back to original problem structure
    if structural_changes:
        flow_result = translate_result(flow_result, preproc_result, problem)

    return preproc_result, flow_result
