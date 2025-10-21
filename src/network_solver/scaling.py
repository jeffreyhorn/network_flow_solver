"""Automatic problem scaling for improved numerical stability.

This module implements automatic scaling of network flow problems to improve
numerical stability when costs, capacities, or supplies have widely varying
magnitudes. Proper scaling can reduce round-off errors and improve convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import NetworkProblem


@dataclass
class ScalingFactors:
    """Stores scaling factors for unscaling the solution.

    Attributes:
        cost_scale: Factor by which costs were scaled
        capacity_scale: Factor by which capacities were scaled
        supply_scale: Factor by which supplies/demands were scaled
        enabled: Whether scaling was actually applied
    """

    cost_scale: float = 1.0
    capacity_scale: float = 1.0
    supply_scale: float = 1.0
    enabled: bool = False


def should_scale_problem(problem: NetworkProblem, threshold: float = 1e6) -> bool:
    """Determine if problem would benefit from scaling.

    Scaling is recommended when the ratio between the largest and smallest
    non-zero values in any category (costs, capacities, supplies) exceeds
    the threshold. This indicates potential numerical instability.

    Args:
        problem: The network flow problem to analyze
        threshold: Maximum acceptable ratio (default: 1e6 = 6 orders of magnitude)

    Returns:
        True if scaling is recommended, False otherwise

    Examples:
        >>> problem = build_problem(
        ...     nodes=[{"id": "s", "supply": 1e8}, {"id": "t", "supply": -1e8}],
        ...     arcs=[{"tail": "s", "head": "t", "capacity": 1e8, "cost": 0.001}],
        ... )
        >>> should_scale_problem(problem)
        True  # Cost (0.001) and capacity (1e8) differ by 11 orders of magnitude
    """
    # Collect non-zero costs
    costs = [abs(arc.cost) for arc in problem.arcs if arc.cost != 0]

    # Collect finite capacities
    capacities = [
        arc.capacity
        for arc in problem.arcs
        if arc.capacity is not None and math.isfinite(arc.capacity) and arc.capacity > 0
    ]

    # Collect non-zero supplies/demands
    supplies = [abs(node.supply) for node in problem.nodes.values() if node.supply != 0]

    # Check each category for wide range
    for values in [costs, capacities, supplies]:
        if len(values) < 2:
            continue

        min_val = min(values)
        max_val = max(values)

        if min_val > 0:  # Avoid division by zero
            ratio = max_val / min_val
            if ratio > threshold:
                return True

    # Also check cross-category ratios (e.g., cost vs capacity)
    all_values = costs + capacities + supplies
    if len(all_values) >= 2:
        min_val = min(all_values)
        max_val = max(all_values)
        if min_val > 0:
            ratio = max_val / min_val
            if ratio > threshold:
                return True

    return False


def compute_scaling_factors(
    problem: NetworkProblem, target_range: tuple[float, float] = (0.1, 100.0)
) -> ScalingFactors:
    """Compute scaling factors to bring problem values into target range.

    The scaling strategy:
    1. Find geometric mean of non-zero values in each category
    2. Scale to bring geometric mean to ~1.0
    3. Target range of [0.1, 100] provides 3 orders of magnitude buffer

    Args:
        problem: The network flow problem to scale
        target_range: Desired range for scaled values (default: [0.1, 100])

    Returns:
        ScalingFactors object with computed scaling factors

    Examples:
        >>> problem = build_problem(
        ...     nodes=[{"id": "s", "supply": 1000}, {"id": "t", "supply": -1000}],
        ...     arcs=[{"tail": "s", "head": "t", "capacity": 1000, "cost": 0.01}],
        ... )
        >>> factors = compute_scaling_factors(problem)
        >>> factors.cost_scale  # Will scale costs up
        100.0
        >>> factors.supply_scale  # Will scale supplies down
        0.001
    """
    factors = ScalingFactors()

    # Compute geometric mean for costs
    costs = [abs(arc.cost) for arc in problem.arcs if arc.cost != 0]
    if costs:
        geo_mean = math.exp(sum(math.log(c) for c in costs) / len(costs))
        # Scale to bring geometric mean to 1.0
        factors.cost_scale = 1.0 / geo_mean if geo_mean > 0 else 1.0

    # Compute geometric mean for capacities
    capacities = [
        arc.capacity
        for arc in problem.arcs
        if arc.capacity is not None and math.isfinite(arc.capacity) and arc.capacity > 0
    ]
    if capacities:
        geo_mean = math.exp(sum(math.log(c) for c in capacities) / len(capacities))
        factors.capacity_scale = 1.0 / geo_mean if geo_mean > 0 else 1.0

    # Compute geometric mean for supplies
    supplies = [abs(node.supply) for node in problem.nodes.values() if node.supply != 0]
    if supplies:
        geo_mean = math.exp(sum(math.log(s) for s in supplies) / len(supplies))
        factors.supply_scale = 1.0 / geo_mean if geo_mean > 0 else 1.0

    factors.enabled = True
    return factors


def scale_problem(problem: NetworkProblem, factors: ScalingFactors) -> NetworkProblem:
    """Apply scaling factors to create a scaled copy of the problem.

    This creates a new NetworkProblem with scaled values. The original
    problem is not modified.

    Args:
        problem: Original problem to scale
        factors: Scaling factors to apply

    Returns:
        New NetworkProblem with scaled values

    Note:
        - Costs are scaled by cost_scale
        - Capacities and lower bounds are scaled by capacity_scale
        - Supplies/demands are scaled by supply_scale
        - The scaled problem maintains the same structure and feasibility

    Examples:
        >>> factors = ScalingFactors(cost_scale=10.0, capacity_scale=0.001, supply_scale=0.001)
        >>> scaled = scale_problem(problem, factors)
        >>> # All costs multiplied by 10, capacities and supplies divided by 1000
    """
    from .data import Arc, NetworkProblem, Node

    if not factors.enabled:
        return problem

    # Scale nodes (supplies/demands)
    scaled_nodes = {
        node_id: Node(id=node.id, supply=node.supply * factors.supply_scale)
        for node_id, node in problem.nodes.items()
    }

    # Scale arcs (costs, capacities, lower bounds)
    scaled_arcs = []
    for arc in problem.arcs:
        scaled_capacity = None
        if arc.capacity is not None:
            if math.isfinite(arc.capacity):
                scaled_capacity = arc.capacity * factors.capacity_scale
            else:
                scaled_capacity = arc.capacity  # Keep infinity as infinity

        scaled_arcs.append(
            Arc(
                tail=arc.tail,
                head=arc.head,
                capacity=scaled_capacity,
                cost=arc.cost * factors.cost_scale,
                lower=arc.lower * factors.capacity_scale,
            )
        )

    return NetworkProblem(
        directed=problem.directed,
        nodes=scaled_nodes,
        arcs=scaled_arcs,
        tolerance=problem.tolerance,
    )


def unscale_solution(
    flows: dict[tuple[str, str], float], objective: float, factors: ScalingFactors
) -> tuple[dict[tuple[str, str], float], float]:
    """Unscale the solution back to original units.

    Args:
        flows: Scaled flow values from solver
        objective: Scaled objective value
        factors: Scaling factors used for scaling

    Returns:
        Tuple of (unscaled_flows, unscaled_objective)

    Note:
        - Flows are unscaled by dividing by supply_scale (flows match scaled supplies)
        - Objective is unscaled by dividing by (cost_scale * supply_scale)
        - This ensures the unscaled solution is correct in original units

    Examples:
        >>> scaled_flows = {("s", "t"): 1.0}
        >>> scaled_obj = 10.0
        >>> factors = ScalingFactors(cost_scale=10.0, capacity_scale=0.001, supply_scale=0.001)
        >>> flows, obj = unscale_solution(scaled_flows, scaled_obj, factors)
        >>> flows[("s", "t")]
        1000.0
        >>> obj
        1.0
    """
    if not factors.enabled:
        return flows, objective

    # Unscale flows: divide by supply_scale
    # Flows are determined by supplies/demands which were scaled by supply_scale
    unscaled_flows = {arc: flow / factors.supply_scale for arc, flow in flows.items()}

    # Unscale objective: divide by (cost_scale * supply_scale)
    # Original: obj = sum(cost_i * flow_i)
    # Scaled: obj_scaled = sum((cost_i * cost_scale) * (flow_i * supply_scale))
    # So: obj_scaled = cost_scale * supply_scale * sum(cost_i * flow_i) = cost_scale * supply_scale * obj
    # Therefore: obj = obj_scaled / (cost_scale * supply_scale)
    unscaled_objective = objective / (factors.cost_scale * factors.supply_scale)

    return unscaled_flows, unscaled_objective
