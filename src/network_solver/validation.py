"""Numeric validation and preprocessing for network problems.

This module provides utilities to validate numeric properties of network problems
and apply preprocessing steps to improve numerical stability.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import NetworkProblem


@dataclass
class NumericWarning:
    """Warning about numeric issues in a network problem.

    Attributes:
        severity: Severity level ('low', 'medium', 'high')
        category: Category of warning (e.g., 'range', 'precision', 'conditioning')
        message: Human-readable warning message
        recommendation: Suggested action to resolve the issue
    """

    severity: str
    category: str
    message: str
    recommendation: str


@dataclass
class NumericAnalysis:
    """Results of numeric analysis on a network problem.

    Attributes:
        is_well_conditioned: Whether the problem appears numerically stable
        warnings: List of numeric warnings detected
        cost_range: Ratio of max to min absolute non-zero cost
        capacity_range: Ratio of max to min absolute non-zero capacity
        supply_range: Ratio of max to min absolute non-zero supply
        has_extreme_values: Whether any extreme values (>1e10 or <1e-10) detected
        recommended_tolerance: Suggested tolerance value based on numeric properties
    """

    is_well_conditioned: bool
    warnings: list[NumericWarning]
    cost_range: float
    capacity_range: float
    supply_range: float
    has_extreme_values: bool
    recommended_tolerance: float


def analyze_numeric_properties(problem: NetworkProblem) -> NumericAnalysis:
    """Analyze numeric properties of a network problem.

    Checks for potential numeric issues including:
    - Extreme value ranges (very large or very small numbers)
    - Wide coefficient ranges that may cause numerical issues
    - Precision loss in floating-point arithmetic
    """
    warnings_list: list[NumericWarning] = []

    # Collect all numeric values (non-zero)
    costs = [abs(arc.cost) for arc in problem.arcs if arc.cost != 0.0]
    capacities = [
        abs(arc.capacity) for arc in problem.arcs
        if arc.capacity is not None and not math.isinf(arc.capacity) and arc.capacity != 0.0
    ]
    supplies = [abs(node.supply) for node in problem.nodes.values() if node.supply != 0.0]

    # Check for empty lists
    if not costs:
        costs = [1.0]  # Default if all costs are zero
    if not capacities:
        capacities = [1.0]  # Default if no finite capacities
    if not supplies:
        supplies = [1.0]  # Default if all supplies are zero

    # Calculate ranges
    cost_range = max(costs) / min(costs) if costs else 1.0
    capacity_range = max(capacities) / min(capacities) if capacities else 1.0
    supply_range = max(supplies) / min(supplies) if supplies else 1.0

    # Check for extreme values
    has_extreme_values = False
    extreme_threshold_high = 1e10
    extreme_threshold_low = 1e-10

    # Check costs
    for arc in problem.arcs:
        if abs(arc.cost) > extreme_threshold_high:
            has_extreme_values = True
            warnings_list.append(NumericWarning(
                severity="medium",
                category="range",
                message=f"Arc {arc.tail}->{arc.head} has very large cost {arc.cost:.2e}",
                recommendation="Consider scaling costs to range [0.01, 1000] for better stability",
            ))
        elif 0 < abs(arc.cost) < extreme_threshold_low:
            has_extreme_values = True
            warnings_list.append(NumericWarning(
                severity="low",
                category="range",
                message=f"Arc {arc.tail}->{arc.head} has very small cost {arc.cost:.2e}",
                recommendation="Consider scaling costs to avoid precision loss",
            ))

    # Check capacities
    for arc in problem.arcs:
        if arc.capacity is not None and not math.isinf(arc.capacity):
            if arc.capacity > extreme_threshold_high:
                has_extreme_values = True
                warnings_list.append(NumericWarning(
                    severity="medium",
                    category="range",
                    message=f"Arc {arc.tail}->{arc.head} has very large capacity {arc.capacity:.2e}",
                    recommendation="Consider scaling capacities or using infinite capacity",
                ))
            elif 0 < arc.capacity < extreme_threshold_low:
                has_extreme_values = True
                warnings_list.append(NumericWarning(
                    severity="low",
                    category="range",
                    message=f"Arc {arc.tail}->{arc.head} has very small capacity {arc.capacity:.2e}",
                    recommendation="Consider scaling capacities or removing near-zero arcs",
                ))

    # Check supplies
    for node_id, node in problem.nodes.items():
        if abs(node.supply) > extreme_threshold_high:
            has_extreme_values = True
            warnings_list.append(NumericWarning(
                severity="medium",
                category="range",
                message=f"Node {node_id} has very large supply/demand {node.supply:.2e}",
                recommendation="Consider scaling supplies/demands for better numerical stability",
            ))
        elif 0 < abs(node.supply) < extreme_threshold_low:
            has_extreme_values = True
            warnings_list.append(NumericWarning(
                severity="low",
                category="range",
                message=f"Node {node_id} has very small supply/demand {node.supply:.2e}",
                recommendation="Consider scaling supplies/demands or removing near-zero values",
            ))

    # Check coefficient ranges
    if cost_range > 1e8:
        warnings_list.append(NumericWarning(
            severity="high",
            category="conditioning",
            message=f"Cost range is very wide: {cost_range:.2e} (max/min ratio)",
            recommendation="Wide cost ranges can cause numerical instability. Consider cost scaling.",
        ))
    elif cost_range > 1e6:
        warnings_list.append(NumericWarning(
            severity="medium",
            category="conditioning",
            message=f"Cost range is wide: {cost_range:.2e} (max/min ratio)",
            recommendation="Consider scaling costs to improve numerical stability",
        ))

    if capacity_range > 1e8:
        warnings_list.append(NumericWarning(
            severity="high",
            category="conditioning",
            message=f"Capacity range is very wide: {capacity_range:.2e} (max/min ratio)",
            recommendation="Wide capacity ranges can cause issues. Consider capacity scaling.",
        ))
    elif capacity_range > 1e6:
        warnings_list.append(NumericWarning(
            severity="medium",
            category="conditioning",
            message=f"Capacity range is wide: {capacity_range:.2e} (max/min ratio)",
            recommendation="Consider scaling capacities to improve stability",
        ))

    if supply_range > 1e8:
        warnings_list.append(NumericWarning(
            severity="high",
            category="conditioning",
            message=f"Supply/demand range is very wide: {supply_range:.2e} (max/min ratio)",
            recommendation="Wide supply ranges can cause issues. Consider supply scaling.",
        ))
    elif supply_range > 1e6:
        warnings_list.append(NumericWarning(
            severity="medium",
            category="conditioning",
            message=f"Supply/demand range is wide: {supply_range:.2e} (max/min ratio)",
            recommendation="Consider scaling supplies/demands",
        ))

    # Determine if well-conditioned
    # Consider medium and high severity as problematic
    high_severity_count = sum(1 for w in warnings_list if w.severity == "high")
    medium_severity_count = sum(1 for w in warnings_list if w.severity == "medium")
    is_well_conditioned = (
        high_severity_count == 0
        and medium_severity_count == 0
        and cost_range < 1e8
        and capacity_range < 1e8
        and supply_range < 1e8
    )

    # Recommend tolerance based on coefficient ranges
    max_range = max(cost_range, capacity_range, supply_range)
    if max_range > 1e6:
        recommended_tolerance = 1e-4
    elif max_range > 1e4:
        recommended_tolerance = 1e-5
    else:
        recommended_tolerance = 1e-6

    return NumericAnalysis(
        is_well_conditioned=is_well_conditioned,
        warnings=warnings_list,
        cost_range=cost_range,
        capacity_range=capacity_range,
        supply_range=supply_range,
        has_extreme_values=has_extreme_values,
        recommended_tolerance=recommended_tolerance,
    )


def validate_numeric_properties(
    problem: NetworkProblem,
    strict: bool = False,
    warn: bool = True,
) -> None:
    """Validate numeric properties and optionally warn about issues.

    Args:
        problem: The network problem to validate
        strict: If True, raise exception on high-severity warnings
        warn: If True, emit Python warnings for detected issues

    Raises:
        ValueError: If strict=True and high-severity issues are found
    """
    analysis = analyze_numeric_properties(problem)

    if warn and analysis.warnings:
        # Group warnings by severity
        high_warnings = [w for w in analysis.warnings if w.severity == "high"]
        medium_warnings = [w for w in analysis.warnings if w.severity == "medium"]
        low_warnings = [w for w in analysis.warnings if w.severity == "low"]

        # Emit warnings
        if high_warnings:
            msg = "High-severity numeric issues detected:\n"
            for w in high_warnings:
                msg += f"  - {w.message}\n    → {w.recommendation}\n"
            warnings.warn(msg, UserWarning, stacklevel=2)

        if medium_warnings:
            msg = "Medium-severity numeric issues detected:\n"
            for w in medium_warnings:
                msg += f"  - {w.message}\n    → {w.recommendation}\n"
            warnings.warn(msg, UserWarning, stacklevel=2)

        if low_warnings and len(low_warnings) > 0:
            # Only show first few low-severity warnings to avoid spam
            msg = f"Low-severity numeric issues detected ({len(low_warnings)} total):\n"
            for w in low_warnings[:3]:
                msg += f"  - {w.message}\n"
            if len(low_warnings) > 3:
                msg += f"  ... and {len(low_warnings) - 3} more\n"
            warnings.warn(msg, UserWarning, stacklevel=2)

    if strict:
        high_warnings = [w for w in analysis.warnings if w.severity == "high"]
        if high_warnings:
            error_msg = "Problem has high-severity numeric issues:\n"
            for w in high_warnings:
                error_msg += f"  - {w.message}\n    → {w.recommendation}\n"
            raise ValueError(error_msg)
