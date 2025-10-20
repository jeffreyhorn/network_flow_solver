"""Vectorized operations for network simplex hot paths.

This module provides NumPy-vectorized versions of performance-critical operations
identified through profiling. These functions operate on arrays instead of individual
arc objects to achieve significant speedups.
"""

import math

import numpy as np
from numpy.typing import NDArray


def compute_residuals_vectorized(
    flows: NDArray[np.float64],
    lowers: NDArray[np.float64],
    uppers: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute forward and backward residuals for all arcs at once.

    Args:
        flows: Array of current flows on each arc
        lowers: Array of lower bounds
        uppers: Array of upper bounds (may contain inf)

    Returns:
        Tuple of (forward_residuals, backward_residuals)

    Original hot path: forward_residual() called 188,583 times @ 0.215s
    Expected speedup: 10-50x via vectorization
    """
    # Forward residuals: upper - flow (handle inf uppers)
    forward_res = uppers - flows

    # Backward residuals: flow - lower
    backward_res = flows - lowers

    return forward_res, backward_res


def compute_reduced_costs_vectorized(
    arc_costs: NDArray[np.float64],
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    potentials: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute reduced costs for all arcs at once.

    Args:
        arc_costs: Cost per unit flow on each arc
        arc_tails: Tail node indices
        arc_heads: Head node indices
        potentials: Node potentials from basis

    Returns:
        Array of reduced costs: cost + potential[tail] - potential[head]

    This vectorizes the RC calculation done in every pricing iteration.
    """
    return arc_costs + potentials[arc_tails] - potentials[arc_heads]


def find_entering_arc_devex_vectorized(
    arc_costs: NDArray[np.float64],
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    arc_flows: NDArray[np.float64],
    arc_lowers: NDArray[np.float64],
    arc_uppers: NDArray[np.float64],
    arc_in_tree: NDArray[np.bool_],
    arc_artificial: NDArray[np.bool_],
    potentials: NDArray[np.float64],
    devex_weights: NDArray[np.float64],
    tolerance: float,
    block_start: int,
    block_end: int,
    allow_zero: bool = False,
) -> tuple[int, int, float] | None:
    """Vectorized Devex pricing for a block of arcs.

    Args:
        arc_costs: Cost coefficients
        arc_tails: Tail node indices
        arc_heads: Head node indices
        arc_flows: Current flows
        arc_lowers: Lower bounds
        arc_uppers: Upper bounds
        arc_in_tree: Boolean mask of arcs in basis tree
        arc_artificial: Boolean mask of artificial arcs
        potentials: Node potentials
        devex_weights: Devex weights for pricing
        tolerance: Numerical tolerance
        block_start: Start index of block
        block_end: End index of block
        allow_zero: Whether to consider zero-reduced-cost arcs

    Returns:
        Tuple of (arc_index, direction, merit) or None if no candidate found

    Original hot path: _find_entering_arc_devex @ 0.705s (254 calls)
    Expected speedup: 3-10x via vectorization
    """
    # Extract block
    n = block_end - block_start
    if n == 0:
        return None

    # Create masks for eligible arcs (not in tree, not artificial)
    eligible = ~arc_in_tree[block_start:block_end] & ~arc_artificial[block_start:block_end]

    # Convert potentials to array if needed (it might be a list from basis)
    if not isinstance(potentials, np.ndarray):
        potentials_arr = np.array(potentials, dtype=np.float64)
    else:
        potentials_arr = potentials

    # Compute reduced costs for block
    rc_block = (
        arc_costs[block_start:block_end]
        + potentials_arr[arc_tails[block_start:block_end]]
        - potentials_arr[arc_heads[block_start:block_end]]
    )

    # Compute residuals for block
    forward_res = arc_uppers[block_start:block_end] - arc_flows[block_start:block_end]
    backward_res = arc_flows[block_start:block_end] - arc_lowers[block_start:block_end]

    # Forward direction: rc < -tolerance and forward_res > tolerance
    forward_eligible = eligible & (rc_block < -tolerance) & (forward_res > tolerance)

    # Backward direction: rc > tolerance and backward_res > tolerance
    backward_eligible = eligible & (rc_block > tolerance) & (backward_res > tolerance)

    # Compute merits for candidates
    best_idx = -1
    best_direction = 0
    best_merit = -math.inf

    # Check forward candidates
    if np.any(forward_eligible):
        forward_indices = np.where(forward_eligible)[0]
        weights = devex_weights[block_start + forward_indices]
        weights = np.maximum(weights, 1e-12)  # DEVEX_WEIGHT_MIN
        rc_squared = rc_block[forward_indices] ** 2
        merits = rc_squared / weights

        max_idx = np.argmax(merits)
        if merits[max_idx] > best_merit:
            best_merit = merits[max_idx]
            best_idx = block_start + forward_indices[max_idx]
            best_direction = 1

    # Check backward candidates
    if np.any(backward_eligible):
        backward_indices = np.where(backward_eligible)[0]
        weights = devex_weights[block_start + backward_indices]
        weights = np.maximum(weights, 1e-12)  # DEVEX_WEIGHT_MIN
        rc_squared = rc_block[backward_indices] ** 2
        merits = rc_squared / weights

        max_idx = np.argmax(merits)
        if merits[max_idx] > best_merit:
            best_merit = merits[max_idx]
            best_idx = block_start + backward_indices[max_idx]
            best_direction = -1

    if best_idx >= 0:
        return (best_idx, best_direction, best_merit)

    # Handle zero-reduced-cost candidates if requested
    if allow_zero:
        zero_forward = eligible & (np.abs(rc_block) <= tolerance) & (forward_res > tolerance)
        zero_backward = eligible & (np.abs(rc_block) <= tolerance) & (backward_res > tolerance)

        if np.any(zero_forward):
            idx = block_start + np.where(zero_forward)[0][0]
            return (idx, 1, 0.0)
        if np.any(zero_backward):
            idx = block_start + np.where(zero_backward)[0][0]
            return (idx, -1, 0.0)

    return None


def update_flows_vectorized(
    arc_indices: NDArray[np.int32],
    arc_signs: NDArray[np.int8],
    theta: float,
    flows: NDArray[np.float64],
) -> None:
    """Update flows along a cycle in-place.

    Args:
        arc_indices: Indices of arcs in the cycle
        arc_signs: Direction of flow on each arc (+1 or -1)
        theta: Amount of flow to push
        flows: Array of arc flows (modified in-place)

    This vectorizes the flow update during pivot operations.
    """
    # Use NumPy advanced indexing for vectorized update
    flows[arc_indices] += arc_signs * theta


def check_eligibility_vectorized(
    arc_in_tree: NDArray[np.bool_],
    arc_artificial: NDArray[np.bool_],
    forward_res: NDArray[np.float64],
    backward_res: NDArray[np.float64],
    reduced_costs: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Check arc eligibility for entering basis (forward/backward).

    Args:
        arc_in_tree: Boolean mask of tree arcs
        arc_artificial: Boolean mask of artificial arcs
        forward_res: Forward residual capacities
        backward_res: Backward residual capacities
        reduced_costs: Reduced costs
        tolerance: Numerical tolerance

    Returns:
        Tuple of (forward_eligible, backward_eligible) boolean masks
    """
    # Base eligibility: not in tree, not artificial
    base_eligible = ~arc_in_tree & ~arc_artificial

    # Forward: negative reduced cost, positive forward residual
    forward_eligible = base_eligible & (reduced_costs < -tolerance) & (forward_res > tolerance)

    # Backward: positive reduced cost, positive backward residual
    backward_eligible = base_eligible & (reduced_costs > tolerance) & (backward_res > tolerance)

    return forward_eligible, backward_eligible
