"""Pricing strategies for network simplex arc selection.

This module contains implementations of different pricing strategies used to
select entering arcs during the simplex algorithm. Pricing strategies determine
which non-basic arc should enter the basis at each iteration.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .basis import TreeBasis
    from .simplex import ArcState

# Constants for Devex weight bounds
DEVEX_WEIGHT_MIN = 1e-12  # Prevent division by zero or runaway weights
DEVEX_WEIGHT_MAX = 1e12  # Cap the Devex weight to avoid catastrophic scaling


class PricingStrategy(ABC):
    """Abstract base class for pricing strategies.

    A pricing strategy is responsible for selecting the entering arc during
    each simplex pivot. Different strategies trade off between computational
    cost per iteration and total number of iterations.
    """

    @abstractmethod
    def select_entering_arc(
        self,
        arcs: list[ArcState],
        basis: TreeBasis,
        actual_arc_count: int,
        allow_zero: bool,
        tolerance: float,
    ) -> tuple[int, int] | None:
        """Select an entering arc for the next pivot.

        Args:
            arcs: List of all arcs in the network.
            basis: Current spanning tree basis with node potentials.
            actual_arc_count: Number of real (non-artificial) arcs.
            allow_zero: Whether to allow zero reduced cost arcs.
            tolerance: Numerical tolerance for comparisons.

        Returns:
            Tuple of (arc_index, direction) where direction is +1 for forward
            or -1 for backward, or None if no improving arc exists.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g., weights, pricing block position)."""
        pass


class DantzigPricing(PricingStrategy):
    """Dantzig pricing: select arc with most negative reduced cost.

    This is the simplest pricing strategy, examining all non-basic arcs and
    selecting the one with the most negative reduced cost. While simple, it
    can require many iterations for large problems.
    """

    def select_entering_arc(
        self,
        arcs: list[ArcState],
        basis: TreeBasis,
        actual_arc_count: int,
        allow_zero: bool,
        tolerance: float,
    ) -> tuple[int, int] | None:
        """Find arc with most negative reduced cost."""
        best: tuple[int, int] | None = None
        best_rc = 0.0

        for idx in range(actual_arc_count):
            arc = arcs[idx]
            if arc.in_tree or arc.artificial:
                continue

            rc = arc.cost + basis.potential[arc.tail] - basis.potential[arc.head]
            forward_res = arc.forward_residual()
            backward_res = arc.backward_residual()

            # Check forward direction (rc < 0 and forward residual > 0)
            if forward_res > tolerance and rc < -tolerance:
                if best is None or rc < best_rc:
                    best = (idx, 1)
                    best_rc = rc
            # Check backward direction (rc > 0 and backward residual > 0)
            elif backward_res > tolerance and rc > tolerance:
                if best is None or -rc < best_rc:
                    best = (idx, -1)
                    best_rc = -rc
            # Check zero reduced cost candidates if allowed
            elif allow_zero and forward_res > tolerance and abs(rc) <= tolerance and best is None:
                best = (idx, 1)
            elif allow_zero and backward_res > tolerance and abs(rc) <= tolerance and best is None:
                best = (idx, -1)

        return best

    def reset(self) -> None:
        """Dantzig pricing is stateless, so nothing to reset."""
        pass


class DevexPricing(PricingStrategy):
    """Devex pricing: steepest edge approximation with block search.

    Devex (Derivative of Extended pricing) uses approximate edge weights to
    normalize reduced costs, selecting the arc that provides the steepest
    descent direction. Uses block-based search to amortize computation.

    Attributes:
        weights: Approximate edge weights for each arc (used for normalization).
        block_size: Number of arcs to examine in each pricing round.
        pricing_block: Current block position (for round-robin search).
    """

    def __init__(
        self,
        arc_count: int,
        block_size: int,
        weights: list[float] | None = None,
    ):
        """Initialize Devex pricing with given parameters.

        Args:
            arc_count: Total number of arcs.
            block_size: Number of arcs per pricing block.
            weights: Optional pre-existing weights list to keep in sync.
                     If None, creates a new array initialized to 1.0.
        """
        self._weights_list: list[float] | None
        if weights is not None:
            # Keep reference to source list for sync and create numpy view
            self._weights_list = weights
            self.weights = np.array(weights, dtype=float)
        else:
            self._weights_list = None
            self.weights = np.ones(arc_count, dtype=float)
        self.block_size = block_size
        self.pricing_block = 0

    def select_entering_arc(
        self,
        arcs: list[ArcState],
        basis: TreeBasis,
        actual_arc_count: int,
        allow_zero: bool,
        tolerance: float,
    ) -> tuple[int, int] | None:
        """Find entering arc using Devex pricing with block search."""
        zero_candidates: list[tuple[int, int]] = []
        best: tuple[int, int] | None = None
        best_merit = -math.inf
        block_count = max(1, (actual_arc_count + self.block_size - 1) // self.block_size)

        for _ in range(block_count):
            start = self.pricing_block * self.block_size
            if start >= actual_arc_count:
                self.pricing_block = 0
                start = 0
            end = min(start + self.block_size, actual_arc_count)
            best_merit = -math.inf
            best = None
            zero_candidates = []

            for idx in range(start, end):
                arc = arcs[idx]
                if arc.in_tree or arc.artificial:
                    continue

                rc = arc.cost + basis.potential[arc.tail] - basis.potential[arc.head]
                forward_res = arc.forward_residual()
                backward_res = arc.backward_residual()

                # Check forward direction
                if forward_res > tolerance and rc < -tolerance:
                    weight = self._update_weight(idx, arc, basis)
                    merit = (rc * rc) / weight
                    if self._is_better_candidate(merit, idx, best_merit, best, tolerance):
                        best_merit = merit
                        best = (idx, 1)
                    continue

                # Check backward direction
                if backward_res > tolerance and rc > tolerance:
                    weight = self._update_weight(idx, arc, basis)
                    merit = (rc * rc) / weight
                    if self._is_better_candidate(merit, idx, best_merit, best, tolerance):
                        best_merit = merit
                        best = (idx, -1)
                    continue

                # Collect zero-reduced-cost candidates if allowed
                if allow_zero and forward_res > tolerance and abs(rc) <= tolerance:
                    zero_candidates.append((idx, 1))
                elif allow_zero and backward_res > tolerance and abs(rc) <= tolerance:
                    zero_candidates.append((idx, -1))

            if best is not None:
                return best
            if allow_zero and zero_candidates:
                self.pricing_block = (self.pricing_block + 1) % block_count
                return zero_candidates[0]

            self.pricing_block = (self.pricing_block + 1) % block_count

        return None

    def _update_weight(self, arc_idx: int, arc: ArcState, basis: TreeBasis) -> float:
        """Update and return the Devex weight for the given arc.

        The weight approximates the squared norm of the basis representation
        of the arc's column, providing a normalization factor for reduced costs.
        """
        weight: float = float(max(self.weights[arc_idx], DEVEX_WEIGHT_MIN))
        projection = basis.project_column(arc)

        if projection is not None:
            # Recompute Devex weight using the latest basis solve
            weight = float(np.dot(projection, projection))
            if not math.isfinite(weight) or weight <= DEVEX_WEIGHT_MIN:
                weight = DEVEX_WEIGHT_MIN
            elif weight > DEVEX_WEIGHT_MAX:
                weight = DEVEX_WEIGHT_MAX
            self.weights[arc_idx] = weight
            # Sync back to source list if provided
            if self._weights_list is not None:
                self._weights_list[arc_idx] = weight

        return weight

    def _is_better_candidate(
        self,
        merit: float,
        idx: int,
        best_merit: float,
        best: tuple[int, int] | None,
        tolerance: float,
    ) -> bool:
        """Check if current candidate is better than the best found so far.

        Compares merit values with tie-breaking by arc index for determinism.
        """
        better = merit > best_merit + tolerance
        tie = not better and abs(merit - best_merit) <= tolerance
        return better or (tie and (best is None or idx < best[0]))

    def reset(self) -> None:
        """Reset Devex weights to 1.0 and pricing block to 0."""
        self.weights.fill(1.0)
        self.pricing_block = 0
