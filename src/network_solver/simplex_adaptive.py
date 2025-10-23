"""Adaptive parameter tuning for network simplex solver.

This module provides runtime adaptation of solver parameters based on
observed performance metrics. Adaptive tuning helps the solver automatically
adjust to problem characteristics without manual parameter configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import SolverOptions


class AdaptiveTuner:
    """Manages adaptive tuning of solver parameters during execution.

    This class encapsulates the logic for dynamically adjusting solver
    parameters based on runtime performance metrics:
    - Block size for pricing (based on degeneracy ratio)
    - Forrest-Tomlin update limit (based on numerical conditioning)

    Attributes:
        logger: Logger for recording adaptation decisions.
        actual_arc_count: Number of real (non-artificial) arcs.
        auto_tune_block_size: Whether block size auto-tuning is enabled.
        block_size: Current block size for pricing.
        adaptation_interval: Number of iterations between adaptations.
        last_adaptation_iteration: Last iteration when adaptation occurred.
        degenerate_pivot_count: Count of degenerate pivots.
        total_pivot_count: Total pivot count.
        current_ft_limit: Current Forrest-Tomlin update limit.
        options: Solver configuration options.
    """

    def __init__(
        self,
        actual_arc_count: int,
        initial_block_size: int,
        auto_tune: bool,
        options: SolverOptions,
        logger: logging.Logger,
    ):
        """Initialize adaptive tuner.

        Args:
            actual_arc_count: Number of real (non-artificial) arcs in problem.
            initial_block_size: Starting block size for pricing.
            auto_tune: Whether to enable automatic block size tuning.
            options: Solver configuration options.
            logger: Logger for recording adaptation decisions.
        """
        self.logger = logger
        self.actual_arc_count = actual_arc_count
        self.auto_tune_block_size = auto_tune
        self.block_size = initial_block_size

        # Adaptation state
        self.adaptation_interval = 50  # Adapt every 50 iterations
        self.last_adaptation_iteration = 0
        self.degenerate_pivot_count = 0
        self.total_pivot_count = 0

        # Forrest-Tomlin adaptation state
        self.current_ft_limit = options.ft_update_limit
        self.options = options

    @staticmethod
    def compute_initial_block_size(arc_count: int) -> int:
        """Compute initial block size based on problem size (static heuristic).

        Strategy:
        - Very small (<100 arcs): arc_count // 4 (but at least 1)
        - Small (100-1000): arc_count // 4
        - Medium (1000-10000): arc_count // 8
        - Large (>10000): arc_count // 16

        Note: We avoid full scans (block_size == arc_count) due to potential
        issues with pricing logic edge cases.

        Args:
            arc_count: Number of arcs in the problem.

        Returns:
            Initial block size (at least 1).
        """
        if arc_count < 100:
            return max(1, arc_count // 4)  # Very small
        elif arc_count < 1000:
            return max(1, arc_count // 4)  # Small
        elif arc_count < 10000:
            return max(1, arc_count // 8)  # Medium
        else:
            return max(1, arc_count // 16)  # Large

    def adapt_block_size(self, iteration: int) -> None:
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

        # Need sufficient samples to make a decision (and avoid division by zero)
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
            self.last_adaptation_iteration = iteration

    def adjust_ft_limit(self, condition_number: float | None) -> None:
        """Adaptively adjust ft_update_limit based on numerical behavior.

        Strategy:
        - If condition number triggered rebuild early, decrease limit (be more conservative)
        - Use problem characteristics to determine appropriate limit
        - Keep within configured min/max bounds

        Args:
            condition_number: Condition number from basis factorization, or None if not available.
        """
        if condition_number is None:
            return

        # If condition number is very high, be more aggressive with rebuilds
        if condition_number > self.options.condition_number_threshold * 10:
            # Very ill-conditioned: reduce limit significantly
            new_limit = max(self.options.adaptive_ft_min, int(self.current_ft_limit * 0.5))
        elif condition_number > self.options.condition_number_threshold:
            # Moderately ill-conditioned: reduce limit gradually
            new_limit = max(self.options.adaptive_ft_min, int(self.current_ft_limit * 0.8))
        else:
            # Condition number is good, can potentially increase limit
            new_limit = min(self.options.adaptive_ft_max, int(self.current_ft_limit * 1.1))

        if new_limit != self.current_ft_limit:
            self.logger.debug(
                f"Adjusted ft_update_limit: {self.current_ft_limit} → {new_limit}",
                extra={
                    "old_limit": self.current_ft_limit,
                    "new_limit": new_limit,
                    "condition_number": f"{condition_number:.2e}",
                },
            )
            self.current_ft_limit = new_limit

    def record_pivot(self, is_degenerate: bool) -> None:
        """Record pivot statistics for adaptation.

        Args:
            is_degenerate: Whether the pivot was degenerate (theta = 0).
        """
        self.total_pivot_count += 1
        if is_degenerate:
            self.degenerate_pivot_count += 1
