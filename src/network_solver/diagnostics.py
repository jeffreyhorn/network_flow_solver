"""Convergence diagnostics and stalling detection for network simplex.

This module provides utilities to monitor solver progress and detect
convergence issues such as stalling, cycling, and slow progress.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConvergenceMonitor:
    """Monitors convergence progress and detects stalling.

    Tracks objective value history and iteration patterns to identify:
    - Stalling: objective value not improving
    - Slow convergence: very small improvements per iteration
    - Cycling: same basis repeatedly visited
    - Degeneracy: high ratio of zero-pivot iterations

    Attributes:
        window_size: Number of recent iterations to track
        stall_threshold: Relative improvement threshold for stalling detection
        degeneracy_threshold: Ratio threshold for degeneracy warning

    Examples:
        >>> monitor = ConvergenceMonitor(window_size=50, stall_threshold=1e-8)
        >>> for iteration in range(num_iterations):
        ...     monitor.record_iteration(objective_value, is_degenerate_pivot)
        ...     if monitor.is_stalled():
        ...         print("Warning: solver may be stalled")
        ...     if monitor.is_highly_degenerate():
        ...         print("Warning: high degeneracy detected")
    """

    window_size: int = 50
    stall_threshold: float = 1e-8
    degeneracy_threshold: float = 0.5

    # History tracking
    objective_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    degenerate_pivots: int = 0
    total_pivots: int = 0

    # Stalling detection
    consecutive_no_improvement: int = 0
    last_significant_improvement_iter: int = 0

    def __post_init__(self) -> None:
        """Initialize history with correct maxlen."""
        self.objective_history = deque(maxlen=self.window_size)

    def record_iteration(
        self,
        objective: float,
        is_degenerate: bool = False,
        iteration: int = 0,
    ) -> None:
        """Record an iteration's progress.

        Args:
            objective: Current objective value
            is_degenerate: Whether this was a degenerate pivot
            iteration: Current iteration number
        """
        self.objective_history.append(objective)
        self.total_pivots += 1
        if is_degenerate:
            self.degenerate_pivots += 1

        # Check for improvement
        if len(self.objective_history) >= 2:
            prev_obj = self.objective_history[-2]
            current_obj = self.objective_history[-1]

            # Calculate relative improvement (handle zero objective)
            if abs(prev_obj) > 1e-12:
                rel_improvement = abs(current_obj - prev_obj) / abs(prev_obj)
            else:
                rel_improvement = abs(current_obj - prev_obj)

            if rel_improvement < self.stall_threshold:
                self.consecutive_no_improvement += 1
            else:
                self.consecutive_no_improvement = 0
                self.last_significant_improvement_iter = iteration

    def is_stalled(self, min_consecutive: int = 10) -> bool:
        """Check if solver appears to be stalled.

        Args:
            min_consecutive: Minimum consecutive iterations without improvement

        Returns:
            True if stalled for at least min_consecutive iterations
        """
        return self.consecutive_no_improvement >= min_consecutive

    def is_highly_degenerate(self) -> bool:
        """Check if degeneracy ratio is high.

        Returns:
            True if degenerate pivot ratio exceeds threshold
        """
        if self.total_pivots < 10:
            return False
        ratio = self.degenerate_pivots / self.total_pivots
        return ratio > self.degeneracy_threshold

    def get_degeneracy_ratio(self) -> float:
        """Get current degeneracy ratio.

        Returns:
            Ratio of degenerate pivots to total pivots
        """
        if self.total_pivots == 0:
            return 0.0
        return self.degenerate_pivots / self.total_pivots

    def get_recent_improvement(self) -> float | None:
        """Get improvement over the monitoring window.

        Returns:
            Relative improvement from oldest to newest in window, or None if insufficient data
        """
        if len(self.objective_history) < 2:
            return None

        oldest = self.objective_history[0]
        newest = self.objective_history[-1]

        if abs(oldest) > 1e-12:
            return abs(newest - oldest) / abs(oldest)
        else:
            return abs(newest - oldest)

    def get_average_improvement_rate(self) -> float | None:
        """Get average improvement rate per iteration in window.

        Returns:
            Average relative improvement per iteration, or None if insufficient data
        """
        improvement = self.get_recent_improvement()
        if improvement is None:
            return None

        # Average per iteration
        num_iters = len(self.objective_history) - 1
        if num_iters == 0:
            return None

        return improvement / num_iters

    def should_increase_tolerance(self) -> bool:
        """Check if tolerance should be increased due to stalling.

        Returns:
            True if solver is stalled and tolerance increase might help
        """
        # If stalled for long time with minimal improvement, increasing tolerance may help
        recent_improvement = self.get_recent_improvement()
        return (
            self.is_stalled(min_consecutive=20) and
            recent_improvement is not None and
            recent_improvement < 1e-10
        )

    def get_diagnostic_summary(self) -> dict[str, float | bool | int]:
        """Get summary of convergence diagnostics.

        Returns:
            Dictionary with diagnostic metrics
        """
        return {
            'total_pivots': self.total_pivots,
            'degenerate_pivots': self.degenerate_pivots,
            'degeneracy_ratio': self.get_degeneracy_ratio(),
            'is_stalled': self.is_stalled(),
            'is_highly_degenerate': self.is_highly_degenerate(),
            'consecutive_no_improvement': self.consecutive_no_improvement,
            'recent_improvement': self.get_recent_improvement() or 0.0,
            'avg_improvement_rate': self.get_average_improvement_rate() or 0.0,
        }


@dataclass
class BasisHistory:
    """Tracks basis state history to detect cycling.

    Maintains a hash-based history of recent basis states to detect
    if the same basis is visited repeatedly (cycling).

    Attributes:
        max_history: Maximum number of basis states to track

    Examples:
        >>> history = BasisHistory(max_history=100)
        >>> for iteration in range(num_iterations):
        ...     tree_arcs = get_current_tree_arcs()
        ...     if history.is_cycling(tree_arcs):
        ...         print("Warning: cycling detected")
        ...     history.record_basis(tree_arcs)
    """

    max_history: int = 100
    history: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    visit_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize history with correct maxlen."""
        self.history = deque(maxlen=self.max_history)

    def _hash_basis(self, tree_arcs: set[tuple[int, int]]) -> int:
        """Compute hash of basis state.

        Args:
            tree_arcs: Set of arc indices in current tree

        Returns:
            Hash value representing basis state
        """
        # Convert to sorted tuple for consistent hashing
        return hash(tuple(sorted(tree_arcs)))

    def record_basis(self, tree_arcs: set[tuple[int, int]]) -> None:
        """Record current basis state.

        Args:
            tree_arcs: Set of arc indices in current tree
        """
        basis_hash = self._hash_basis(tree_arcs)
        self.history.append(basis_hash)

        # Track visit count
        self.visit_counts[basis_hash] = self.visit_counts.get(basis_hash, 0) + 1

        # Clean up old counts to prevent unbounded growth
        if len(self.visit_counts) > self.max_history * 2:
            # Keep only hashes in current history
            current_hashes = set(self.history)
            old_keys = [k for k in self.visit_counts if k not in current_hashes]
            for k in old_keys:
                del self.visit_counts[k]

    def is_cycling(self, min_revisits: int = 3) -> bool:
        """Check if current basis has been visited too many times.

        Args:
            min_revisits: Minimum number of visits to consider cycling

        Returns:
            True if any recent basis has been visited >= min_revisits times
        """
        if not self.history:
            return False

        # Check if any basis in recent history has been revisited
        recent_hashes = list(self.history)[-20:]  # Check last 20
        for basis_hash in recent_hashes:
            if self.visit_counts.get(basis_hash, 0) >= min_revisits:
                return True

        return False

    def get_cycle_length(self) -> int | None:
        """Estimate cycle length if cycling is detected.

        Returns:
            Estimated cycle length, or None if no cycling detected
        """
        if len(self.history) < 4:
            return None

        # Look for repeated patterns in recent history
        recent = list(self.history)[-20:]

        # Try to find repeating pattern
        for pattern_len in range(2, len(recent) // 2):
            pattern = recent[-pattern_len:]
            prev_segment = recent[-2 * pattern_len:-pattern_len]

            if pattern == prev_segment:
                return pattern_len

        return None

    def get_most_frequent_basis_count(self) -> int:
        """Get visit count of most frequently visited basis.

        Returns:
            Maximum visit count among all bases in history
        """
        if not self.visit_counts:
            return 0
        return max(self.visit_counts.values())
