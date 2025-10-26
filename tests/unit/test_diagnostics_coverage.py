"""Additional tests to improve diagnostics.py coverage to >90%."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.diagnostics import BasisHistory, ConvergenceMonitor


class TestConvergenceMonitorEdgeCases:
    """Test ConvergenceMonitor edge cases for better coverage."""

    def test_is_highly_degenerate_with_few_pivots(self):
        """Test is_highly_degenerate returns False when total_pivots < 10 (line 99)."""
        monitor = ConvergenceMonitor(window_size=10)

        # Record only 5 pivots (< 10)
        for i in range(5):
            monitor.record_iteration(objective=100.0 - i, is_degenerate=True, iteration=i)

        # Should return False even though all pivots are degenerate
        assert monitor.is_highly_degenerate() is False
        assert monitor.total_pivots == 5

    def test_get_degeneracy_ratio_with_zero_pivots(self):
        """Test get_degeneracy_ratio returns 0.0 when no pivots recorded (line 110)."""
        monitor = ConvergenceMonitor(window_size=10)

        # No pivots recorded yet
        assert monitor.total_pivots == 0
        assert monitor.get_degeneracy_ratio() == 0.0

    def test_get_recent_improvement_with_insufficient_data(self):
        """Test get_recent_improvement returns None with < 2 data points (line 120)."""
        monitor = ConvergenceMonitor(window_size=10)

        # No data
        assert monitor.get_recent_improvement() is None

        # Only 1 data point
        monitor.record_iteration(objective=100.0, is_degenerate=False, iteration=0)
        assert len(monitor.objective_history) == 1
        assert monitor.get_recent_improvement() is None

    def test_get_average_improvement_rate_with_no_improvement_data(self):
        """Test get_average_improvement_rate returns None when improvement is None (line 138)."""
        monitor = ConvergenceMonitor(window_size=10)

        # No data points - get_recent_improvement will return None
        assert monitor.get_average_improvement_rate() is None


class TestBasisHistoryEdgeCases:
    """Test BasisHistory edge cases for better coverage."""

    def test_is_cycling_with_empty_history(self):
        """Test is_cycling returns False with empty history (line 240)."""
        history = BasisHistory()

        # Empty history
        assert len(history.history) == 0
        assert history.is_cycling(min_revisits=2) is False

    def test_get_cycle_length_with_insufficient_history(self):
        """Test get_cycle_length returns None when history < 4 (line 257)."""
        history = BasisHistory()

        # No history
        assert history.get_cycle_length() is None

        # 1 basis
        history.record_basis({(1, 2), (2, 3)})
        assert len(history.history) == 1
        assert history.get_cycle_length() is None

        # 2 bases
        history.record_basis({(4, 5), (5, 6)})
        assert len(history.history) == 2
        assert history.get_cycle_length() is None

        # 3 bases
        history.record_basis({(7, 8), (8, 9)})
        assert len(history.history) == 3
        assert history.get_cycle_length() is None

    def test_get_cycle_length_with_no_pattern(self):
        """Test get_cycle_length returns None when no repeating pattern found (line 270)."""
        history = BasisHistory()

        # Add 10 unique bases with no pattern
        for i in range(10):
            history.record_basis({(i, i + 1), (i + 1, i + 2)})

        # Should find no cycle
        assert history.get_cycle_length() is None

    def test_get_cycle_length_with_repeating_pattern(self):
        """Test get_cycle_length detects repeating pattern (line 270 branch)."""
        history = BasisHistory()

        # Create a repeating pattern: A, B, A, B, A, B
        basis_a = {(1, 2), (2, 3)}
        basis_b = {(4, 5), (5, 6)}

        # Add pattern 3 times to ensure detection
        for _ in range(3):
            history.record_basis(basis_a)
            history.record_basis(basis_b)

        # Should detect cycle of length 2
        cycle_len = history.get_cycle_length()
        assert cycle_len == 2

    def test_get_most_frequent_basis_count_with_empty_counts(self):
        """Test get_most_frequent_basis_count returns 0 when empty (line 279)."""
        history = BasisHistory()

        # No bases recorded
        assert len(history.visit_counts) == 0
        assert history.get_most_frequent_basis_count() == 0
