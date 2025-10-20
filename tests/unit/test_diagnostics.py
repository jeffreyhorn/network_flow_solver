"""
Tests for convergence diagnostics.
"""

import pytest

from network_solver.diagnostics import BasisHistory, ConvergenceMonitor


class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ConvergenceMonitor(window_size=20, stall_threshold=1e-6)
        assert monitor.window_size == 20
        assert monitor.stall_threshold == 1e-6
        assert monitor.total_pivots == 0
        assert monitor.degenerate_pivots == 0

    def test_record_iteration_basic(self):
        """Test recording iterations."""
        monitor = ConvergenceMonitor()

        monitor.record_iteration(100.0, is_degenerate=False, iteration=1)
        assert monitor.total_pivots == 1
        assert monitor.degenerate_pivots == 0
        assert len(monitor.objective_history) == 1

        monitor.record_iteration(90.0, is_degenerate=True, iteration=2)
        assert monitor.total_pivots == 2
        assert monitor.degenerate_pivots == 1
        assert len(monitor.objective_history) == 2

    def test_degeneracy_detection(self):
        """Test degeneracy ratio calculation."""
        monitor = ConvergenceMonitor(degeneracy_threshold=0.3)

        # Record 10 iterations, 2 degenerate
        for i in range(10):
            is_degen = i < 2
            monitor.record_iteration(100.0 - i, is_degenerate=is_degen, iteration=i)

        assert monitor.get_degeneracy_ratio() == pytest.approx(0.2)
        assert not monitor.is_highly_degenerate()

        # Add more degenerate pivots
        for i in range(10, 20):
            monitor.record_iteration(90.0 - (i - 10), is_degenerate=True, iteration=i)

        # Now 12/20 = 0.6 degenerate
        assert monitor.get_degeneracy_ratio() == pytest.approx(0.6)
        assert monitor.is_highly_degenerate()

    def test_stalling_detection(self):
        """Test stalling detection."""
        monitor = ConvergenceMonitor(stall_threshold=1e-8)

        # Good progress initially
        for i in range(5):
            monitor.record_iteration(100.0 - i * 10, is_degenerate=False, iteration=i)

        assert not monitor.is_stalled()

        # Now stall with minimal improvement
        base_obj = 50.0
        for i in range(5, 20):
            # Very small improvements
            monitor.record_iteration(base_obj - (i - 5) * 1e-10, is_degenerate=False, iteration=i)

        assert monitor.is_stalled(min_consecutive=10)

    def test_improvement_tracking(self):
        """Test improvement rate calculation."""
        monitor = ConvergenceMonitor(window_size=10)

        # Linear improvement: 100 -> 90 over 10 iterations
        for i in range(10):
            monitor.record_iteration(100.0 - i, is_degenerate=False, iteration=i)

        improvement = monitor.get_recent_improvement()
        assert improvement is not None
        # Improvement is 91->100 = 9/100 = 0.09 (we measure oldest to newest)
        assert improvement == pytest.approx(0.09, abs=0.01)

        avg_rate = monitor.get_average_improvement_rate()
        assert avg_rate is not None
        assert avg_rate > 0

    def test_window_size_limit(self):
        """Test that history respects window size."""
        monitor = ConvergenceMonitor(window_size=5)

        # Record more than window size
        for i in range(10):
            monitor.record_iteration(100.0 - i, is_degenerate=False, iteration=i)

        # Should only keep last 5
        assert len(monitor.objective_history) == 5
        assert monitor.objective_history[0] == pytest.approx(95.0)
        assert monitor.objective_history[-1] == pytest.approx(91.0)

    def test_tolerance_increase_recommendation(self):
        """Test recommendation to increase tolerance."""
        monitor = ConvergenceMonitor(stall_threshold=1e-8, window_size=20)

        # Normal progress - should not recommend increase
        for i in range(10):
            monitor.record_iteration(100.0 - i * 5, is_degenerate=False, iteration=i)

        assert not monitor.should_increase_tolerance()

        # Now stall with minimal improvement (only recent 20 matter due to window)
        base_obj = 50.0
        for i in range(10, 50):
            monitor.record_iteration(base_obj - (i - 10) * 1e-12, is_degenerate=False, iteration=i)

        assert monitor.should_increase_tolerance()

    def test_diagnostic_summary(self):
        """Test diagnostic summary generation."""
        monitor = ConvergenceMonitor()

        for i in range(10):
            is_degen = i % 3 == 0
            monitor.record_iteration(100.0 - i * 2, is_degenerate=is_degen, iteration=i)

        summary = monitor.get_diagnostic_summary()

        assert "total_pivots" in summary
        assert "degenerate_pivots" in summary
        assert "degeneracy_ratio" in summary
        assert "is_stalled" in summary
        assert "recent_improvement" in summary

        assert summary["total_pivots"] == 10
        assert isinstance(summary["degeneracy_ratio"], float)
        assert isinstance(summary["is_stalled"], bool)

    def test_zero_objective_handling(self):
        """Test handling of zero objective values."""
        monitor = ConvergenceMonitor()

        # Test with zero objective
        monitor.record_iteration(0.0, is_degenerate=False, iteration=1)
        monitor.record_iteration(1e-10, is_degenerate=False, iteration=2)

        # Should not crash
        improvement = monitor.get_recent_improvement()
        assert improvement is not None


class TestBasisHistory:
    """Tests for BasisHistory class."""

    def test_initialization(self):
        """Test history initialization."""
        history = BasisHistory(max_history=50)
        assert history.max_history == 50
        assert len(history.history) == 0
        assert len(history.visit_counts) == 0

    def test_record_basis(self):
        """Test recording basis states."""
        history = BasisHistory()

        basis1 = {(0, 1), (1, 2), (2, 3)}
        basis2 = {(0, 2), (1, 3), (2, 4)}

        history.record_basis(basis1)
        assert len(history.history) == 1
        assert len(history.visit_counts) == 1

        history.record_basis(basis2)
        assert len(history.history) == 2
        assert len(history.visit_counts) == 2

    def test_cycling_detection(self):
        """Test cycling detection."""
        history = BasisHistory()

        basis1 = {(0, 1), (1, 2)}
        basis2 = {(0, 2), (1, 3)}

        # Record pattern: basis1, basis2, basis1, basis2, basis1
        for _ in range(3):
            history.record_basis(basis1)
            history.record_basis(basis2)

        # Should detect cycling (basis1 visited 3 times)
        assert history.is_cycling(min_revisits=3)

    def test_no_cycling_with_unique_bases(self):
        """Test that unique bases don't trigger cycling detection."""
        history = BasisHistory()

        # Record all unique bases
        for i in range(10):
            basis = {(i, i + 1), (i + 1, i + 2)}
            history.record_basis(basis)

        assert not history.is_cycling()

    def test_cycle_length_detection(self):
        """Test cycle length estimation."""
        history = BasisHistory()

        # Create repeating pattern of length 2
        basis1 = {(0, 1), (1, 2)}
        basis2 = {(0, 2), (1, 3)}

        # Repeat pattern 3 times to ensure detection
        for _ in range(3):
            history.record_basis(basis1)
            history.record_basis(basis2)

        cycle_length = history.get_cycle_length()
        # Should detect cycle of length 2
        assert cycle_length == 2

    def test_history_size_limit(self):
        """Test that history respects max size."""
        history = BasisHistory(max_history=5)

        # Record more than max
        for i in range(10):
            basis = {(i, i + 1)}
            history.record_basis(basis)

        # Should only keep last 5
        assert len(history.history) == 5

    def test_visit_count_tracking(self):
        """Test visit count tracking."""
        history = BasisHistory()

        basis1 = {(0, 1), (1, 2)}
        basis2 = {(0, 2), (1, 3)}

        # Visit basis1 three times, basis2 twice
        history.record_basis(basis1)
        history.record_basis(basis2)
        history.record_basis(basis1)
        history.record_basis(basis2)
        history.record_basis(basis1)

        assert history.get_most_frequent_basis_count() == 3

    def test_hash_consistency(self):
        """Test that equivalent bases produce same hash."""
        history = BasisHistory()

        # Same basis in different order
        basis1 = {(0, 1), (1, 2), (2, 3)}
        basis2 = {(2, 3), (0, 1), (1, 2)}

        hash1 = history._hash_basis(basis1)
        hash2 = history._hash_basis(basis2)

        assert hash1 == hash2

    def test_cleanup_old_counts(self):
        """Test cleanup of old visit counts."""
        history = BasisHistory(max_history=10)

        # Fill history beyond 2x max to trigger cleanup
        for i in range(25):
            basis = {(i, i + 1)}
            history.record_basis(basis)

        # Old counts should be cleaned up
        assert len(history.visit_counts) <= history.max_history * 2
