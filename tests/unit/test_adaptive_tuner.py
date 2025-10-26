"""Direct unit tests for AdaptiveTuner class."""

import logging

import pytest

from network_solver.data import SolverOptions
from network_solver.simplex_adaptive import AdaptiveTuner


class TestComputeInitialBlockSize:
    """Test static method for computing initial block size."""

    def test_very_small_problem_less_than_100_arcs(self):
        """Very small problems (<100 arcs) should use arc_count // 4."""
        assert AdaptiveTuner.compute_initial_block_size(40) == 10
        assert AdaptiveTuner.compute_initial_block_size(80) == 20
        assert AdaptiveTuner.compute_initial_block_size(4) == 1  # Minimum of 1

    def test_small_problem_100_to_1000_arcs(self):
        """Small problems (100-1000 arcs) should use arc_count // 4."""
        assert AdaptiveTuner.compute_initial_block_size(100) == 25
        assert AdaptiveTuner.compute_initial_block_size(400) == 100
        assert AdaptiveTuner.compute_initial_block_size(999) == 249

    def test_medium_problem_1000_to_10000_arcs(self):
        """Medium problems (1000-10000 arcs) should use arc_count // 8."""
        assert AdaptiveTuner.compute_initial_block_size(1000) == 125
        assert AdaptiveTuner.compute_initial_block_size(4000) == 500
        assert AdaptiveTuner.compute_initial_block_size(9999) == 1249

    def test_large_problem_more_than_10000_arcs(self):
        """Large problems (>10000 arcs) should use arc_count // 16."""
        assert AdaptiveTuner.compute_initial_block_size(10000) == 625
        assert AdaptiveTuner.compute_initial_block_size(16000) == 1000
        assert AdaptiveTuner.compute_initial_block_size(100000) == 6250

    def test_edge_case_one_arc(self):
        """Should handle single arc problem."""
        assert AdaptiveTuner.compute_initial_block_size(1) == 1

    def test_edge_case_zero_arcs(self):
        """Should handle zero arcs (returns 1 as minimum)."""
        # max(1, 0 // 4) = max(1, 0) = 1
        assert AdaptiveTuner.compute_initial_block_size(0) == 1


class TestAdaptiveTunerInitialization:
    """Test AdaptiveTuner initialization."""

    def test_basic_initialization(self):
        """Test basic AdaptiveTuner initialization."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        assert tuner.actual_arc_count == 1000
        assert tuner.block_size == 125
        assert tuner.auto_tune_block_size is True
        assert tuner.adaptation_interval == 50
        assert tuner.last_adaptation_iteration == 0
        assert tuner.degenerate_pivot_count == 0
        assert tuner.total_pivot_count == 0
        assert tuner.current_ft_limit == options.ft_update_limit

    def test_initialization_with_auto_tune_disabled(self):
        """Test initialization with auto-tuning disabled."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=False,
            options=options,
            logger=logger,
        )

        assert tuner.auto_tune_block_size is False


class TestRecordPivot:
    """Test pivot recording for adaptation."""

    def test_record_normal_pivot(self):
        """Test recording a normal (non-degenerate) pivot."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        tuner.record_pivot(is_degenerate=False)

        assert tuner.total_pivot_count == 1
        assert tuner.degenerate_pivot_count == 0

    def test_record_degenerate_pivot(self):
        """Test recording a degenerate pivot."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        tuner.record_pivot(is_degenerate=True)

        assert tuner.total_pivot_count == 1
        assert tuner.degenerate_pivot_count == 1

    def test_record_multiple_pivots(self):
        """Test recording multiple pivots."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        tuner.record_pivot(is_degenerate=False)
        tuner.record_pivot(is_degenerate=True)
        tuner.record_pivot(is_degenerate=True)
        tuner.record_pivot(is_degenerate=False)

        assert tuner.total_pivot_count == 4
        assert tuner.degenerate_pivot_count == 2


class TestAdaptBlockSize:
    """Test block size adaptation logic."""

    def test_no_adaptation_when_auto_tune_disabled(self):
        """Should not adapt when auto_tune_block_size is False."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=False,  # Disabled
            options=options,
            logger=logger,
        )

        # Record some pivots
        for _ in range(20):
            tuner.record_pivot(is_degenerate=True)

        # Try to adapt
        changed = tuner.adapt_block_size(iteration=100)

        assert changed is False  # Should not change
        assert tuner.block_size == 125  # Should remain initial value

    def test_no_adaptation_before_interval(self):
        """Should not adapt if not enough iterations have passed."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record some pivots
        for _ in range(20):
            tuner.record_pivot(is_degenerate=True)

        # Try to adapt too early (interval is 50)
        changed = tuner.adapt_block_size(iteration=20)

        assert changed is False

    def test_no_adaptation_with_insufficient_samples(self):
        """Should not adapt if total_pivot_count < 10."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record only a few pivots
        for _ in range(5):
            tuner.record_pivot(is_degenerate=True)

        # Try to adapt after interval
        changed = tuner.adapt_block_size(iteration=60)

        assert changed is False

    def test_increase_block_size_on_high_degeneracy(self):
        """Should increase block size when degeneracy > 30%."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=100,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record high degeneracy (8 out of 10 = 80%)
        for _ in range(8):
            tuner.record_pivot(is_degenerate=True)
        for _ in range(2):
            tuner.record_pivot(is_degenerate=False)

        original_size = tuner.block_size
        changed = tuner.adapt_block_size(iteration=60)

        assert changed is True
        assert tuner.block_size == int(original_size * 1.5)  # Should increase by 50%
        assert tuner.block_size <= tuner.actual_arc_count  # Capped at arc count

    def test_decrease_block_size_on_low_degeneracy(self):
        """Should decrease block size when degeneracy < 10%."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=100,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record low degeneracy (0 out of 20 = 0%)
        for _ in range(20):
            tuner.record_pivot(is_degenerate=False)

        original_size = tuner.block_size
        changed = tuner.adapt_block_size(iteration=60)

        assert changed is True
        assert tuner.block_size == int(original_size * 0.75)  # Should decrease by 25%
        assert tuner.block_size >= 10  # Min of 10

    def test_no_change_on_moderate_degeneracy(self):
        """Should not change block size for moderate degeneracy (10-30%)."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=100,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record moderate degeneracy (3 out of 15 = 20%)
        for _ in range(3):
            tuner.record_pivot(is_degenerate=True)
        for _ in range(12):
            tuner.record_pivot(is_degenerate=False)

        original_size = tuner.block_size
        changed = tuner.adapt_block_size(iteration=60)

        assert changed is False  # No change in moderate range
        assert tuner.block_size == original_size

    def test_block_size_capped_at_arc_count(self):
        """Block size should not exceed actual arc count."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=100,  # Small arc count
            initial_block_size=80,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record high degeneracy to trigger increase
        for _ in range(20):
            tuner.record_pivot(is_degenerate=True)

        tuner.adapt_block_size(iteration=60)

        # Should be capped at arc count
        assert tuner.block_size <= 100

    def test_block_size_has_minimum_of_10(self):
        """Block size should have minimum of 10."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=15,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record low degeneracy to trigger decrease
        for _ in range(20):
            tuner.record_pivot(is_degenerate=False)

        tuner.adapt_block_size(iteration=60)

        # Should be at least 10
        assert tuner.block_size >= 10

    def test_counters_reset_after_adaptation(self):
        """Pivot counters should reset after adaptation."""
        options = SolverOptions()
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=100,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Record some pivots
        for _ in range(20):
            tuner.record_pivot(is_degenerate=True)

        tuner.adapt_block_size(iteration=60)

        # Counters should be reset
        assert tuner.degenerate_pivot_count == 0
        assert tuner.total_pivot_count == 0
        assert tuner.last_adaptation_iteration == 60


class TestAdjustFTLimit:
    """Test Forrest-Tomlin update limit adjustment."""

    def test_no_adjustment_when_condition_number_none(self):
        """Should not adjust if condition number is None."""
        options = SolverOptions(ft_update_limit=64)
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        original_limit = tuner.current_ft_limit
        tuner.adjust_ft_limit(condition_number=None)

        assert tuner.current_ft_limit == original_limit

    def test_reduce_limit_on_very_high_condition_number(self):
        """Should reduce limit significantly for very high condition number."""
        options = SolverOptions(
            ft_update_limit=100,
            condition_number_threshold=1e12,
            adaptive_ft_min=20,
            adaptive_ft_max=200,
        )
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Very high condition number (> threshold * 10)
        tuner.adjust_ft_limit(condition_number=1e14)

        # Should reduce by 50%
        assert tuner.current_ft_limit == max(20, int(100 * 0.5))

    def test_reduce_limit_on_high_condition_number(self):
        """Should reduce limit gradually for moderately high condition number."""
        options = SolverOptions(
            ft_update_limit=100,
            condition_number_threshold=1e12,
            adaptive_ft_min=20,
            adaptive_ft_max=200,
        )
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # High but not extreme condition number
        tuner.adjust_ft_limit(condition_number=5e12)

        # Should reduce by 20% (80% of original)
        assert tuner.current_ft_limit == max(20, int(100 * 0.8))

    def test_increase_limit_on_good_condition_number(self):
        """Should increase limit when condition number is good."""
        options = SolverOptions(
            ft_update_limit=100,
            condition_number_threshold=1e12,
            adaptive_ft_min=20,
            adaptive_ft_max=200,
        )
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Good condition number (well below threshold)
        tuner.adjust_ft_limit(condition_number=1e8)

        # Should increase by 10%
        assert tuner.current_ft_limit == min(200, int(100 * 1.1))

    def test_limit_respects_adaptive_ft_min(self):
        """FT limit should not go below adaptive_ft_min."""
        options = SolverOptions(
            ft_update_limit=30,
            condition_number_threshold=1e12,
            adaptive_ft_min=25,
            adaptive_ft_max=200,
        )
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Try to reduce below minimum
        tuner.adjust_ft_limit(condition_number=1e14)

        # Should be clamped at minimum
        assert tuner.current_ft_limit == 25

    def test_limit_respects_adaptive_ft_max(self):
        """FT limit should not go above adaptive_ft_max."""
        options = SolverOptions(
            ft_update_limit=190,
            condition_number_threshold=1e12,
            adaptive_ft_min=20,
            adaptive_ft_max=200,
        )
        logger = logging.getLogger("test")
        tuner = AdaptiveTuner(
            actual_arc_count=1000,
            initial_block_size=125,
            auto_tune=True,
            options=options,
            logger=logger,
        )

        # Try to increase above maximum
        tuner.adjust_ft_limit(condition_number=1e8)

        # Should be clamped at maximum
        assert tuner.current_ft_limit == 200
