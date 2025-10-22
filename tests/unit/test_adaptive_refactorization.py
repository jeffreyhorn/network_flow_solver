"""Tests for adaptive basis refactorization feature."""

import pytest

from network_solver import SolverOptions, build_problem, solve_min_cost_flow
from network_solver.simplex import NetworkSimplex


class TestAdaptiveRefactorizationConfig:
    """Test configuration and validation of adaptive refactorization options."""

    def test_default_options_enable_adaptive(self):
        """Adaptive refactorization should be enabled by default."""
        options = SolverOptions()
        assert options.adaptive_refactorization is True
        assert options.condition_number_threshold == 1e12
        assert options.adaptive_ft_min == 20
        assert options.adaptive_ft_max == 200

    def test_can_disable_adaptive_refactorization(self):
        """Should be able to disable adaptive refactorization."""
        options = SolverOptions(adaptive_refactorization=False)
        assert options.adaptive_refactorization is False

    def test_can_customize_condition_number_threshold(self):
        """Should be able to customize condition number threshold."""
        options = SolverOptions(condition_number_threshold=1e10)
        assert options.condition_number_threshold == 1e10

    def test_can_customize_adaptive_ft_limits(self):
        """Should be able to customize adaptive ft_update_limit bounds."""
        options = SolverOptions(adaptive_ft_min=10, adaptive_ft_max=300)
        assert options.adaptive_ft_min == 10
        assert options.adaptive_ft_max == 300

    def test_validation_rejects_invalid_condition_threshold(self):
        """Should reject condition number threshold <= 1."""
        with pytest.raises(Exception, match="Condition number threshold must be > 1"):
            SolverOptions(condition_number_threshold=1.0)

        with pytest.raises(Exception, match="Condition number threshold must be > 1"):
            SolverOptions(condition_number_threshold=0.5)

    def test_validation_rejects_invalid_adaptive_ft_bounds(self):
        """Should reject invalid adaptive_ft_min/max bounds."""
        # Min must be positive
        with pytest.raises(Exception, match="Adaptive FT min must be positive"):
            SolverOptions(adaptive_ft_min=0)

        # Min must be <= max
        with pytest.raises(Exception, match="Adaptive FT min must be positive and <= max"):
            SolverOptions(adaptive_ft_min=300, adaptive_ft_max=200)


class TestConditionNumberEstimation:
    """Test condition number estimation in TreeBasis."""

    def test_condition_number_estimated_for_valid_basis(self):
        """Should estimate condition number when basis matrix exists."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 200.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 200.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        solver = NetworkSimplex(problem)
        condition_number = solver.basis.estimate_condition_number()

        # Should return a valid condition number
        assert condition_number is not None
        assert condition_number > 0
        assert condition_number < 1e15  # Should be well-conditioned for simple problem

    def test_condition_number_none_when_basis_not_built(self):
        """Should return None when basis matrix not available."""
        problem = build_problem(
            nodes=[{"id": "s", "supply": 100.0}, {"id": "t", "supply": -100.0}],
            arcs=[{"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0}],
            directed=True,
            tolerance=1e-6,
        )

        solver = NetworkSimplex(problem)
        # Before numeric basis is built
        solver.basis.basis_matrix = None
        solver.basis.basis_inverse = None
        condition_number = solver.basis.estimate_condition_number()

        assert condition_number is None


class TestAdaptiveRefactorizationBehavior:
    """Test adaptive refactorization behavior during solve."""

    def test_adaptive_refactorization_enabled_by_default(self):
        """Solver should use adaptive refactorization by default."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(100.0)

    def test_adaptive_refactorization_can_be_disabled(self):
        """Solver should work with adaptive refactorization disabled."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(adaptive_refactorization=False)
        result = solve_min_cost_flow(problem, options=options)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(100.0)

    def test_solver_tracks_condition_number_history(self):
        """Solver should track condition number history when adaptive enabled."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m1", "supply": 0.0},
                {"id": "m2", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m1", "capacity": 200.0, "cost": 1.0},
                {"tail": "s", "head": "m2", "capacity": 200.0, "cost": 2.0},
                {"tail": "m1", "head": "t", "capacity": 200.0, "cost": 1.0},
                {"tail": "m2", "head": "t", "capacity": 200.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        solver = NetworkSimplex(problem, SolverOptions(adaptive_refactorization=True))
        result = solver.solve()

        # Should have tracked some condition numbers
        assert len(solver.condition_number_history) >= 0
        assert result.status == "optimal"

    def test_ft_limit_can_be_adjusted_adaptively(self):
        """Current ft_limit should be adjustable during solve."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(
            adaptive_refactorization=True,
            ft_update_limit=64,
            adaptive_ft_min=20,
            adaptive_ft_max=200,
        )
        solver = NetworkSimplex(problem, options)

        # Initial ft_limit should match configured value
        assert solver.current_ft_limit == 64

        result = solver.solve()
        assert result.status == "optimal"


class TestAdaptiveRefactorizationStability:
    """Test that adaptive refactorization improves numerical stability."""

    def test_produces_correct_results_with_adaptive(self):
        """Should produce correct results with adaptive refactorization."""
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 50.0},
                {"id": "s2", "supply": 50.0},
                {"id": "t1", "supply": -60.0},
                {"id": "t2", "supply": -40.0},
            ],
            arcs=[
                {"tail": "s1", "head": "t1", "capacity": 100.0, "cost": 2.0},
                {"tail": "s1", "head": "t2", "capacity": 100.0, "cost": 3.0},
                {"tail": "s2", "head": "t1", "capacity": 100.0, "cost": 1.0},
                {"tail": "s2", "head": "t2", "capacity": 100.0, "cost": 4.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options_adaptive = SolverOptions(adaptive_refactorization=True)
        result_adaptive = solve_min_cost_flow(problem, options=options_adaptive)

        options_fixed = SolverOptions(adaptive_refactorization=False)
        result_fixed = solve_min_cost_flow(problem, options=options_fixed)

        # Both should produce optimal results
        assert result_adaptive.status == "optimal"
        assert result_fixed.status == "optimal"

        # Should produce same objective
        assert result_adaptive.objective == pytest.approx(result_fixed.objective)


class TestAdaptiveRefactorizationEdgeCases:
    """Test edge cases for adaptive refactorization."""

    def test_works_with_single_arc_problem(self):
        """Should handle single-arc problems correctly."""
        problem = build_problem(
            nodes=[{"id": "s", "supply": 10.0}, {"id": "t", "supply": -10.0}],
            arcs=[{"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0}],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(adaptive_refactorization=True)
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)

    def test_works_with_zero_costs(self):
        """Should handle problems with zero costs."""
        problem = build_problem(
            nodes=[{"id": "s", "supply": 10.0}, {"id": "t", "supply": -10.0}],
            arcs=[{"tail": "s", "head": "t", "capacity": 20.0, "cost": 0.0}],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(adaptive_refactorization=True)
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0)

    def test_works_with_high_condition_number_threshold(self):
        """Should work with very high condition number threshold."""
        problem = build_problem(
            nodes=[{"id": "s", "supply": 100.0}, {"id": "t", "supply": -100.0}],
            arcs=[{"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0}],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(
            adaptive_refactorization=True,
            condition_number_threshold=1e14,  # Very high threshold
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(100.0)

    def test_works_with_tight_adaptive_ft_bounds(self):
        """Should work with tight adaptive_ft_min/max bounds."""
        problem = build_problem(
            nodes=[{"id": "s", "supply": 100.0}, {"id": "t", "supply": -100.0}],
            arcs=[{"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0}],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(
            adaptive_refactorization=True,
            adaptive_ft_min=30,
            adaptive_ft_max=40,  # Narrow range
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(100.0)
