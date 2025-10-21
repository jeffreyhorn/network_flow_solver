"""Unit tests for automatic problem scaling.

Tests for scaling detection, factor computation, scaling/unscaling,
and integration with the solver.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import SolverOptions, build_problem  # noqa: E402
from network_solver.scaling import (  # noqa: E402
    ScalingFactors,
    compute_scaling_factors,
    scale_problem,
    should_scale_problem,
    unscale_solution,
)
from network_solver.solver import solve_min_cost_flow  # noqa: E402


class TestScalingDetection:
    """Test detection of problems that need scaling."""

    def test_should_not_scale_balanced_problem(self):
        """Problems with values in similar ranges don't need scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 100.0, "cost": 10.0},
            ],
        )

        assert not should_scale_problem(problem)

    def test_should_scale_wide_cost_range(self):
        """Problems with costs varying by >6 orders of magnitude need scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 100.0, "cost": 1e-6},
                {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1e3},
            ],
        )

        # Cost range: 1e-6 to 1e3 = 9 orders of magnitude
        assert should_scale_problem(problem)

    def test_should_scale_wide_capacity_range(self):
        """Problems with capacities varying widely need scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 1e-3, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 1e8, "cost": 1.0},
            ],
        )

        # Capacity range: 1e-3 to 1e8 = 11 orders of magnitude
        assert should_scale_problem(problem)

    def test_should_scale_wide_supply_range(self):
        """Problems with supplies varying widely need scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s1", "supply": 1e-3},
                {"id": "s2", "supply": 1e8},
                {"id": "t", "supply": -(1e-3 + 1e8)},
            ],
            arcs=[
                {"tail": "s1", "head": "t", "capacity": 1e9, "cost": 1.0},
                {"tail": "s2", "head": "t", "capacity": 1e9, "cost": 1.0},
            ],
        )

        # Supply range: 1e-3 to 1e8 = 11 orders of magnitude
        assert should_scale_problem(problem)

    def test_should_scale_cross_category_range(self):
        """Wide range across different categories triggers scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1e8},
                {"id": "t", "supply": -1e8},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 1e8, "cost": 1e-6},
            ],
        )

        # Cost (1e-6) vs supply (1e8) = 14 orders of magnitude
        assert should_scale_problem(problem)

    def test_infinite_capacity_ignored(self):
        """Infinite capacities don't affect scaling detection."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": None, "cost": 1.0},
            ],
        )

        assert not should_scale_problem(problem)


class TestScalingFactors:
    """Test computation of scaling factors."""

    def test_compute_factors_for_large_costs(self):
        """Large costs should be scaled down."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1e6},
            ],
        )

        factors = compute_scaling_factors(problem)

        # Geometric mean of costs is 1e6, so scale factor should be ~1e-6
        assert factors.cost_scale == pytest.approx(1e-6, rel=0.1)
        assert factors.enabled

    def test_compute_factors_for_small_costs(self):
        """Small costs should be scaled up."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1e-6},
            ],
        )

        factors = compute_scaling_factors(problem)

        # Geometric mean of costs is 1e-6, so scale factor should be ~1e6
        assert factors.cost_scale == pytest.approx(1e6, rel=0.1)

    def test_compute_factors_multiple_arcs(self):
        """Scaling based on geometric mean of all values."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1000.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -1000.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 100.0, "cost": 10.0},
                {"tail": "m", "head": "t", "capacity": 1000.0, "cost": 100.0},
            ],
        )

        factors = compute_scaling_factors(problem)

        # Should compute geometric means
        assert factors.enabled
        assert factors.cost_scale > 0
        assert factors.capacity_scale > 0
        assert factors.supply_scale > 0


class TestScalingTransformations:
    """Test scaling and unscaling transformations."""

    def test_scale_problem_applies_factors(self):
        """Scaling multiplies values by factors."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1000.0},
                {"id": "t", "supply": -1000.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 1000.0, "cost": 0.001},
            ],
        )

        factors = ScalingFactors(
            cost_scale=1000.0,
            capacity_scale=0.001,
            supply_scale=0.001,
            enabled=True,
        )

        scaled = scale_problem(problem, factors)

        # Check scaled values
        assert scaled.nodes["s"].supply == pytest.approx(1.0)  # 1000 * 0.001
        assert scaled.nodes["t"].supply == pytest.approx(-1.0)

        arc = next(a for a in scaled.arcs if a.tail == "s" and a.head == "t")
        assert arc.cost == pytest.approx(1.0)  # 0.001 * 1000
        assert arc.capacity == pytest.approx(1.0)  # 1000 * 0.001

    def test_scale_problem_preserves_structure(self):
        """Scaling preserves problem structure."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 50.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 50.0, "cost": 3.0},
            ],
        )

        factors = ScalingFactors(
            cost_scale=10.0, capacity_scale=0.1, supply_scale=0.01, enabled=True
        )
        scaled = scale_problem(problem, factors)

        # Same nodes and arcs
        assert set(scaled.nodes.keys()) == set(problem.nodes.keys())
        assert len(scaled.arcs) == len(problem.arcs)
        assert scaled.directed == problem.directed

    def test_scale_problem_handles_infinite_capacity(self):
        """Infinite capacity stays infinite after scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": None, "cost": 1.0},
            ],
        )

        factors = ScalingFactors(
            cost_scale=10.0, capacity_scale=0.5, supply_scale=0.01, enabled=True
        )
        scaled = scale_problem(problem, factors)

        arc = next(a for a in scaled.arcs if a.tail == "s" and a.head == "t")
        assert arc.capacity is None

    def test_unscale_solution_reverses_scaling(self):
        """Unscaling restores original units."""
        scaled_flows = {("s", "t"): 1.0}
        scaled_objective = 1.0

        factors = ScalingFactors(
            cost_scale=1000.0,  # Costs were multiplied by 1000
            capacity_scale=0.001,  # Capacities were multiplied by 0.001
            supply_scale=0.001,  # Supplies were multiplied by 0.001
            enabled=True,
        )

        flows, objective = unscale_solution(scaled_flows, scaled_objective, factors)

        # Flow should be divided by supply_scale
        assert flows[("s", "t")] == pytest.approx(1000.0)  # 1.0 / 0.001

        # Objective should be divided by (cost_scale * supply_scale)
        assert objective == pytest.approx(1.0)  # 1.0 / (1000.0 * 0.001)

    def test_unscale_disabled_returns_unchanged(self):
        """Unscaling with disabled factors returns original values."""
        flows = {("s", "t"): 100.0}
        objective = 200.0

        factors = ScalingFactors(enabled=False)

        unscaled_flows, unscaled_obj = unscale_solution(flows, objective, factors)

        assert unscaled_flows == flows
        assert unscaled_obj == objective


class TestScalingIntegration:
    """Test automatic scaling integration with solver."""

    def test_solve_with_auto_scaling_enabled(self):
        """Solver automatically scales and unscales."""
        # Problem with wide cost range
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1e8},
                {"id": "t", "supply": -1e8},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 1e8, "cost": 1e-6},
            ],
        )

        options = SolverOptions(auto_scale=True)
        result = solve_min_cost_flow(problem, options)

        assert result.status == "optimal"
        # Flow should be in original units
        assert result.flows[("s", "t")] == pytest.approx(1e8)
        # Objective should be in original units: 1e8 * 1e-6 = 100
        assert result.objective == pytest.approx(100.0)

    def test_solve_with_auto_scaling_disabled(self):
        """Solver works without scaling."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 100.0, "cost": 2.0},
            ],
        )

        options = SolverOptions(auto_scale=False)
        result = solve_min_cost_flow(problem, options)

        assert result.status == "optimal"
        assert result.flows[("s", "t")] == pytest.approx(100.0)
        assert result.objective == pytest.approx(200.0)

    def test_scaling_preserves_optimality(self):
        """Scaled and unscaled solutions are equivalent."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1000.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -1000.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 1000.0, "cost": 0.001},
                {"tail": "m", "head": "t", "capacity": 1000.0, "cost": 0.002},
                {"tail": "s", "head": "t", "capacity": 500.0, "cost": 0.01},
            ],
        )

        # Solve with scaling
        result_scaled = solve_min_cost_flow(problem, SolverOptions(auto_scale=True))

        # Solve without scaling
        result_unscaled = solve_min_cost_flow(problem, SolverOptions(auto_scale=False))

        assert result_scaled.status == result_unscaled.status
        assert result_scaled.objective == pytest.approx(result_unscaled.objective, rel=1e-6)

        # Flows should match (accounting for floating point precision)
        for arc in result_scaled.flows:
            assert result_scaled.flows[arc] == pytest.approx(result_unscaled.flows[arc], rel=1e-6)

    def test_scaling_improves_numerical_stability(self):
        """Scaling helps with extreme value ranges."""
        # Problem with very different scales that might cause numerical issues
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s1", "supply": 1e-8},
                {"id": "s2", "supply": 1e8},
                {"id": "t", "supply": -(1e-8 + 1e8)},
            ],
            arcs=[
                {"tail": "s1", "head": "t", "capacity": 1e-8, "cost": 1e8},
                {"tail": "s2", "head": "t", "capacity": 1e8, "cost": 1e-8},
            ],
        )

        # With scaling (should work well)
        result = solve_min_cost_flow(problem, SolverOptions(auto_scale=True))
        assert result.status == "optimal"

    def test_scaling_default_enabled(self):
        """Scaling is enabled by default."""
        options = SolverOptions()
        assert options.auto_scale is True


class TestScalingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scaling_with_zero_costs(self):
        """Zero costs are ignored in scaling computation."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 100.0, "cost": 0.0},
                {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1e6},
            ],
        )

        factors = compute_scaling_factors(problem)
        # Should compute factors based on non-zero costs only
        assert factors.enabled

    def test_scaling_with_lower_bounds(self):
        """Lower bounds are scaled along with capacities."""
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 1000.0, "cost": 1.0, "lower": 50.0},
            ],
        )

        factors = ScalingFactors(
            cost_scale=1.0, capacity_scale=0.01, supply_scale=0.01, enabled=True
        )
        scaled = scale_problem(problem, factors)

        arc = next(a for a in scaled.arcs if a.tail == "s" and a.head == "t")
        assert arc.lower == pytest.approx(0.5)  # 50 * 0.01
        assert arc.capacity == pytest.approx(10.0)  # 1000 * 0.01

    def test_scaling_disabled_when_all_factors_are_one(self):
        """Scaling is disabled when all factors would be 1.0."""
        # Problem where geometric mean of each category is exactly 1.0
        problem = build_problem(
            directed=True,
            tolerance=1e-6,
            nodes=[
                {"id": "s", "supply": 1.0},
                {"id": "t", "supply": -1.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 1.0, "cost": 1.0},
            ],
        )

        factors = compute_scaling_factors(problem)
        # All factors should be 1.0 (or very close)
        assert factors.cost_scale == pytest.approx(1.0, abs=1e-9)
        assert factors.capacity_scale == pytest.approx(1.0, abs=1e-9)
        assert factors.supply_scale == pytest.approx(1.0, abs=1e-9)
        # Scaling should be disabled since no actual scaling is needed
        assert factors.enabled is False
