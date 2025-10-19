"""Unit tests for warm-start functionality.

NOTE: Warm-start is currently experimental and has known issues.
These tests document the expected behavior for when warm-start is fully implemented.
Most tests are marked as xfail (expected to fail) until the implementation is complete.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import Basis, build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402

# Mark for experimental warm-start feature
pytestmark = pytest.mark.xfail(reason="Warm-start is experimental and not fully implemented yet")


class TestWarmStartBasics:
    """Test basic warm-start functionality."""

    def test_warm_start_with_none_basis(self):
        """Test that None basis is handled gracefully (cold start)."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem, warm_start_basis=None)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)
        assert result.basis is not None

    def test_warm_start_identical_problem(self):
        """Test warm-start on identical problem should converge immediately."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # First solve
        result1 = solve_min_cost_flow(problem)
        assert result1.status == "optimal"

        # Second solve with warm-start on identical problem
        result2 = solve_min_cost_flow(problem, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        assert result2.objective == pytest.approx(result1.objective)
        # Should converge very quickly (0 or very few iterations)
        assert result2.iterations <= result1.iterations

    def test_warm_start_simple_capacity_increase(self):
        """Test warm-start with capacity increase."""
        nodes = [
            {"id": "s", "supply": 100.0},
            {"id": "t", "supply": -100.0},
        ]

        # Original problem with limited capacity
        problem1 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Increased capacity - should still be optimal
        problem2 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "t", "capacity": 150.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        assert result2.objective == pytest.approx(result1.objective)


class TestWarmStartEdgeCases:
    """Test edge cases and error handling for warm-start."""

    def test_warm_start_incompatible_basis_different_arcs(self):
        """Test warm-start with basis from completely different problem."""
        # Problem 1: simple 2-node problem
        problem1 = build_problem(
            nodes=[
                {"id": "A", "supply": 10.0},
                {"id": "B", "supply": -10.0},
            ],
            arcs=[
                {"tail": "A", "head": "B", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Problem 2: different nodes entirely
        problem2 = build_problem(
            nodes=[
                {"id": "X", "supply": 5.0},
                {"id": "Y", "supply": -5.0},
            ],
            arcs=[
                {"tail": "X", "head": "Y", "capacity": 10.0, "cost": 2.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Warm-start should fail gracefully and fall back to cold start
        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        assert result2.objective == pytest.approx(10.0)

    def test_warm_start_with_empty_basis(self):
        """Test warm-start with an empty basis."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Create empty basis
        empty_basis = Basis(tree_arcs=set(), arc_flows={})

        # Should fall back to cold start
        result = solve_min_cost_flow(problem, warm_start_basis=empty_basis)
        assert result.status == "optimal"

    def test_warm_start_capacity_decrease_infeasible_flow(self):
        """Test warm-start when capacity decrease makes previous flow infeasible."""
        nodes = [
            {"id": "s", "supply": 100.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -100.0},
        ]

        # Original problem
        problem1 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 100.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Decrease capacity - previous flow of 100 won't fit
        problem2 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 50.0, "cost": 1.0},  # Reduced
                {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Should handle gracefully via Phase 1
        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "infeasible"


class TestWarmStartModifications:
    """Test warm-start with various problem modifications."""

    def test_warm_start_supply_change(self):
        """Test warm-start with supply/demand changes."""
        arcs = [
            {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
        ]

        # Original problem
        problem1 = build_problem(
            nodes=[
                {"id": "s", "supply": 50.0},
                {"id": "t", "supply": -50.0},
            ],
            arcs=arcs,
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"
        assert result1.objective == pytest.approx(50.0)

        # Increased demand
        problem2 = build_problem(
            nodes=[
                {"id": "s", "supply": 75.0},
                {"id": "t", "supply": -75.0},
            ],
            arcs=arcs,
            directed=True,
            tolerance=1e-6,
        )

        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        assert result2.objective == pytest.approx(75.0)

    def test_warm_start_cost_change(self):
        """Test warm-start with cost changes."""
        nodes = [
            {"id": "s", "supply": 10.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -10.0},
        ]

        # Original costs
        problem1 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 1.0},
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 5.0},  # Expensive direct
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Change costs - direct route becomes cheaper
        problem2 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 3.0},  # More expensive
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 3.0},  # More expensive
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},  # Cheaper
            ],
            directed=True,
            tolerance=1e-6,
        )

        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"

    def test_warm_start_add_arc(self):
        """Test warm-start when adding a new arc to the network."""
        nodes = [
            {"id": "s", "supply": 10.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -10.0},
        ]

        # Original problem - must go through middle node
        problem1 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 2.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"
        original_obj = result1.objective

        # Add direct arc
        problem2 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 2.0},
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},  # New cheaper direct
            ],
            directed=True,
            tolerance=1e-6,
        )

        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        # Should find better solution with new arc
        assert result2.objective <= original_obj

    def test_warm_start_remove_arc(self):
        """Test warm-start when removing an arc from the network."""
        nodes = [
            {"id": "s", "supply": 10.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -10.0},
        ]

        # Original problem with direct arc
        problem1 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 2.0},
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Remove direct arc (basis will contain arc that no longer exists)
        problem2 = build_problem(
            nodes=nodes,
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 2.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Should fall back to cold start since basis arc is missing
        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"


class TestWarmStartSequential:
    """Test sequential warm-starts (chaining multiple solves)."""

    def test_sequential_warm_starts(self):
        """Test multiple sequential warm-starts."""
        nodes = [
            {"id": "s", "supply": 100.0},
            {"id": "t", "supply": -100.0},
        ]

        capacities = [100.0, 110.0, 120.0, 130.0]
        basis = None
        prev_result = None

        for capacity in capacities:
            problem = build_problem(
                nodes=nodes,
                arcs=[
                    {"tail": "s", "head": "t", "capacity": capacity, "cost": 1.0},
                ],
                directed=True,
                tolerance=1e-6,
            )

            result = solve_min_cost_flow(problem, warm_start_basis=basis)
            assert result.status == "optimal"

            # Update basis for next iteration
            basis = result.basis

            # Objective should stay the same (already at capacity)
            if prev_result:
                assert result.objective == pytest.approx(prev_result.objective)

            prev_result = result

    def test_sequential_warm_starts_with_increasing_demand(self):
        """Test sequential warm-starts with gradually increasing demand."""
        arcs = [
            {"tail": "s", "head": "t", "capacity": 200.0, "cost": 1.0},
        ]

        demands = [50.0, 75.0, 100.0, 125.0, 150.0]
        basis = None
        prev_objective = 0.0

        for demand in demands:
            problem = build_problem(
                nodes=[
                    {"id": "s", "supply": demand},
                    {"id": "t", "supply": -demand},
                ],
                arcs=arcs,
                directed=True,
                tolerance=1e-6,
            )

            result = solve_min_cost_flow(problem, warm_start_basis=basis)
            assert result.status == "optimal"
            assert result.objective == pytest.approx(demand)

            # Objective should increase with demand
            assert result.objective >= prev_objective

            basis = result.basis
            prev_objective = result.objective


class TestWarmStartComplexNetworks:
    """Test warm-start on more complex network structures."""

    def test_warm_start_multi_commodity_style(self):
        """Test warm-start on a network with multiple sources and sinks."""
        nodes = [
            {"id": "s1", "supply": 30.0},
            {"id": "s2", "supply": 40.0},
            {"id": "m", "supply": 0.0},
            {"id": "t1", "supply": -35.0},
            {"id": "t2", "supply": -35.0},
        ]

        arcs_base = [
            {"tail": "s1", "head": "m", "capacity": 50.0, "cost": 1.0},
            {"tail": "s2", "head": "m", "capacity": 50.0, "cost": 1.0},
            {"tail": "m", "head": "t1", "capacity": 50.0, "cost": 1.0},
            {"tail": "m", "head": "t2", "capacity": 50.0, "cost": 1.0},
        ]

        problem1 = build_problem(nodes=nodes, arcs=arcs_base, directed=True, tolerance=1e-6)
        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Increase capacity on one arc
        arcs_modified = [
            {"tail": "s1", "head": "m", "capacity": 60.0, "cost": 1.0},  # Increased
            {"tail": "s2", "head": "m", "capacity": 50.0, "cost": 1.0},
            {"tail": "m", "head": "t1", "capacity": 50.0, "cost": 1.0},
            {"tail": "m", "head": "t2", "capacity": 50.0, "cost": 1.0},
        ]

        problem2 = build_problem(nodes=nodes, arcs=arcs_modified, directed=True, tolerance=1e-6)
        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"
        assert result2.objective == pytest.approx(result1.objective)

    def test_warm_start_with_cycles(self):
        """Test warm-start on a network with cycles."""
        nodes = [
            {"id": "s", "supply": 10.0},
            {"id": "a", "supply": 0.0},
            {"id": "b", "supply": 0.0},
            {"id": "t", "supply": -10.0},
        ]

        # Network with a cycle (s -> a -> b -> a -> t is possible)
        arcs = [
            {"tail": "s", "head": "a", "capacity": 20.0, "cost": 1.0},
            {"tail": "a", "head": "b", "capacity": 20.0, "cost": 1.0},
            {"tail": "b", "head": "a", "capacity": 20.0, "cost": 2.0},  # Back edge
            {"tail": "a", "head": "t", "capacity": 20.0, "cost": 1.0},
        ]

        problem1 = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        result1 = solve_min_cost_flow(problem1)
        assert result1.status == "optimal"

        # Modify costs
        arcs_modified = [
            {"tail": "s", "head": "a", "capacity": 20.0, "cost": 2.0},  # Increased
            {"tail": "a", "head": "b", "capacity": 20.0, "cost": 1.0},
            {"tail": "b", "head": "a", "capacity": 20.0, "cost": 2.0},
            {"tail": "a", "head": "t", "capacity": 20.0, "cost": 2.0},  # Increased
        ]

        problem2 = build_problem(nodes=nodes, arcs=arcs_modified, directed=True, tolerance=1e-6)
        result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
        assert result2.status == "optimal"


class TestWarmStartBasisExtraction:
    """Test basis extraction and validation."""

    def test_basis_extraction_contains_correct_arcs(self):
        """Test that extracted basis contains the correct tree arcs."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 20.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 20.0, "cost": 1.0},
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 5.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem)
        assert result.status == "optimal"
        assert result.basis is not None

        # Basis should have tree_arcs and arc_flows
        assert isinstance(result.basis.tree_arcs, set)
        assert isinstance(result.basis.arc_flows, dict)

        # Should have n-1 tree arcs for n nodes (3 nodes = 2 tree arcs)
        assert len(result.basis.tree_arcs) == 2

        # All tree arcs should have flows
        for arc in result.basis.tree_arcs:
            assert arc in result.basis.arc_flows

    def test_basis_flows_match_result_flows(self):
        """Test that basis arc flows match the result flows."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem)
        assert result.status == "optimal"

        # Basis flows should match result flows for tree arcs
        for arc in result.basis.tree_arcs:
            assert arc in result.flows
            assert result.basis.arc_flows[arc] == pytest.approx(result.flows[arc])
