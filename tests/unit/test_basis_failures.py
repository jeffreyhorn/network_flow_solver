"""Test basis-related edge cases through solver API.

This module tests scenarios that exercise basis operations and may trigger
edge cases or numerical warnings during the solve process.

Target coverage areas:
- Warm-start with invalid/incompatible basis structures
- Problems that stress basis updates during pivoting
- Scenarios with many basis changes
"""

import pytest

from network_solver import build_problem, solve_min_cost_flow
from network_solver.data import Basis


class TestWarmStartBasisFailures:
    """Test warm-start scenarios where basis may be rejected or cause issues."""

    def test_warm_start_with_invalid_basis_structure(self):
        """Test warm-start with a basis that doesn't form a valid tree.

        This tests the validation/rejection path when a warm-start basis
        is provided but doesn't satisfy the spanning tree property.
        """
        # Create a simple transportation problem
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
                {"tail": "s2", "head": "t1", "capacity": 100.0, "cost": 4.0},
                {"tail": "s2", "head": "t2", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Create an invalid basis with a cycle (not a tree)
        invalid_basis = Basis(
            tree_arcs={
                ("s1", "t1"),
                ("t1", "s2"),  # Creates a cycle
                ("s2", "t2"),
            },
            arc_flows={
                ("s1", "t1"): 50.0,
                ("t1", "s2"): 10.0,
                ("s2", "t2"): 40.0,
            },
        )

        # Solver should either reject this basis or handle it gracefully
        result = solve_min_cost_flow(problem, warm_start_basis=invalid_basis)

        # Should still find optimal solution (may have ignored invalid basis)
        assert result.status == "optimal"
        assert result.objective is not None

    def test_warm_start_with_empty_basis(self):
        """Test warm-start with an empty basis.

        This tests the handling when an empty/minimal basis is provided.
        """
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 30.0},
                {"id": "t", "supply": -30.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 50.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Create an empty basis
        empty_basis = Basis(tree_arcs=set(), arc_flows={})

        # Solver should handle empty basis (likely ignoring it)
        result = solve_min_cost_flow(problem, warm_start_basis=empty_basis)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(30.0)


class TestManyBasisChanges:
    """Test problems that require many basis changes (pivots)."""

    def test_problem_requiring_many_pivots(self):
        """Test a problem designed to require many pivot operations.

        This stresses the basis update mechanisms by creating a problem
        where the initial basis is far from optimal.
        """
        # Create a larger transportation problem
        num_sources = 5
        num_sinks = 5
        supply_per_source = 20.0

        nodes = []
        for i in range(num_sources):
            nodes.append({"id": f"s{i}", "supply": supply_per_source})
        for j in range(num_sinks):
            nodes.append({"id": f"t{j}", "supply": -supply_per_source})

        # Create arcs with costs designed to make initial basis suboptimal
        arcs = []
        for i in range(num_sources):
            for j in range(num_sinks):
                # Costs that favor diagonal routing (s0->t0, s1->t1, etc.)
                cost = abs(i - j) + 1.0
                arcs.append(
                    {
                        "tail": f"s{i}",
                        "head": f"t{j}",
                        "capacity": 50.0,
                        "cost": cost,
                    }
                )

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        # Solve without warm-start (many pivots expected)
        result = solve_min_cost_flow(problem)

        assert result.status == "optimal"
        assert result.objective is not None
        assert result.iterations > 0  # Should require some iterations

    def test_problem_with_many_degenerate_pivots(self):
        """Test a problem that triggers many degenerate pivot operations.

        Degenerate pivots can stress basis update logic.
        """
        # Create a highly symmetric problem (tends to create degeneracy)
        nodes = [
            {"id": "s1", "supply": 25.0},
            {"id": "s2", "supply": 25.0},
            {"id": "s3", "supply": 25.0},
            {"id": "s4", "supply": 25.0},
            {"id": "t1", "supply": -25.0},
            {"id": "t2", "supply": -25.0},
            {"id": "t3", "supply": -25.0},
            {"id": "t4", "supply": -25.0},
        ]

        # All arcs have the same cost (creates degeneracy)
        arcs = []
        for i in range(1, 5):
            for j in range(1, 5):
                arcs.append(
                    {
                        "tail": f"s{i}",
                        "head": f"t{j}",
                        "capacity": 30.0,
                        "cost": 1.0,  # All same cost
                    }
                )

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)

        assert result.status == "optimal"
        # Objective should be total supply * cost
        assert result.objective == pytest.approx(100.0)


class TestBasisNumericalStress:
    """Test problems that may stress numerical stability of basis operations."""

    def test_problem_with_extreme_cost_ranges_during_pivots(self):
        """Test problem with extreme cost variations during pivoting.

        This can stress the numerical stability of reduced cost calculations
        and basis updates.
        """
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 100.0},
                {"id": "s2", "supply": 100.0},
                {"id": "t1", "supply": -100.0},
                {"id": "t2", "supply": -100.0},
            ],
            arcs=[
                # Mix of very small and very large costs
                {"tail": "s1", "head": "t1", "capacity": 150.0, "cost": 1e-8},
                {"tail": "s1", "head": "t2", "capacity": 150.0, "cost": 1e8},
                {"tail": "s2", "head": "t1", "capacity": 150.0, "cost": 1e7},
                {"tail": "s2", "head": "t2", "capacity": 150.0, "cost": 1e-7},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem)

        assert result.status == "optimal"
        assert result.objective is not None

    def test_problem_with_tight_capacities_forcing_many_updates(self):
        """Test problem with tight capacities requiring many basis changes.

        Tight capacities can force the solver to make many basis changes
        as arcs hit their capacity limits.
        """
        # Create a problem where capacities are just at the needed flow
        nodes = [
            {"id": "s1", "supply": 30.0},
            {"id": "s2", "supply": 20.0},
            {"id": "m", "supply": 0.0},  # Intermediate node
            {"id": "t1", "supply": -25.0},
            {"id": "t2", "supply": -25.0},
        ]

        arcs = [
            # Tight capacities through intermediate node
            {"tail": "s1", "head": "m", "capacity": 30.0, "cost": 1.0},
            {"tail": "s2", "head": "m", "capacity": 20.0, "cost": 1.0},
            {"tail": "m", "head": "t1", "capacity": 25.0, "cost": 2.0},
            {"tail": "m", "head": "t2", "capacity": 25.0, "cost": 2.0},
            # Alternative expensive paths
            {"tail": "s1", "head": "t1", "capacity": 50.0, "cost": 100.0},
            {"tail": "s2", "head": "t2", "capacity": 50.0, "cost": 100.0},
        ]

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)

        assert result.status == "optimal"
        assert result.objective is not None
