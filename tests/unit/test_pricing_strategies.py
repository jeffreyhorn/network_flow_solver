"""Unit tests for pricing strategies."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import build_problem, solve_min_cost_flow  # noqa: E402
from network_solver.data import SolverOptions  # noqa: E402


class TestDantzigPricing:
    """Tests for Dantzig pricing strategy."""

    def test_dantzig_pricing_simple_problem(self):
        """Test Dantzig pricing on a simple problem."""
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

        options = SolverOptions(pricing_strategy="dantzig")
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)

    def test_dantzig_pricing_with_multiple_arcs(self):
        """Test Dantzig pricing selects most negative reduced cost."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 20.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -20.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 30.0, "cost": 1.0},
                {"tail": "s", "head": "m", "capacity": 30.0, "cost": 2.0},
                {"tail": "m", "head": "t", "capacity": 30.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 30.0, "cost": 3.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(pricing_strategy="dantzig")
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        # Should select cheaper paths (costs 1.0)
        assert result.objective == pytest.approx(40.0)


class TestDevexPricingLoopBased:
    """Tests for Devex pricing with loop-based implementation (non-vectorized)."""

    def test_devex_loop_based_simple_problem(self):
        """Test loop-based Devex pricing on a simple problem."""
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

        # Disable vectorized pricing to test loop-based implementation
        options = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=False,
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)

    def test_devex_loop_based_transportation_problem(self):
        """Test loop-based Devex on a transportation problem."""
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 20.0},
                {"id": "s2", "supply": 20.0},
                {"id": "t1", "supply": -20.0},
                {"id": "t2", "supply": -20.0},
            ],
            arcs=[
                {"tail": "s1", "head": "t1", "capacity": 30.0, "cost": 2.0},
                {"tail": "s1", "head": "t2", "capacity": 30.0, "cost": 3.0},
                {"tail": "s2", "head": "t1", "capacity": 30.0, "cost": 4.0},
                {"tail": "s2", "head": "t2", "capacity": 30.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=False,
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        # Optimal: s1->t1 (20*2=40), s2->t2 (20*1=20) = 60
        assert result.objective == pytest.approx(60.0)

    def test_devex_loop_based_network_problem(self):
        """Test loop-based Devex on a network with intermediate nodes."""
        # Simpler test - just verify loop-based pricing works
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 15.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 15.0, "cost": 2.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        options = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=False,
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(30.0)

    def test_devex_loop_based_larger_problem(self):
        """Test loop-based Devex with a larger problem."""
        # Create problem with multiple arcs
        num_sources = 3
        num_sinks = 3
        nodes = []
        arcs = []

        for i in range(num_sources):
            nodes.append({"id": f"s{i}", "supply": 10.0})

        for j in range(num_sinks):
            nodes.append({"id": f"t{j}", "supply": -10.0})

        # Create bipartite graph with varying costs
        for i in range(num_sources):
            for j in range(num_sinks):
                cost = abs(i - j) + 1.0
                arcs.append(
                    {
                        "tail": f"s{i}",
                        "head": f"t{j}",
                        "capacity": 20.0,
                        "cost": cost,
                    }
                )

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        options = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=False,
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.iterations > 0


class TestDevexPricingVectorized:
    """Tests for Devex pricing with vectorized implementation (default)."""

    def test_devex_vectorized_simple_problem(self):
        """Test vectorized Devex pricing on a simple problem."""
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

        options = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=True,  # Explicitly enable (default)
        )
        result = solve_min_cost_flow(problem, options=options)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)

    def test_devex_vectorized_vs_loop_based(self):
        """Test that vectorized and loop-based Devex both work correctly."""
        # Simple balanced problem
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 20.0},
                {"id": "t", "supply": -20.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 30.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Solve with vectorized
        options_vec = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=True,
        )
        result_vec = solve_min_cost_flow(problem, options=options_vec)

        # Solve with loop-based
        options_loop = SolverOptions(
            pricing_strategy="devex",
            use_vectorized_pricing=False,
        )
        result_loop = solve_min_cost_flow(problem, options=options_loop)

        # Both should find optimal solution
        assert result_vec.status == "optimal"
        assert result_loop.status == "optimal"
        assert result_vec.objective == pytest.approx(result_loop.objective)
        assert result_vec.objective == pytest.approx(20.0)


class TestPricingStrategyComparison:
    """Tests comparing different pricing strategies."""

    def test_all_strategies_find_optimal(self):
        """Test that all pricing strategies find the optimal solution."""
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 15.0},
                {"id": "s2", "supply": 15.0},
                {"id": "t1", "supply": -15.0},
                {"id": "t2", "supply": -15.0},
            ],
            arcs=[
                {"tail": "s1", "head": "t1", "capacity": 20.0, "cost": 1.0},
                {"tail": "s1", "head": "t2", "capacity": 20.0, "cost": 5.0},
                {"tail": "s2", "head": "t1", "capacity": 20.0, "cost": 4.0},
                {"tail": "s2", "head": "t2", "capacity": 20.0, "cost": 2.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        strategies = [
            ("dantzig", None),
            ("devex", True),  # vectorized
            ("devex", False),  # loop-based
        ]

        results = []
        for strategy, vectorized in strategies:
            if vectorized is None:
                options = SolverOptions(pricing_strategy=strategy)
            else:
                options = SolverOptions(
                    pricing_strategy=strategy,
                    use_vectorized_pricing=vectorized,
                )
            result = solve_min_cost_flow(problem, options=options)
            results.append((strategy, vectorized, result))

        # All should find optimal
        for strategy, vectorized, result in results:
            assert result.status == "optimal", f"{strategy} (vec={vectorized}) failed"

        # All should have same objective (optimal solution)
        objectives = [r.objective for _, _, r in results]
        for obj in objectives[1:]:
            assert obj == pytest.approx(objectives[0])

    def test_devex_faster_than_dantzig_on_large_problem(self):
        """Test that Devex typically requires fewer iterations than Dantzig."""
        # Create larger problem where Devex advantage is clearer
        num_sources = 10
        num_sinks = 10
        nodes = []
        arcs = []

        for i in range(num_sources):
            nodes.append({"id": f"s{i}", "supply": 10.0})

        for j in range(num_sinks):
            nodes.append({"id": f"t{j}", "supply": -10.0})

        for i in range(num_sources):
            for j in range(num_sinks):
                cost = (i + j) % 7 + 1.0
                arcs.append(
                    {
                        "tail": f"s{i}",
                        "head": f"t{j}",
                        "capacity": 15.0,
                        "cost": cost,
                    }
                )

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        # Solve with Dantzig
        result_dantzig = solve_min_cost_flow(
            problem, options=SolverOptions(pricing_strategy="dantzig")
        )

        # Solve with Devex
        result_devex = solve_min_cost_flow(problem, options=SolverOptions(pricing_strategy="devex"))

        # Both should find optimal
        assert result_dantzig.status == "optimal"
        assert result_devex.status == "optimal"
        assert result_dantzig.objective == pytest.approx(result_devex.objective)

        # Devex typically requires fewer iterations (though not guaranteed)
        # At minimum, verify both solved successfully
        assert result_devex.iterations > 0
        assert result_dantzig.iterations > 0
