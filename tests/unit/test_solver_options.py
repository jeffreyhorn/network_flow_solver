"""Tests for SolverOptions configuration."""

import pytest

from network_solver import (
    InvalidProblemError,
    SolverOptions,
    build_problem,
    solve_min_cost_flow,
)
from network_solver.simplex import NetworkSimplex


def test_solver_options_defaults():
    """Test that SolverOptions has sensible default values."""
    options = SolverOptions()
    assert options.max_iterations is None
    assert options.tolerance == 1e-6
    assert options.pricing_strategy == "devex"
    assert options.block_size is None
    assert options.ft_update_limit == 64


def test_solver_options_custom_values():
    """Test that SolverOptions accepts custom values."""
    options = SolverOptions(
        max_iterations=1000,
        tolerance=1e-8,
        pricing_strategy="dantzig",
        block_size=50,
        ft_update_limit=100,
    )
    assert options.max_iterations == 1000
    assert options.tolerance == 1e-8
    assert options.pricing_strategy == "dantzig"
    assert options.block_size == 50
    assert options.ft_update_limit == 100


def test_solver_options_invalid_tolerance():
    """Test that invalid tolerance raises an error."""
    with pytest.raises(InvalidProblemError, match="Tolerance must be positive"):
        SolverOptions(tolerance=0.0)

    with pytest.raises(InvalidProblemError, match="Tolerance must be positive"):
        SolverOptions(tolerance=-1e-6)


def test_solver_options_invalid_pricing_strategy():
    """Test that invalid pricing strategy raises an error."""
    with pytest.raises(InvalidProblemError, match="Invalid pricing strategy"):
        SolverOptions(pricing_strategy="invalid")

    with pytest.raises(InvalidProblemError, match="Invalid pricing strategy"):
        SolverOptions(pricing_strategy="steepest_edge")


def test_solver_options_invalid_block_size():
    """Test that invalid block size raises an error."""
    with pytest.raises(InvalidProblemError, match="Block size must be positive"):
        SolverOptions(block_size=0)

    with pytest.raises(InvalidProblemError, match="Block size must be positive"):
        SolverOptions(block_size=-10)

    with pytest.raises(InvalidProblemError, match="Invalid block_size"):
        SolverOptions(block_size="invalid")


def test_solver_options_invalid_ft_update_limit():
    """Test that invalid FT update limit raises an error."""
    with pytest.raises(InvalidProblemError, match="FT update limit must be positive"):
        SolverOptions(ft_update_limit=0)

    with pytest.raises(InvalidProblemError, match="FT update limit must be positive"):
        SolverOptions(ft_update_limit=-5)


def test_solve_with_custom_tolerance():
    """Test that custom tolerance is applied correctly."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Use tighter tolerance
    options = SolverOptions(tolerance=1e-10)
    result = solve_min_cost_flow(problem, options=options)

    assert result.status == "optimal"
    assert result.flows[("s", "t")] == pytest.approx(10.0, abs=1e-10)


def test_solve_with_dantzig_pricing():
    """Test that Dantzig pricing strategy works correctly."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 100.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 50.0, "cost": 3.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(pricing_strategy="dantzig")
    result = solve_min_cost_flow(problem, options=options)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(200.0, abs=1e-6)


def test_solve_with_devex_pricing():
    """Test that Devex pricing strategy works correctly (default)."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 100.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 100.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 50.0, "cost": 3.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(pricing_strategy="devex")
    result = solve_min_cost_flow(problem, options=options)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(200.0, abs=1e-6)


def test_solve_with_custom_max_iterations():
    """Test that max_iterations from options is respected."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 150.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Set a very low iteration limit
    options = SolverOptions(max_iterations=1)
    result = solve_min_cost_flow(problem, options=options)

    # Should hit iteration limit or solve (depending on problem complexity)
    assert result.status in ("optimal", "iteration_limit")
    assert result.iterations <= 1


def test_solve_with_custom_block_size():
    """Test that custom block size is applied correctly."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size=1)

    solver = NetworkSimplex(problem, options=options)
    assert solver.block_size == 1

    result = solver.solve()
    assert result.status == "optimal"


def test_solve_with_custom_ft_update_limit():
    """Test that FT update limit affects solver behavior."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m1", "capacity": 100.0, "cost": 1.0},
        {"tail": "s", "head": "m2", "capacity": 100.0, "cost": 2.0},
        {"tail": "m1", "head": "t", "capacity": 100.0, "cost": 1.0},
        {"tail": "m2", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Use very small FT update limit to force frequent rebuilds
    options = SolverOptions(ft_update_limit=2)
    solver = NetworkSimplex(problem, options=options)
    result = solver.solve()

    assert result.status == "optimal"
    # Should have forced at least one rebuild if problem required multiple pivots
    # (We can't assert on ft_rebuilds since simple problems may not need rebuilds)


def test_max_iterations_parameter_overrides_options():
    """Test that max_iterations parameter overrides options value."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 150.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Set options with high limit
    options = SolverOptions(max_iterations=10000)

    # Override with low limit via parameter
    result = solve_min_cost_flow(problem, options=options, max_iterations=1)

    assert result.iterations <= 1


def test_optimal_at_exact_iteration_limit():
    """Test that solver reports 'optimal' when optimum is found at exact iteration limit.

    Regression test for issue where solver incorrectly reported 'iteration_limit'
    when the last allowed pivot produced the optimal solution.
    """
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 10.0, "cost": 2.0},
        {"tail": "s", "head": "b", "capacity": 10.0, "cost": 3.0},
        {"tail": "a", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "b", "head": "t", "capacity": 10.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # First solve without limit to determine required iterations
    result_unlimited = solve_min_cost_flow(problem)
    assert result_unlimited.status == "optimal"
    required_iterations = result_unlimited.iterations

    # Now solve with exact iteration budget - should still report optimal
    result_exact = solve_min_cost_flow(problem, max_iterations=required_iterations)
    assert result_exact.status == "optimal"
    assert result_exact.iterations == required_iterations
    assert result_exact.objective == pytest.approx(30.0)

    # Verify one less iteration is insufficient
    result_insufficient = solve_min_cost_flow(problem, max_iterations=required_iterations - 1)
    assert result_insufficient.status == "iteration_limit"
    assert result_insufficient.iterations == required_iterations - 1


def test_options_with_none_values():
    """Test that None values in options fall back to defaults."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Options with None for max_iterations and block_size
    options = SolverOptions(max_iterations=None, block_size=None)
    solver = NetworkSimplex(problem, options=options)

    # Should use defaults
    assert solver.block_size == max(1, solver.actual_arc_count // 8)

    result = solver.solve()
    assert result.status == "optimal"


def test_solver_options_applied_to_network_simplex():
    """Test that SolverOptions are properly applied to NetworkSimplex."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(
        tolerance=1e-8,
        pricing_strategy="dantzig",
        block_size=5,
        ft_update_limit=50,
    )

    solver = NetworkSimplex(problem, options=options)

    assert solver.tolerance == 1e-8
    assert solver.options.pricing_strategy == "dantzig"
    assert solver.block_size == 5
    assert solver.options.ft_update_limit == 50


def test_block_size_auto_string():
    """Test that block_size='auto' enables auto-tuning."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    assert solver.auto_tune_block_size is True
    # Should compute initial block size based on heuristic
    assert solver.block_size > 0


def test_block_size_none_enables_auto_tuning():
    """Test that block_size=None enables auto-tuning (default)."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size=None)
    solver = NetworkSimplex(problem, options=options)

    assert solver.auto_tune_block_size is True


def test_block_size_int_disables_auto_tuning():
    """Test that explicit int block_size disables auto-tuning."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size=25)
    solver = NetworkSimplex(problem, options=options)

    assert solver.auto_tune_block_size is False
    assert solver.block_size == 25


def test_initial_block_size_heuristic_very_small():
    """Test initial block size heuristic for very small problems (<100 arcs)."""
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(10)]
    nodes[0]["supply"] = 50.0
    nodes[-1]["supply"] = -50.0

    # Create 50 arcs (very small problem)
    arcs = []
    for i in range(5):
        for j in range(i + 1, min(i + 11, 10)):
            arcs.append({"tail": f"n{i}", "head": f"n{j}", "capacity": 10.0, "cost": 1.0})

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Very small: should use num_arcs // 4
    expected = max(1, solver.actual_arc_count // 4)
    assert solver.block_size == expected


def test_initial_block_size_heuristic_small():
    """Test initial block size heuristic for small problems (100-1000 arcs)."""
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(25)]
    nodes[0]["supply"] = 100.0
    nodes[-1]["supply"] = -100.0

    # Create ~250 arcs (small problem)
    arcs = []
    for i in range(25):
        for j in range(i + 1, min(i + 11, 25)):
            arcs.append({"tail": f"n{i}", "head": f"n{j}", "capacity": 10.0, "cost": float(i + j)})

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Small: should use num_arcs // 4
    expected = max(1, solver.actual_arc_count // 4)
    assert solver.block_size == expected


def test_initial_block_size_heuristic_medium():
    """Test initial block size heuristic for medium problems (1000-10000 arcs)."""
    # Create a problem with ~1500 arcs by building a denser graph
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(50)]
    nodes[0]["supply"] = 500.0
    nodes[-1]["supply"] = -500.0

    arcs = []
    for i in range(50):
        for j in range(i + 1, min(i + 31, 50)):
            arcs.append({"tail": f"n{i}", "head": f"n{j}", "capacity": 20.0, "cost": float(i + j)})

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Medium: should use num_arcs // 8
    expected = max(1, solver.actual_arc_count // 8)
    assert solver.block_size == expected


def test_auto_tuning_solves_correctly():
    """Test that auto-tuning doesn't break correctness."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m1", "capacity": 100.0, "cost": 1.0},
        {"tail": "s", "head": "m2", "capacity": 100.0, "cost": 2.0},
        {"tail": "m1", "head": "t", "capacity": 100.0, "cost": 1.0},
        {"tail": "m2", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Solve with auto-tuning
    options_auto = SolverOptions(block_size="auto")
    result_auto = solve_min_cost_flow(problem, options=options_auto)

    # Solve with fixed block size
    options_fixed = SolverOptions(block_size=10)
    result_fixed = solve_min_cost_flow(problem, options=options_fixed)

    # Both should find optimal solution
    assert result_auto.status == "optimal"
    assert result_fixed.status == "optimal"
    assert result_auto.objective == pytest.approx(result_fixed.objective, abs=1e-6)
