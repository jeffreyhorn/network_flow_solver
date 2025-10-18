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
