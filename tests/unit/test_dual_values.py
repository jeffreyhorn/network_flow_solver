"""Tests for dual values (node potentials) in flow results."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


def test_dual_values_simple_feasible_problem():
    """Test that dual values are returned for a simple optimal solution."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"
    assert "s" in result.duals
    assert "t" in result.duals
    assert len(result.duals) == 2

    # Dual values should satisfy complementary slackness
    # Reduced cost formula: rc = cost + dual[tail] - dual[head]
    # For basic arcs at positive flow: rc = 0
    # So: cost + dual[tail] - dual[head] = 0
    # Therefore: dual[tail] - dual[head] = -cost
    dual_diff = result.duals["s"] - result.duals["t"]
    assert pytest.approx(dual_diff, abs=1e-6) == -2.0


def test_dual_values_multi_node_problem():
    """Test dual values for a problem with multiple nodes."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "b", "capacity": 10.0, "cost": 3.0},
        {"tail": "a", "head": "t", "capacity": 10.0, "cost": 2.0},
        {"tail": "b", "head": "t", "capacity": 10.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"
    assert len(result.duals) == 4
    assert "s" in result.duals
    assert "a" in result.duals
    assert "b" in result.duals
    assert "t" in result.duals

    # All dual values should be finite
    for node_id, dual in result.duals.items():
        assert abs(dual) < 1e10, f"Dual for {node_id} is too large: {dual}"


def test_dual_values_infeasible_problem():
    """Test that dual values are empty for infeasible problems."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "t", "supply": -5.0},
    ]
    # No arcs - infeasible
    arcs = []

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "infeasible"
    assert result.duals == {}


def test_dual_values_complementary_slackness():
    """Test complementary slackness conditions for dual values."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "a", "supply": 0.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 15.0, "cost": 3.0},
        {"tail": "a", "head": "t", "capacity": 15.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"

    # For each arc in the solution, check reduced cost
    # Reduced cost = cost + dual[tail] - dual[head]
    # For basic arcs (positive flow), reduced cost should be 0
    for (tail, head), flow in result.flows.items():
        if flow > 1e-6:  # Arc has positive flow
            cost = next(arc["cost"] for arc in arcs if arc["tail"] == tail and arc["head"] == head)
            reduced_cost = cost + result.duals[tail] - result.duals[head]
            assert abs(reduced_cost) < 1e-6, (
                f"Arc ({tail}, {head}) has positive flow but non-zero reduced cost: {reduced_cost}"
            )


def test_dual_values_sensitivity_interpretation():
    """Test that dual values can be used for sensitivity analysis."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 5.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"
    original_objective = result.objective

    # Increase supply at s by 1 unit (and demand at t)
    nodes_modified = [
        {"id": "s", "supply": 11.0},
        {"id": "t", "supply": -11.0},
    ]
    problem2 = build_problem(nodes=nodes_modified, arcs=arcs, directed=True, tolerance=1e-6)
    solver2 = NetworkSimplex(problem2)
    result2 = solver2.solve()

    # The change in objective equals the arc cost since we send 1 more unit
    objective_change = result2.objective - original_objective
    assert pytest.approx(objective_change, abs=1e-6) == 5.0

    # Verify the dual relationship: dual[s] - dual[t] = -cost (from complementary slackness)
    assert pytest.approx(result.duals["s"] - result.duals["t"], abs=1e-6) == -5.0


def test_dual_values_zero_cost_arcs():
    """Test dual values when some arcs have zero cost."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "a", "supply": 0.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 10.0, "cost": 0.0},  # Zero cost
        {"tail": "a", "head": "t", "capacity": 10.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"
    assert len(result.duals) == 3

    # For zero-cost arc with positive flow: dual[s] - dual[a] should equal 0
    if result.flows.get(("s", "a"), 0) > 1e-6:
        reduced_cost = 0.0 + result.duals["s"] - result.duals["a"]
        assert abs(reduced_cost) < 1e-6
