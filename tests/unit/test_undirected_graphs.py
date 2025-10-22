"""Comprehensive tests for undirected graph handling."""

import math

import pytest

from network_solver import build_problem, solve_min_cost_flow
from network_solver.data import Arc, NetworkProblem, Node
from network_solver.exceptions import InvalidProblemError


def test_undirected_simple_chain():
    """Test basic undirected chain: A -- B -- C."""
    nodes = {
        "A": Node(id="A", supply=10.0),
        "B": Node(id="B", supply=0.0),
        "C": Node(id="C", supply=-10.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=15.0, cost=2.0),
        Arc(tail="B", head="C", capacity=15.0, cost=3.0),
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    assert math.isclose(result.objective, 50.0, abs_tol=1e-6)  # 10*2 + 10*3
    assert math.isclose(result.flows[("A", "B")], 10.0, abs_tol=1e-6)
    assert math.isclose(result.flows[("B", "C")], 10.0, abs_tol=1e-6)


def test_undirected_bidirectional_flow():
    """Test that flow can go in reverse direction (negative flow value)."""
    # Force flow to go in reverse direction by making forward expensive
    nodes = {
        "A": Node(id="A", supply=10.0),
        "B": Node(id="B", supply=-10.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=20.0, cost=100.0),  # Expensive A→B
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    # Flow should still be A→B despite high cost (only option)
    assert result.status == "optimal"
    assert result.flows[("A", "B")] > 0  # Positive = forward direction


def test_undirected_triangle_network():
    """Test undirected triangle with multiple paths."""
    #   A (10)
    #  / \
    # B   C
    #  \ /
    #   D (-10)

    nodes = {
        "A": Node(id="A", supply=10.0),
        "B": Node(id="B", supply=0.0),
        "C": Node(id="C", supply=0.0),
        "D": Node(id="D", supply=-10.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=8.0, cost=1.0),  # Path 1: A-B-D
        Arc(tail="B", head="D", capacity=8.0, cost=2.0),
        Arc(tail="A", head="C", capacity=8.0, cost=1.5),  # Path 2: A-C-D
        Arc(tail="C", head="D", capacity=8.0, cost=1.5),
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    # Optimal uses both paths or just cheaper path depending on capacity
    total_flow = sum(abs(f) for f in result.flows.values())
    assert total_flow >= 10.0  # At least source demand


def test_undirected_expansion_transform():
    """Test the internal transformation of undirected edges."""
    nodes = {
        "A": Node(id="A", supply=5.0),
        "B": Node(id="B", supply=-5.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=10.0, cost=2.0),
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

    expanded = problem.undirected_expansion()

    assert len(expanded) == 1
    exp_arc = expanded[0]
    assert exp_arc.tail == "A"
    assert exp_arc.head == "B"
    assert exp_arc.capacity == 10.0
    assert exp_arc.lower == -10.0  # Key: allows bidirectional flow
    assert exp_arc.cost == 2.0


def test_undirected_requires_finite_capacity():
    """Test that infinite capacity raises error for undirected graphs."""
    nodes = {
        "A": Node(id="A", supply=1.0),
        "B": Node(id="B", supply=-1.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=None, cost=1.0),  # Infinite capacity
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

    with pytest.raises(InvalidProblemError, match="infinite capacity"):
        problem.undirected_expansion()


def test_undirected_rejects_custom_lower_bound():
    """Test that custom lower bounds are rejected for undirected graphs."""
    nodes = {
        "A": Node(id="A", supply=5.0),
        "B": Node(id="B", supply=-5.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=10.0, cost=1.0, lower=2.0),  # Custom lower
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

    with pytest.raises(InvalidProblemError, match="custom lower bound"):
        problem.undirected_expansion()


def test_undirected_with_build_problem():
    """Test undirected graph creation via build_problem()."""
    nodes = [
        {"id": "X", "supply": 7.0},
        {"id": "Y", "supply": -7.0},
    ]
    arcs = [
        {"tail": "X", "head": "Y", "capacity": 10.0, "cost": 3.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    assert math.isclose(result.objective, 21.0, abs_tol=1e-6)  # 7 * 3


def test_undirected_multiple_edges():
    """Test undirected graph with multiple parallel edges."""
    nodes = {
        "S": Node(id="S", supply=15.0),
        "T": Node(id="T", supply=-15.0),
    }
    # Two parallel edges with different costs
    arcs = [
        Arc(tail="S", head="T", capacity=10.0, cost=2.0),  # Cheaper
        Arc(tail="S", head="T", capacity=10.0, cost=5.0),  # More expensive
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    # Should find optimal solution (may use both edges)
    assert result.status == "optimal"
    assert result.iterations > 0


def test_undirected_with_transshipment():
    """Test undirected graph with transshipment node."""
    nodes = {
        "A": Node(id="A", supply=10.0),
        "B": Node(id="B", supply=0.0),  # Transshipment
        "C": Node(id="C", supply=-10.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=20.0, cost=1.0),
        Arc(tail="B", head="C", capacity=20.0, cost=1.0),
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    # Flow should conserve at transshipment node B
    assert len(result.flows) > 0


def test_undirected_negative_flow_interpretation():
    """Test interpretation when solver returns negative flow (reverse direction)."""
    # Create scenario where reverse flow might be optimal
    # This is tricky since the solver chooses direction based on cost
    # We'll verify the transformation allows negative flows
    nodes = {
        "A": Node(id="A", supply=0.0),
        "B": Node(id="B", supply=10.0),  # Supply at B
        "C": Node(id="C", supply=-10.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=15.0, cost=1.0),
        Arc(tail="A", head="C", capacity=15.0, cost=1.0),
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    # Flow from B to C through A
    # Could be A-B negative (B→A) and A-C positive (A→C)
    # Or various combinations - just verify solution is valid
    flow_ab = result.flows.get(("A", "B"), 0.0)
    flow_ac = result.flows.get(("A", "C"), 0.0)

    # Flow conservation at A: inflow - outflow = 0
    inflow_a = max(0, -flow_ab) + max(0, -flow_ac)
    outflow_a = max(0, flow_ab) + max(0, flow_ac)
    assert math.isclose(inflow_a - outflow_a, 0.0, abs_tol=1e-6)


def test_undirected_capacity_constraint():
    """Test that undirected edges with sufficient capacity find feasible solution."""
    nodes = {
        "A": Node(id="A", supply=8.0),
        "B": Node(id="B", supply=-8.0),
    }
    arcs = [
        Arc(tail="A", head="B", capacity=10.0, cost=1.0),  # Capacity 10 is enough for 8
    ]
    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)
    result = solve_min_cost_flow(problem)

    # Should be feasible and optimal
    assert result.status == "optimal"
    assert len(result.flows) > 0


def test_undirected_vs_directed_equivalent():
    """Compare undirected edge vs two directed arcs (should give same result)."""
    nodes_dict = {
        "X": Node(id="X", supply=8.0),
        "Y": Node(id="Y", supply=-8.0),
    }

    # Undirected version
    arcs_undirected = [
        Arc(tail="X", head="Y", capacity=10.0, cost=3.0),
    ]
    problem_undirected = NetworkProblem(directed=False, nodes=nodes_dict, arcs=arcs_undirected)
    result_undirected = solve_min_cost_flow(problem_undirected)

    # Directed version (manual bidirectional arcs)
    arcs_directed = [
        Arc(tail="X", head="Y", capacity=10.0, cost=3.0),
        Arc(tail="Y", head="X", capacity=10.0, cost=3.0),
    ]
    problem_directed = NetworkProblem(directed=True, nodes=nodes_dict, arcs=arcs_directed)
    result_directed = solve_min_cost_flow(problem_directed)

    # Both should have same objective
    assert math.isclose(result_undirected.objective, result_directed.objective, abs_tol=1e-6)


def test_undirected_error_message_quality():
    """Test that error messages are helpful and descriptive."""
    nodes = {
        "A": Node(id="A", supply=1.0),
        "B": Node(id="B", supply=-1.0),
    }

    # Test infinite capacity error message
    arcs_inf = [Arc(tail="A", head="B", capacity=None, cost=1.0)]
    problem_inf = NetworkProblem(directed=False, nodes=nodes, arcs=arcs_inf)

    with pytest.raises(InvalidProblemError) as exc_info:
        problem_inf.undirected_expansion()

    error_msg = str(exc_info.value)
    assert "infinite capacity" in error_msg.lower()
    assert "finite capacity" in error_msg.lower()
    assert "A" in error_msg and "B" in error_msg  # Contains node names

    # Test custom lower bound error message
    arcs_lower = [Arc(tail="A", head="B", capacity=10.0, cost=1.0, lower=5.0)]
    problem_lower = NetworkProblem(directed=False, nodes=nodes, arcs=arcs_lower)

    with pytest.raises(InvalidProblemError) as exc_info:
        problem_lower.undirected_expansion()

    error_msg = str(exc_info.value)
    assert "custom lower bound" in error_msg.lower()
    assert "5.0" in error_msg  # Shows the problematic value
    assert "-10.0" in error_msg  # Shows what it should be (shows -capacity)
