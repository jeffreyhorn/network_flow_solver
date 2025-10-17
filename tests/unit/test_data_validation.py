import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402

# These tests assert the light-weight validation in data.build_problem so callers
# receive actionable errors before the solver begins expensive work.


def test_build_problem_rejects_unbalanced_supply():
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "t", "supply": -4.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 1.0},
    ]

    with pytest.raises(ValueError, match="Problem is unbalanced"):
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def test_build_problem_checks_missing_tail():
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "missing", "head": "t", "capacity": 5.0, "cost": 2.0},
    ]

    with pytest.raises(KeyError) as excinfo:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert "Arc tail missing" in str(excinfo.value)


def test_build_problem_checks_missing_head():
    nodes = [
        {"id": "s", "supply": 1.5},
        {"id": "t", "supply": -1.5},
    ]
    arcs = [
        {"tail": "s", "head": "absent", "capacity": 4.0, "cost": 3.0},
    ]

    with pytest.raises(KeyError) as excinfo:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert "Arc head absent" in str(excinfo.value)


def test_build_problem_rejects_duplicate_nodes():
    nodes = [
        {"id": "dup", "supply": 0.0},
        {"id": "dup", "supply": 0.0},
    ]
    arcs = []

    with pytest.raises(ValueError, match="Duplicate node id dup"):
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def test_undirected_expansion_is_noop_for_directed_graph():
    nodes = [
        {"id": "a", "supply": 2.0},
        {"id": "b", "supply": -2.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 3.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    expanded = problem.undirected_expansion()

    assert expanded == tuple(problem.arcs)
    assert expanded[0] is problem.arcs[0]


def test_undirected_expansion_adds_negative_lower_bounds():
    nodes = [
        {"id": "a", "supply": 5.0},
        {"id": "b", "supply": -5.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 7.5, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
    expanded = problem.undirected_expansion()

    assert len(expanded) == 1
    expanded_arc = expanded[0]
    assert expanded_arc.tail == "a"
    assert expanded_arc.head == "b"
    assert expanded_arc.capacity == pytest.approx(7.5)
    assert expanded_arc.lower == pytest.approx(-7.5)


def test_undirected_expansion_requires_finite_capacity():
    nodes = [
        {"id": "a", "supply": 1.0},
        {"id": "b", "supply": -1.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": None, "cost": 0.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
    with pytest.raises(ValueError, match="Undirected arcs require finite capacity"):
        problem.undirected_expansion()


def test_undirected_expansion_rejects_custom_lower_bounds():
    nodes = [
        {"id": "a", "supply": 3.0},
        {"id": "b", "supply": -3.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 4.0, "cost": 1.0, "lower": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
    with pytest.raises(ValueError, match="do not support custom lower bounds"):
        problem.undirected_expansion()


def test_arc_rejects_self_loops():
    """Test that Arc __post_init__ rejects self-loops."""
    from network_solver.data import Arc

    with pytest.raises(ValueError, match="Self-loops are not supported"):
        Arc(tail="a", head="a", capacity=5.0, cost=1.0)


def test_arc_rejects_capacity_less_than_lower():
    """Test that Arc __post_init__ rejects capacity < lower bound."""
    from network_solver.data import Arc

    with pytest.raises(ValueError, match="Arc capacity must be >= lower bound"):
        Arc(tail="a", head="b", capacity=5.0, cost=1.0, lower=10.0)


def test_build_problem_handles_missing_supply():
    """Test that build_problem defaults supply to 0.0 when not provided."""
    nodes = [
        {"id": "a"},  # No supply field
        {"id": "b", "supply": 0.0},
    ]
    arcs = []

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert problem.nodes["a"].supply == 0.0


def test_build_problem_handles_missing_arc_fields():
    """Test that build_problem uses defaults for missing arc fields."""
    nodes = [
        {"id": "a", "supply": 1.0},
        {"id": "b", "supply": -1.0},
    ]
    arcs = [
        {"tail": "a", "head": "b"},  # Missing cost, lower, capacity
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    arc = problem.arcs[0]
    assert arc.cost == 0.0
    assert arc.lower == 0.0
    assert arc.capacity is None


def test_build_problem_converts_numeric_node_ids():
    """Test that build_problem converts numeric node IDs to strings."""
    nodes = [
        {"id": 1, "supply": 1.0},
        {"id": 2, "supply": -1.0},
    ]
    arcs = [
        {"tail": 1, "head": 2, "capacity": 5.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert "1" in problem.nodes
    assert "2" in problem.nodes
    assert problem.arcs[0].tail == "1"
    assert problem.arcs[0].head == "2"


def test_build_problem_validates_tolerance():
    """Test that build_problem accepts tolerance parameter."""
    nodes = [
        {"id": "a", "supply": 0.1},
        {"id": "b", "supply": -0.1},
    ]
    arcs = []

    # Should pass with large tolerance
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=0.2)
    assert problem.tolerance == 0.2

    # Should fail with strict tolerance (make imbalance larger)
    nodes_unbalanced = [
        {"id": "a", "supply": 1.0},
        {"id": "b", "supply": -0.5},
    ]
    with pytest.raises(ValueError, match="Problem is unbalanced"):
        build_problem(nodes=nodes_unbalanced, arcs=[], directed=True, tolerance=1e-6)


def test_network_problem_validate_called_in_build():
    """Test that validate() is called during build_problem."""
    nodes = [
        {"id": "a", "supply": 5.0},
        {"id": "b", "supply": -4.99},  # Unbalanced
    ]
    arcs = []

    with pytest.raises(ValueError, match="Problem is unbalanced"):
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-3)


def test_empty_graph():
    """Test that build_problem handles graphs with no arcs."""
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = []

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert len(problem.arcs) == 0
    assert len(problem.nodes) == 2


def test_single_node_graph():
    """Test that build_problem handles single-node graphs."""
    nodes = [
        {"id": "a", "supply": 0.0},
    ]
    arcs = []

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    assert len(problem.nodes) == 1
    assert len(problem.arcs) == 0


def test_negative_capacity():
    """Test that build_problem accepts negative capacity (None is also valid)."""
    from network_solver.data import Arc

    # Negative capacity with lower=0 should fail
    with pytest.raises(ValueError, match="Arc capacity must be >= lower bound"):
        Arc(tail="a", head="b", capacity=-5.0, cost=1.0, lower=0.0)


def test_undirected_expansion_with_zero_capacity():
    """Test undirected expansion with zero capacity edge."""
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 0.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
    expanded = problem.undirected_expansion()

    assert len(expanded) == 1
    assert expanded[0].capacity == 0.0
    assert expanded[0].lower == 0.0
