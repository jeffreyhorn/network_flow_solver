"""Tests for utility functions."""

import pytest

from network_solver import (
    build_problem,
    compute_bottleneck_arcs,
    extract_path,
    solve_min_cost_flow,
    validate_flow,
)
from network_solver.data import FlowResult


def test_extract_path_simple():
    """Test extracting a simple path from source to target."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    path = extract_path(result, problem, "s", "t")

    assert path is not None
    assert path.nodes == ["s", "t"]
    assert path.arcs == [("s", "t")]
    assert path.flow == pytest.approx(10.0, abs=1e-6)
    assert path.cost == pytest.approx(20.0, abs=1e-6)


def test_extract_path_multi_hop():
    """Test extracting a path through multiple nodes."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m1", "capacity": 100.0, "cost": 1.0},
        {"tail": "m1", "head": "m2", "capacity": 100.0, "cost": 1.0},
        {"tail": "m2", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    path = extract_path(result, problem, "s", "t")

    assert path is not None
    assert path.nodes == ["s", "m1", "m2", "t"]
    assert path.arcs == [("s", "m1"), ("m1", "m2"), ("m2", "t")]
    assert path.flow == pytest.approx(100.0, abs=1e-6)
    assert path.cost == pytest.approx(300.0, abs=1e-6)


def test_extract_path_no_path():
    """Test that extract_path returns None when no path exists."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
        {"id": "isolated", "supply": 0.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    # No path from s to isolated node
    path = extract_path(result, problem, "s", "isolated")
    assert path is None


def test_extract_path_same_node():
    """Test extracting path when source equals target."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    path = extract_path(result, problem, "s", "s")

    assert path is not None
    assert path.nodes == ["s"]
    assert path.arcs == []
    assert path.flow == 0.0
    assert path.cost == 0.0


def test_extract_path_invalid_node():
    """Test that extract_path raises error for invalid nodes."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    with pytest.raises(ValueError, match="Source node 'invalid' not found"):
        extract_path(result, problem, "invalid", "t")

    with pytest.raises(ValueError, match="Target node 'invalid' not found"):
        extract_path(result, problem, "s", "invalid")


@pytest.mark.xfail(
    reason="Hits Phase 1 early termination bug - see test_phase1_early_termination.py"
)
def test_extract_path_with_branching():
    """Test path extraction when multiple paths exist."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        # Two parallel paths
        {"tail": "s", "head": "m1", "capacity": 60.0, "cost": 1.0},
        {"tail": "s", "head": "m2", "capacity": 40.0, "cost": 2.0},
        {"tail": "m1", "head": "t", "capacity": 60.0, "cost": 1.0},
        {"tail": "m2", "head": "t", "capacity": 40.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    path = extract_path(result, problem, "s", "t")

    assert path is not None
    assert path.nodes[0] == "s"
    assert path.nodes[-1] == "t"
    assert len(path.nodes) == 3  # s -> intermediate -> t
    assert path.flow > 0


def test_validate_flow_valid_solution():
    """Test that validate_flow accepts a valid solution."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    validation = validate_flow(problem, result)

    assert validation.is_valid
    assert len(validation.errors) == 0
    assert len(validation.capacity_violations) == 0
    assert len(validation.lower_bound_violations) == 0


def test_validate_flow_capacity_violation():
    """Test that validate_flow detects capacity violations."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Create an invalid result with flow exceeding capacity
    result = FlowResult(
        objective=100.0,
        flows={("s", "t"): 30.0},  # Exceeds capacity of 20
        status="optimal",
        iterations=1,
    )

    validation = validate_flow(problem, result)

    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert ("s", "t") in validation.capacity_violations
    assert "exceeds capacity" in validation.errors[0]


def test_validate_flow_lower_bound_violation():
    """Test that validate_flow detects lower bound violations."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0, "lower": 5.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Create an invalid result with flow below lower bound
    result = FlowResult(
        objective=10.0,
        flows={("s", "t"): 2.0},  # Below lower bound of 5
        status="optimal",
        iterations=1,
    )

    validation = validate_flow(problem, result)

    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert ("s", "t") in validation.lower_bound_violations
    assert "below lower bound" in validation.errors[0]


def test_validate_flow_balance_violation():
    """Test that validate_flow detects flow conservation violations."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 20.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 20.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Create an invalid result with flow imbalance at node m
    result = FlowResult(
        objective=20.0,
        flows={
            ("s", "m"): 10.0,
            ("m", "t"): 5.0,  # Only 5 out, but 10 in - violates conservation
        },
        status="optimal",
        iterations=1,
    )

    validation = validate_flow(problem, result)

    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert "flow imbalance" in validation.errors[0]


def test_validate_flow_with_tolerance():
    """Test that validate_flow respects tolerance parameter."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 2.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Create a result with tiny capacity violation within tolerance
    result = FlowResult(
        objective=20.0,
        flows={("s", "t"): 10.0 + 1e-9},  # Slightly exceeds capacity
        status="optimal",
        iterations=1,
    )

    # Should be valid with large tolerance
    validation = validate_flow(problem, result, tolerance=1e-6)
    assert validation.is_valid

    # Should be invalid with tight tolerance
    validation = validate_flow(problem, result, tolerance=1e-12)
    assert not validation.is_valid


def test_compute_bottleneck_arcs_simple():
    """Test identifying bottleneck arcs at capacity."""
    nodes = [
        {"id": "s", "supply": 50.0},  # Fixed: match bottleneck capacity
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -50.0},  # Fixed: match bottleneck capacity
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 50.0, "cost": 1.0},  # Bottleneck
        {"tail": "m", "head": "t", "capacity": 200.0, "cost": 1.0},  # Not bottleneck
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.95)

    # The arc (s, m) should be at capacity (100% utilization)
    assert len(bottlenecks) >= 1
    bottleneck = bottlenecks[0]
    assert bottleneck.tail == "s"
    assert bottleneck.head == "m"
    assert bottleneck.utilization == pytest.approx(1.0, abs=1e-2)
    assert bottleneck.flow == pytest.approx(50.0, abs=1e-6)
    assert bottleneck.capacity == 50.0
    assert bottleneck.slack == pytest.approx(0.0, abs=1e-6)


def test_compute_bottleneck_arcs_threshold():
    """Test that bottleneck threshold filters correctly."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 100.0, "cost": 1.0},  # 100% utilization
        {"tail": "m", "head": "t", "capacity": 150.0, "cost": 1.0},  # ~67% utilization
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    # With threshold 0.95, only (s, m) should be included
    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.95)
    assert len(bottlenecks) == 1
    assert bottlenecks[0].tail == "s"

    # With threshold 0.5, both arcs should be included
    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.5)
    assert len(bottlenecks) == 2


def test_compute_bottleneck_arcs_no_bottlenecks():
    """Test compute_bottleneck_arcs when no arcs are near capacity."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 1000.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.95)

    assert len(bottlenecks) == 0


def test_compute_bottleneck_arcs_infinite_capacity():
    """Test that infinite capacity arcs are excluded from bottlenecks."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": None, "cost": 1.0},  # Infinite capacity
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.0)

    # Should not include infinite capacity arc
    assert len(bottlenecks) == 0


@pytest.mark.xfail(
    reason="Hits Phase 1 early termination bug - see test_phase1_early_termination.py"
)
def test_compute_bottleneck_arcs_sorting():
    """Test that bottlenecks are sorted by utilization."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "m1", "capacity": 60.0, "cost": 1.0},  # 100% utilization
        {"tail": "s", "head": "m2", "capacity": 50.0, "cost": 2.0},  # 80% utilization
        {"tail": "m1", "head": "t", "capacity": 100.0, "cost": 1.0},  # 60% utilization
        {"tail": "m2", "head": "t", "capacity": 100.0, "cost": 1.0},  # 40% utilization
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.5)

    # Should be sorted by utilization (descending)
    assert len(bottlenecks) >= 2
    for i in range(len(bottlenecks) - 1):
        assert bottlenecks[i].utilization >= bottlenecks[i + 1].utilization


def test_compute_bottleneck_arcs_with_cost():
    """Test that bottleneck arcs include cost information."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 50.0, "cost": 5.5},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.95)

    assert len(bottlenecks) == 1
    assert bottlenecks[0].cost == 5.5


def test_utilities_integration():
    """Test using all utilities together on a complex problem."""
    nodes = [
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 150.0},
        {"id": "warehouse_1", "supply": -80.0},
        {"id": "warehouse_2", "supply": -120.0},
        {"id": "warehouse_3", "supply": -50.0},
    ]
    arcs = [
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 100.0, "cost": 2.5},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "factory_a", "head": "warehouse_3", "capacity": 100.0, "cost": 1.5},
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 150.0, "cost": 1.8},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 150.0, "cost": 2.2},
        {"tail": "factory_b", "head": "warehouse_3", "capacity": 150.0, "cost": 2.8},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    # Validate the solution
    validation = validate_flow(problem, result)
    assert validation.is_valid

    # Extract a path
    path = extract_path(result, problem, "factory_a", "warehouse_1")
    if path is not None:
        assert path.nodes[0] == "factory_a"
        assert path.nodes[-1] == "warehouse_1"

    # Find bottlenecks
    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.8)
    # Should identify any arcs with >80% utilization
    for bottleneck in bottlenecks:
        assert bottleneck.utilization >= 0.8
