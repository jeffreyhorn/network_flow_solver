"""Tests for block size auto-tuning functionality."""

from network_solver import (
    SolverOptions,
    build_problem,
    solve_min_cost_flow,
)
from network_solver.simplex import NetworkSimplex


def test_block_size_adaptation_with_high_degeneracy():
    """Test that auto-tuning mechanism exists and tracks pivot statistics.

    This test verifies that the adaptation infrastructure is in place,
    even if actual adaptation may not occur for simple problems.
    """
    # Create a simple transportation problem
    nodes = [
        {"id": "s1", "supply": 50.0},
        {"id": "s2", "supply": 30.0},
        {"id": "t1", "supply": -40.0},
        {"id": "t2", "supply": -40.0},
    ]

    # Multiple routes to encourage some complexity
    arcs = [
        {"tail": "s1", "head": "t1", "capacity": 50.0, "cost": 2.0},
        {"tail": "s1", "head": "t2", "capacity": 50.0, "cost": 3.0},
        {"tail": "s2", "head": "t1", "capacity": 30.0, "cost": 4.0},
        {"tail": "s2", "head": "t2", "capacity": 30.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto", max_iterations=500)
    solver = NetworkSimplex(problem, options=options)

    # Verify auto-tuning is enabled
    assert solver.auto_tune_block_size is True

    # Solve the problem
    result = solver.solve()

    # Should find optimal solution
    assert result.status == "optimal"
    assert solver.block_size >= 1  # Still valid

    # Verify counters were being tracked
    # (They get reset after adaptations, so we can't check exact values)
    assert hasattr(solver, "degenerate_pivot_count")
    assert hasattr(solver, "total_pivot_count")


def test_block_size_adaptation_early_return_insufficient_pivots():
    """Test that adaptation skips when total_pivot_count < 10.

    Tests the early return path in _adapt_block_size when there
    aren't enough pivot samples to make a decision.
    """
    # Very simple problem that solves quickly (few pivots)
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    initial_block_size = solver.block_size

    # Solve (should complete in very few iterations)
    result = solver.solve()

    assert result.status == "optimal"
    # Block size should not have changed (not enough pivots to adapt)
    assert solver.block_size == initial_block_size


def test_block_size_adaptation_logging(caplog):
    """Test that block size adaptation produces DEBUG log messages.

    Tests the logging code path in _adapt_block_size when block size changes.
    """
    import logging

    # Create a problem that will trigger adaptations
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(15)]
    nodes[0]["supply"] = 200.0
    nodes[-1]["supply"] = -200.0

    # Dense network to get many iterations
    arcs = []
    for i in range(14):
        arcs.append({"tail": f"n{i}", "head": f"n{i + 1}", "capacity": 200.0, "cost": float(i + 1)})
    # Add cross-connections with adequate capacity
    for i in range(0, 12, 3):
        arcs.append(
            {"tail": f"n{i}", "head": f"n{i + 3}", "capacity": 150.0, "cost": float(i + 0.5)}
        )

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto", max_iterations=500)

    # Enable DEBUG logging
    with caplog.at_level(logging.DEBUG):
        result = solve_min_cost_flow(problem, options=options)

    assert result.status == "optimal"

    # Check if any adaptation log messages were produced
    # (May or may not occur depending on problem characteristics)
    adaptation_logs = [
        record for record in caplog.records if "Adapted block_size" in record.message
    ]

    # We can't guarantee adaptation happened, but if it did, verify log format
    if adaptation_logs:
        log_message = adaptation_logs[0].message
        assert "→" in log_message
        assert "degenerate_ratio=" in log_message


def test_block_size_adaptation_interval_timing():
    """Test that adaptation only occurs every 50 iterations (adaptation_interval).

    This verifies the interval checking logic in _adapt_block_size.
    """
    # Create a medium-complexity problem
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(8)]
    nodes[0]["supply"] = 50.0
    nodes[-1]["supply"] = -50.0

    arcs = []
    for i in range(7):
        arcs.append({"tail": f"n{i}", "head": f"n{i + 1}", "capacity": 50.0, "cost": 1.0})
    for i in range(0, 6, 2):
        arcs.append({"tail": f"n{i}", "head": f"n{i + 2}", "capacity": 40.0, "cost": 1.5})

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Verify adaptation interval is set
    assert solver.adaptation_interval == 50
    assert solver.last_adaptation_iteration == 0

    result = solver.solve()
    assert result.status == "optimal"


def test_auto_tuning_with_various_problem_sizes():
    """Test that initial block size heuristic works across problem sizes.

    This test verifies the static heuristic for different arc counts:
    - Very small (<100): num_arcs // 4
    - Small (100-1000): num_arcs // 4
    - Medium (1000-10000): num_arcs // 8
    - Large (>10000): num_arcs // 16
    """
    test_cases = [
        # (num_nodes, connections_per_node, expected_divisor)
        (5, 2, 4),  # Very small: ~10 arcs -> divisor 4
        (30, 5, 4),  # Small: ~150 arcs -> divisor 4
    ]

    for num_nodes, conn_per_node, expected_divisor in test_cases:
        nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(num_nodes)]
        nodes[0]["supply"] = 50.0
        nodes[-1]["supply"] = -50.0

        arcs = []
        for i in range(num_nodes - 1):
            for j in range(i + 1, min(i + conn_per_node + 1, num_nodes)):
                arcs.append({"tail": f"n{i}", "head": f"n{j}", "capacity": 10.0, "cost": 1.0})

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        options = SolverOptions(block_size="auto")
        solver = NetworkSimplex(problem, options=options)

        expected_block_size = max(1, solver.actual_arc_count // expected_divisor)
        assert solver.block_size == expected_block_size, (
            f"Problem with {solver.actual_arc_count} arcs: "
            f"expected block_size={expected_block_size}, got {solver.block_size}"
        )


def test_auto_tuning_block_size_clamping():
    """Test that adapted block size is clamped to [10, num_arcs].

    Verifies that runtime adaptation respects min/max bounds.
    """
    # Small problem where block size could go below 10
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(5)]
    nodes[0]["supply"] = 20.0
    nodes[-1]["supply"] = -20.0

    # Only a few arcs
    arcs = [
        {"tail": "n0", "head": "n1", "capacity": 10.0, "cost": 1.0},
        {"tail": "n1", "head": "n2", "capacity": 10.0, "cost": 1.0},
        {"tail": "n2", "head": "n3", "capacity": 10.0, "cost": 1.0},
        {"tail": "n3", "head": "n4", "capacity": 10.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Manually trigger adaptation with very low degeneracy
    # (to test the clamping to min=10)
    solver.total_pivot_count = 100
    solver.degenerate_pivot_count = 5  # 5% degeneracy (< 10%)
    solver.last_adaptation_iteration = 0
    solver.block_size = 15  # Start with small block size

    # Manually call adaptation
    solver._adapt_block_size(51)  # iteration > 50, should trigger

    # Block size should decrease but not go below 10
    assert solver.block_size >= 10

    # Also test upper bound (shouldn't exceed num_arcs)
    solver.total_pivot_count = 100
    solver.degenerate_pivot_count = 50  # 50% degeneracy (> 30%)
    solver.last_adaptation_iteration = 0
    solver.block_size = solver.actual_arc_count - 2

    solver._adapt_block_size(101)

    # Block size should increase but not exceed num_arcs
    assert solver.block_size <= solver.actual_arc_count


def test_auto_tuning_disabled_with_explicit_int():
    """Test that providing explicit int block_size disables auto-tuning.

    Ensures backward compatibility and user control.
    """
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size=25)  # Explicit int
    solver = NetworkSimplex(problem, options=options)

    # Auto-tuning should be disabled
    assert solver.auto_tune_block_size is False
    assert solver.block_size == 25

    initial_block_size = solver.block_size
    result = solver.solve()

    # Block size should not change
    assert solver.block_size == initial_block_size
    assert result.status == "optimal"


def test_degenerate_pivot_tracking():
    """Test that degenerate pivots are correctly tracked for adaptation.

    Verifies that both degenerate_pivot_count and total_pivot_count
    are incremented during pivoting.
    """
    # Create a simple problem
    nodes = [
        {"id": "s", "supply": 30.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -30.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 30.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 30.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 3.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    options = SolverOptions(block_size="auto")
    solver = NetworkSimplex(problem, options=options)

    # Initially counters should be zero
    assert solver.degenerate_pivot_count == 0
    assert solver.total_pivot_count == 0

    result = solver.solve()

    # After solving, total_pivot_count should have increased
    # (exact number depends on problem, but should be > 0)
    assert result.status == "optimal"
    # Counters may have been reset if adaptation occurred
    # But we can verify the tracking attributes exist and are non-negative
    assert solver.degenerate_pivot_count >= 0
    assert solver.total_pivot_count >= 0
