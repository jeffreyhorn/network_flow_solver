"""Tests for Phase 1 early termination bug.

This test documents a known issue where Phase 1 can terminate prematurely
with zero artificial flow but violated flow conservation. The fix applied
in this PR detects this condition and correctly reports infeasibility,
but the root cause (why Phase 1 terminates early) requires further investigation.
"""

import pytest

from network_solver import build_problem, solve_min_cost_flow


def test_phase1_early_termination_parallel_paths():
    """Test that Phase 1 finds feasible solution for parallel path problem.

    This problem has two paths from a to d:
    - Long path: a->b->c->d (capacity 15 each, cost 3 total)
    - Short path: a->d (capacity 10, cost 4)

    Optimal solution uses 15 units through long path (cost = 45).

    Historical note: This test originally documented a pivot bug where theta
    computation incorrectly skipped the entering arc's capacity constraint,
    causing conservation violations. The bug is now fixed.
    """
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 15.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": 0.0},
            {"id": "d", "supply": -15.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 15.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 15.0, "cost": 1.0},
            {"tail": "c", "head": "d", "capacity": 15.0, "cost": 1.0},
            {"tail": "a", "head": "d", "capacity": 10.0, "cost": 4.0},  # Parallel path
        ],
        directed=True,
        tolerance=1e-6,
    )

    result = solve_min_cost_flow(problem)

    # Solver should find optimal solution
    assert result.status == "optimal"
    assert result.objective == pytest.approx(45.0)

    # Verify flows use the long path (cheaper)
    assert result.flows[("a", "b")] == pytest.approx(15.0)
    assert result.flows[("b", "c")] == pytest.approx(15.0)
    assert result.flows[("c", "d")] == pytest.approx(15.0)
    # Short path unused
    assert ("a", "d") not in result.flows or result.flows[("a", "d")] == pytest.approx(0.0)


def test_phase1_should_find_feasible_solution():
    """Test that documents the expected behavior once Phase 1 is fixed.

    This is marked as xfail because it represents the desired behavior
    after fixing the Phase 1 termination bug. Currently fails because
    Phase 1 terminates early and the conservation check reports infeasibility.
    """
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 15.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": 0.0},
            {"id": "d", "supply": -15.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 15.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 15.0, "cost": 1.0},
            {"tail": "c", "head": "d", "capacity": 15.0, "cost": 1.0},
            {"tail": "a", "head": "d", "capacity": 10.0, "cost": 4.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    result = solve_min_cost_flow(problem)

    # Expected behavior after Phase 1 fix
    assert result.status == "optimal"
    assert result.objective == pytest.approx(45.0)
    assert result.flows[("a", "b")] == pytest.approx(15.0)
    assert result.flows[("b", "c")] == pytest.approx(15.0)
    assert result.flows[("c", "d")] == pytest.approx(15.0)
    assert ("a", "d") not in result.flows or result.flows[("a", "d")] == pytest.approx(0.0)

    # Verify flow conservation
    for node_id, supply in [("a", 15.0), ("b", 0.0), ("c", 0.0), ("d", -15.0)]:
        inflow = sum(flow for (tail, head), flow in result.flows.items() if head == node_id)
        outflow = sum(flow for (tail, head), flow in result.flows.items() if tail == node_id)
        balance = supply + inflow - outflow
        assert abs(balance) < 1e-6, f"Node {node_id} conservation violated: balance={balance}"
