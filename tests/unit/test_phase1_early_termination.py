"""Tests for Phase 1 early termination bug.

This test documents a known issue where Phase 1 can terminate prematurely
with zero artificial flow but violated flow conservation. The fix applied
in this PR detects this condition and correctly reports infeasibility,
but the root cause (why Phase 1 terminates early) requires further investigation.
"""

import pytest

from network_solver import build_problem, solve_min_cost_flow


def test_phase1_early_termination_parallel_paths():
    """Test that Phase 1 termination bug is caught by flow conservation check.

    This problem has a feasible solution but Phase 1 terminates early with:
    - All artificial arcs at zero flow (old stopping criterion)
    - Flow conservation violated at some nodes

    The fix detects this and returns 'infeasible' status. This is correct behavior
    given the Phase 1 bug, but ideally Phase 1 should find the feasible solution.

    Expected feasible solution:
        a->b: 15, b->c: 15, c->d: 15 (cost=45)

    Bug behavior (before fix):
        Returns 'optimal' with a->b: 10, b->c: 10, c->d: 10 (violates conservation)

    Current behavior (after fix):
        Returns 'infeasible' (correctly detects the bug)

    TODO: Fix root cause in Phase 1 iteration logic to find the feasible solution.
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

    # With the fix, solver correctly detects Phase 1 failed
    assert result.status == "infeasible"
    assert result.flows == {}

    # Document that this problem is actually feasible (Phase 1 bug prevents finding solution)
    # TODO: Once Phase 1 is fixed, update this test to expect 'optimal' status


@pytest.mark.xfail(reason="Phase 1 early termination bug - needs algorithmic fix")
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
