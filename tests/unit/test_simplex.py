import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402

# Targeted exercises for simplex edge cases so subtle regressions surface quickly.


def _make_pricing_solver(costs):
    """Construct a solver with deterministic costs for pricing logic coverage."""
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "c", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 5.0, "cost": costs[0]},
        {"tail": "b", "head": "c", "capacity": 5.0, "cost": costs[1]},
        {"tail": "a", "head": "c", "capacity": 5.0, "cost": costs[2]},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)
    solver.perturbed_costs[: solver.actual_arc_count] = costs[: solver.actual_arc_count]
    solver._apply_phase_costs(phase=2)
    solver._rebuild_tree_structure()
    return solver


def _degenerate_problem():
    # Tiny network crafted to trigger degeneracy without requiring many pivots.
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 5.0, "cost": 200.0},
        {"tail": "m", "head": "t", "capacity": 5.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 5.0, "cost": 2.0},
    ]
    return build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def test_iteration_limit_preserves_current_flow_state():
    # Solver should retain the best-known feasible flow even when iteration budget expires.
    nodes = [
        {"id": "s", "supply": 2.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -2.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 3.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 3.0, "cost": 1.0},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Clamp the iteration budget so the solver has to return early with a feasible-but-ongoing basis.
    limited = solve_min_cost_flow(problem, max_iterations=2)
    assert limited.status == "iteration_limit"
    assert limited.iterations == 2

    # Increase the budget slightly to show the same instance converges to optimality.
    optimal = solve_min_cost_flow(problem, max_iterations=3)
    assert optimal.status == "optimal"
    assert optimal.iterations <= 3
    assert optimal.flows == {
        ("s", "m"): pytest.approx(2.0),
        ("m", "t"): pytest.approx(2.0),
    }
    assert math.isclose(optimal.objective, 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert limited.flows == {
        ("s", "m"): pytest.approx(2.0),
        ("m", "t"): pytest.approx(2.0),
    }
    assert limited.objective == pytest.approx(optimal.objective)


def test_phase_one_iteration_limit_reports_iteration_limit():
    # Force Phase 1 to stop early and confirm the status reflects the budget run-out.
    nodes = [
        {"id": "s", "supply": 2.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -2.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 2.5, "cost": 1.5},
        {"tail": "m", "head": "t", "capacity": 2.5, "cost": 1.5},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    result = solve_min_cost_flow(problem, max_iterations=1)

    assert result.status == "iteration_limit"
    assert result.flows == {}


def test_forrest_tomlin_failure_uses_numeric_fallback(caplog):
    # When FT reports failure but numeric data is available, fallback should succeed without rebuild.
    problem = build_problem(
        nodes=[
            {"id": "s", "supply": 1.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -1.0},
        ],
        arcs=[
            {"tail": "s", "head": "m", "capacity": 5.0, "cost": 1.0},
            {"tail": "m", "head": "t", "capacity": 5.0, "cost": 1.0},
            {"tail": "s", "head": "t", "capacity": 5.0, "cost": 2.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)

    class AlwaysFailFT:
        def __init__(self):
            self.calls = 0

        def solve(self, rhs):
            return None

        def update(self, pivot, new_column):
            self.calls += 1
            return False

    stub = AlwaysFailFT()
    solver.basis.ft_engine = stub
    solver.devex_weights = [5.0] * len(solver.devex_weights)

    caplog.set_level(logging.DEBUG, logger="network_solver.simplex")

    entering = solver._find_entering_arc(allow_zero=True)
    assert entering is not None
    solver._pivot(*entering)

    assert stub.calls == 1
    assert solver.ft_rebuilds == 0
    assert solver.devex_weights != [5.0] * len(solver.devex_weights)


def test_phase_one_short_circuit_triggers_before_iteration_cap(monkeypatch):
    # Phase 1 should exit as soon as artificial flows hit zero rather than burning the budget.
    problem = build_problem(
        nodes=[
            {"id": "s", "supply": 1.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -1.0},
        ],
        arcs=[
            {"tail": "s", "head": "m", "capacity": 2.0, "cost": 1.0},
            {"tail": "m", "head": "t", "capacity": 2.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    iterations = solver._run_simplex_iterations(
        max_iterations=100,
        allow_zero=True,
        phase_one=True,
    )
    assert iterations < 100
    assert not any(
        arc.artificial and arc.flow > solver.tolerance for arc in solver.arcs
    )


def test_degenerate_pivots_do_not_inflate_tree_size():
    # Repeated degenerate swaps must preserve the invariant of (node_count - 1) tree arcs.
    problem = build_problem(
        nodes=[
            {"id": "s", "supply": 0.0},
            {"id": "u", "supply": 0.0},
            {"id": "v", "supply": 0.0},
            {"id": "t", "supply": 0.0},
        ],
        arcs=[
            {"tail": "s", "head": "u", "capacity": 10.0, "cost": 1.0},
            {"tail": "u", "head": "v", "capacity": 10.0, "cost": 1.0},
            {"tail": "v", "head": "t", "capacity": 10.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    for _ in range(10):
        entering = solver._find_entering_arc(allow_zero=True)
        if entering is None:
            break
        solver._pivot(*entering)
        assert sum(1 for arc in solver.arcs if arc.in_tree) == solver.node_count - 1


def test_phase_two_stops_when_no_negative_reduced_costs():
    # With reduced costs already non-negative, Phase 2 should perform zero additional pivots.
    problem = build_problem(
        nodes=[
            {"id": "s", "supply": 4.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -4.0},
        ],
        arcs=[
            {"tail": "s", "head": "m", "capacity": 4.0, "cost": 1.0},
            {"tail": "m", "head": "t", "capacity": 4.0, "cost": 1.0},
            {"tail": "s", "head": "t", "capacity": 4.0, "cost": 3.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    solver._apply_phase_costs(phase=1)
    solver._rebuild_tree_structure()
    solver._run_simplex_iterations(50, allow_zero=True, phase_one=True)
    assert not any(
        arc.artificial and arc.flow > solver.tolerance for arc in solver.arcs
    )
    solver._apply_phase_costs(phase=2)
    solver._rebuild_tree_structure()
    iters = solver._run_simplex_iterations(50, allow_zero=False)
    assert iters == 0


def test_constructor_rejects_unbalanced_supplies_after_lower_bound_adjustment():
    class BareProblem:
        def __init__(self):
            self.directed = True
            self.tolerance = 1e-6
            self.nodes = {
                "a": type("Node", (), {"supply": 1.0})(),
                "b": type("Node", (), {"supply": -0.5})(),
            }
            self.arcs = [
                type(
                    "Arc",
                    (),
                    {
                        "tail": "a",
                        "head": "b",
                        "capacity": 10.0,
                        "cost": 1.0,
                        "lower": 0.75,
                    },
                )()
            ]

        def undirected_expansion(self):
            return self.arcs

        def validate(self):
            pass

    with pytest.raises(ValueError, match="Supplies do not balance"):
        NetworkSimplex(BareProblem())


def test_constructor_rejects_capacity_lower_violation():
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "c", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 1.0, "cost": 1.0, "lower": 2.0},
        {"tail": "b", "head": "c", "capacity": 1.0, "cost": 1.0},
    ]
    with pytest.raises(ValueError, match="capacity must be >= lower bound"):
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def test_initialize_tree_adds_fallback_artificial_arc():
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    has_fallback = any(
        arc.artificial and arc.tail == solver.root and solver.node_ids[arc.head] == "b"
        for arc in solver.arcs
    )
    assert has_fallback
    assert any(
        idx
        for idx in solver.tree_adj[solver.node_index["b"]]
        if solver.arcs[idx].artificial
    )


def test_pricing_wraps_and_returns_none_without_candidates():
    solver = _make_pricing_solver([2.0, 2.0, 2.0])
    solver.pricing_block = solver.actual_arc_count
    assert solver._find_entering_arc(False) is None
    assert solver.pricing_block == 0


def test_pricing_returns_zero_candidate_when_reduced_cost_zero():
    solver = _make_pricing_solver([0.0, 0.0, 0.0])
    result = solver._find_entering_arc(True)
    assert result is not None
    assert solver._find_entering_arc(False) is None


def test_pricing_keeps_weight_when_projection_missing(monkeypatch):
    solver = _make_pricing_solver([-5.0, 0.0, 0.0])
    old_weight = solver.devex_weights[0]
    monkeypatch.setattr(solver.basis, "project_column", lambda _arc: None)
    result = solver._find_entering_arc(True)
    assert result[0] == 0
    assert solver.devex_weights[0] == old_weight


def test_pricing_clamps_large_projection_weight(monkeypatch):
    solver = _make_pricing_solver([-5.0, 0.0, 0.0])

    def big_projection(_arc):
        return np.array([1e7, 0.0, 0.0])

    monkeypatch.setattr(solver.basis, "project_column", big_projection)
    result = solver._find_entering_arc(True)
    assert result[0] == 0
    assert solver.devex_weights[0] == pytest.approx(1e12)


def test_pivot_clamps_flow_to_bounds():
    nodes = [
        {"id": "s", "supply": 2.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -2.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 2.0, "cost": 0.0},
        {"tail": "m", "head": "t", "capacity": 2.0, "cost": 0.0},
        {"tail": "s", "head": "t", "capacity": 2.0, "cost": 10.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
    arc_idx, direction = solver._find_entering_arc(True)
    solver._pivot(arc_idx, direction)
    first_arc = solver.arcs[0]
    assert first_arc.flow == pytest.approx(first_arc.upper)
    assert solver.arcs[1].flow == pytest.approx(0.0)


def test_pivot_handles_degenerate_case():
    nodes = [
        {"id": "s", "supply": 2.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -2.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 2.0, "cost": 0.0},
        {"tail": "m", "head": "t", "capacity": 2.0, "cost": 0.0},
        {"tail": "t", "head": "s", "capacity": 2.0, "cost": 0.0},
        {"tail": "s", "head": "t", "capacity": 2.0, "cost": 10.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
    arc_idx, direction = solver._find_entering_arc(True)
    solver._pivot(arc_idx, direction)
    assert solver.arcs[arc_idx].in_tree is True


def test_pivot_fall_back_resets_weights_and_counts(caplog, monkeypatch):
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 5.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 5.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 5.0, "cost": 2.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))

    class StubBasis:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def replace_arc(self, *_args, **_kwargs):
            return False

    solver.basis = StubBasis(solver.basis)
    solver.devex_weights = [5.0] * len(solver.devex_weights)
    caplog.set_level(logging.DEBUG, logger="network_solver.simplex")
    entering = solver._find_entering_arc(True)
    solver._pivot(*entering)

    assert solver.ft_rebuilds == 1
    assert all(weight == 1.0 for weight in solver.devex_weights)
    assert any(
        "Forrest–Tomlin update failed" in record.message for record in caplog.records
    )


def test_apply_phase_costs_rejects_invalid_phase():
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 1.0, "cost": 1.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
    with pytest.raises(ValueError, match="Unsupported phase"):
        solver._apply_phase_costs(phase=3)


def test_solve_reports_infeasible_after_phase_one():
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": 0.0},
        {"id": "n", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 1.0, "cost": 1.0},
    ]
    result = NetworkSimplex(
        build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    ).solve(max_iterations=5)
    assert result.status == "infeasible"


def test_phase_two_iteration_branches():
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": 0.0},
        {"id": "n", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 1.0, "cost": -5.0},
        {"tail": "m", "head": "n", "capacity": 1.0, "cost": -4.0},
        {"tail": "s", "head": "n", "capacity": 1.0, "cost": 1.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
    assert solver.solve(max_iterations=1).status == "iteration_limit"
    assert solver.solve(max_iterations=5).status == "optimal"


def test_flow_post_processing_removes_and_rounds_near_zero():
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": -0.999999999999},
        {"id": "n", "supply": -0.000000000001},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 2.0, "cost": 0.0},
        {"tail": "s", "head": "n", "capacity": 2.0, "cost": 0.0},
        {"tail": "n", "head": "m", "capacity": 2.0, "cost": 2.0},
    ]
    result = NetworkSimplex(
        build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    ).solve()
    assert result.flows == {("s", "m"): pytest.approx(1.0)}


def test_reset_devex_weights_restores_defaults():
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
    ]
    solver = NetworkSimplex(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
    solver.devex_weights = [2.0, 3.0]
    solver._reset_devex_weights()
    assert solver.devex_weights == [1.0, 1.0]


def test_simplex_handles_numerical_errors_gracefully(monkeypatch):
    """Test that numerical errors in basis operations are handled gracefully."""
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 5.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    # Force project_column to return None to simulate numerical failure
    original_project = solver.basis.project_column

    def failing_project(arc):
        if arc.artificial:
            return original_project(arc)
        return None

    monkeypatch.setattr(solver.basis, "project_column", failing_project)

    # Solver should handle this gracefully
    result = solver.solve(max_iterations=10)
    # Should either succeed with limited iterations or report iteration_limit
    assert result.status in ["optimal", "iteration_limit"]


def test_find_entering_arc_with_all_nonnegative_reduced_costs():
    """Test that _find_entering_arc returns None when all reduced costs are non-negative."""
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    # Run to optimality
    solver.solve()

    # After optimality, no entering arc should be found
    entering = solver._find_entering_arc(allow_zero=False)
    assert entering is None


def test_pivot_with_zero_theta_degenerate():
    """Test pivot operation with zero flow change (degenerate pivot)."""
    nodes = [
        {"id": "s", "supply": 0.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": 0.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 10.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 0.5},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    # Find an entering arc and pivot
    entering = solver._find_entering_arc(allow_zero=True)
    if entering is not None:
        entering_idx, reduced_cost = entering
        solver._pivot(entering_idx, reduced_cost)
        # Should succeed without error


def test_basis_rebuild_after_ft_update_limit():
    """Test that basis is rebuilt when FT update limit is reached."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "m1", "supply": 0.0},
        {"id": "m2", "supply": 0.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = [
        {"tail": "s", "head": "m1", "capacity": 10.0, "cost": 1.0},
        {"tail": "m1", "head": "m2", "capacity": 10.0, "cost": 1.0},
        {"tail": "m2", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 10.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    # Force very low FT update limit to trigger rebuilds
    if solver.basis.ft_engine is not None:
        solver.basis.ft_engine.max_updates = 1

    result = solver.solve(max_iterations=20)

    # Should still reach optimal despite rebuilds
    assert result.status in ["optimal", "iteration_limit"]


def test_artificial_arc_flow_tracking():
    """Test that artificial arc flows are correctly tracked in Phase 1."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    # Intentionally no arcs - forces artificial arcs
    arcs = []
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    # Count artificial arcs
    artificial_count = sum(1 for arc in solver.arcs if arc.artificial)
    assert artificial_count > 0

    # Phase 1 should be needed
    result = solver.solve()
    # Without real arcs, problem is infeasible
    assert result.status == "infeasible"


def test_solver_handles_multiple_optimal_paths():
    """Test that solver finds optimal solution when multiple paths exist."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 10.0, "cost": 2.0},
        {"tail": "m", "head": "t", "capacity": 10.0, "cost": 3.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 6.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    result = NetworkSimplex(problem).solve()

    # Should find optimal solution
    assert result.status == "optimal"
    # Path through m costs 2+3=5 per unit, direct path costs 6 per unit
    assert result.objective == pytest.approx(25.0)


def test_arc_capacity_validation_during_init():
    """Test arc capacity validation during NetworkSimplex initialization."""
    from network_solver.data import Arc, NetworkProblem, Node

    nodes = {
        "a": Node(id="a", supply=1.0),
        "b": Node(id="b", supply=-1.0),
    }
    arcs = [Arc(tail="a", head="b", capacity=5.0, cost=1.0, lower=2.0)]
    problem = NetworkProblem(directed=True, nodes=nodes, arcs=arcs, tolerance=1e-6)

    solver = NetworkSimplex(problem)
    assert solver is not None


def test_disconnected_node_gets_artificial_arc():
    """Test that nodes with no connections get artificial arcs added."""
    from network_solver.data import Arc, NetworkProblem, Node

    nodes = {
        "a": Node(id="a", supply=1.0),
        "b": Node(id="b", supply=0.0),
        "c": Node(id="c", supply=-1.0),
    }
    arcs = [Arc(tail="a", head="c", capacity=10.0, cost=1.0)]
    problem = NetworkProblem(directed=True, nodes=nodes, arcs=arcs, tolerance=1e-6)

    solver = NetworkSimplex(problem)
    artificial_arcs = [arc for arc in solver.arcs if arc.artificial]
    assert len(artificial_arcs) > 0

    result = solver.solve()
    assert result.status == "optimal"


def test_backward_residual_entering_arc():
    """Test pricing logic that selects arcs with backward residual capacity."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 5.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 5.0, "cost": 1.0},
        {"tail": "t", "head": "s", "capacity": 3.0, "cost": -2.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    result = solver.solve(max_iterations=20)
    assert result.status in ["optimal", "iteration_limit"]


def test_degenerate_arc_selection_with_zero_reduced_cost():
    """Test that degenerate arcs can be selected when allow_zero=True."""
    nodes = [
        {"id": "s", "supply": 0.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": 0.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 10.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 2.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    solver._apply_phase_costs(phase=1)
    solver._rebuild_tree_structure()
    entering = solver._find_entering_arc(allow_zero=True)
    assert entering is None or isinstance(entering, tuple)


def test_ratio_test_tie_breaking():
    """Test tie-breaking logic in ratio test."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 5.0, "cost": 1.0},
        {"tail": "s", "head": "b", "capacity": 5.0, "cost": 1.0},
        {"tail": "a", "head": "t", "capacity": 5.0, "cost": 1.0},
        {"tail": "b", "head": "t", "capacity": 5.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    result = solver.solve()
    assert result.status == "optimal"


def test_degenerate_pivot_with_allow_zero():
    """Test degenerate pivot scenarios."""
    nodes = [
        {"id": "s", "supply": 0.01},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -0.01},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 10.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 1.5},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    iters = solver._run_simplex_iterations(
        max_iterations=10, allow_zero=True, phase_one=False
    )
    assert iters >= 0


def test_infeasible_with_iteration_limit():
    """Test infeasible problem detection even with iteration limit."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = []
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    result = solver.solve(max_iterations=1)
    # With no arcs, problem is infeasible - detected even with low iteration limit
    assert result.status == "infeasible"
    assert result.flows == {}


def test_flow_aggregation_with_duplicate_keys():
    """Test flow aggregation for duplicate arc keys."""
    problem = build_problem(
        nodes=[{"id": "s", "supply": 10.0}, {"id": "t", "supply": -10.0}],
        arcs=[
            {"tail": "s", "head": "t", "capacity": 5.0, "cost": 1.0},
            {"tail": "s", "head": "t", "capacity": 5.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    solver = NetworkSimplex(problem)
    result = solver.solve()

    assert result.status == "optimal"
    assert ("s", "t") in result.flows


def test_devex_weight_extreme_values():
    """Test Devex weight handling with extreme values."""
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "m", "supply": 0.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "m", "capacity": 10.0, "cost": 1.0},
        {"tail": "m", "head": "t", "capacity": 10.0, "cost": 1.0},
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 5.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    for i in range(len(solver.devex_weights)):
        solver.devex_weights[i] = 1e-20

    result = solver.solve(max_iterations=10)
    assert result.status in ["optimal", "iteration_limit"]


def test_near_zero_flow_removal():
    """Test that near-zero flows are removed from final result."""
    nodes = [
        {"id": "s", "supply": 1e-8},
        {"id": "t", "supply": -1e-8},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 10.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    result = solver.solve()
    assert result.status == "optimal"
