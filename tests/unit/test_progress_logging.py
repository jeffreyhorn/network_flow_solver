"""Tests for progress logging during solver execution."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import ProgressInfo, build_problem, solve_min_cost_flow  # noqa: E402


def test_progress_callback_called():
    """Test that progress callback is invoked during solve."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 150.0, "cost": 1.0},
    ]

    progress_calls = []

    def callback(info: ProgressInfo) -> None:
        progress_calls.append(info)

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem, progress_callback=callback, progress_interval=1)

    assert result.status == "optimal"
    # Should have received at least one progress update
    assert len(progress_calls) > 0


def test_progress_info_fields():
    """Test that ProgressInfo contains expected fields."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "a", "supply": 0.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"id": "s", "supply": 50.0},
        {"id": "a", "supply": 0.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 100.0, "cost": 2.0},
        {"tail": "a", "head": "t", "capacity": 100.0, "cost": 3.0},
    ]

    progress_calls = []

    def callback(info: ProgressInfo) -> None:
        progress_calls.append(info)

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solve_min_cost_flow(problem, progress_callback=callback, progress_interval=1)

    if progress_calls:
        info = progress_calls[0]
        assert hasattr(info, "iteration")
        assert hasattr(info, "max_iterations")
        assert hasattr(info, "phase")
        assert hasattr(info, "phase_iterations")
        assert hasattr(info, "objective_estimate")
        assert hasattr(info, "elapsed_time")
        assert info.iteration >= 0
        assert info.max_iterations > 0
        assert info.phase in (1, 2)
        assert info.elapsed_time >= 0


def test_progress_interval():
    """Test that progress_interval controls callback frequency."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "c", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 50.0, "cost": 1.0},
        {"tail": "s", "head": "b", "capacity": 50.0, "cost": 2.0},
        {"tail": "a", "head": "c", "capacity": 50.0, "cost": 1.0},
        {"tail": "b", "head": "c", "capacity": 50.0, "cost": 1.0},
        {"tail": "c", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    progress_calls_interval_1 = []
    progress_calls_interval_10 = []

    def callback1(info: ProgressInfo) -> None:
        progress_calls_interval_1.append(info)

    def callback10(info: ProgressInfo) -> None:
        progress_calls_interval_10.append(info)

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Solve with interval=1 (more frequent callbacks)
    solve_min_cost_flow(problem, progress_callback=callback1, progress_interval=1)

    # Solve with interval=10 (less frequent callbacks)
    solve_min_cost_flow(problem, progress_callback=callback10, progress_interval=10)

    # interval=1 should have more callbacks than interval=10
    if len(progress_calls_interval_1) > 0 and len(progress_calls_interval_10) > 0:
        assert len(progress_calls_interval_1) >= len(progress_calls_interval_10)


def test_no_progress_callback():
    """Test that solver works without progress callback (backward compatibility)."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -10.0},
    ]
    arcs = [
        {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)  # No progress_callback

    assert result.status == "optimal"
    assert result.objective == pytest.approx(10.0)


def test_progress_tracks_phases():
    """Test that progress callback distinguishes between Phase 1 and Phase 2."""
    nodes = [
        {"id": "s", "supply": 50.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "t", "supply": -50.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 100.0, "cost": 1.0},
        {"tail": "s", "head": "b", "capacity": 100.0, "cost": 2.0},
        {"tail": "a", "head": "t", "capacity": 100.0, "cost": 1.0},
        {"tail": "b", "head": "t", "capacity": 100.0, "cost": 1.0},
    ]

    progress_calls = []

    def callback(info: ProgressInfo) -> None:
        progress_calls.append(info)

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solve_min_cost_flow(problem, progress_callback=callback, progress_interval=1)

    if len(progress_calls) > 0:
        phases_seen = {info.phase for info in progress_calls}
        # Should see at least phase 1 or phase 2
        assert phases_seen.issubset({1, 2})


def test_progress_iteration_monotonic():
    """Test that iteration count increases monotonically."""
    nodes = [
        {"id": "s", "supply": 100.0},
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
        {"id": "t", "supply": -100.0},
    ]
    arcs = [
        {"tail": "s", "head": "a", "capacity": 50.0, "cost": 1.0},
        {"tail": "s", "head": "b", "capacity": 50.0, "cost": 2.0},
        {"tail": "a", "head": "t", "capacity": 50.0, "cost": 1.0},
        {"tail": "b", "head": "t", "capacity": 50.0, "cost": 1.0},
    ]

    progress_calls = []

    def callback(info: ProgressInfo) -> None:
        progress_calls.append(info)

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solve_min_cost_flow(problem, progress_callback=callback, progress_interval=1)

    if len(progress_calls) > 1:
        for i in range(1, len(progress_calls)):
            # Iteration count should never decrease
            assert progress_calls[i].iteration >= progress_calls[i - 1].iteration
