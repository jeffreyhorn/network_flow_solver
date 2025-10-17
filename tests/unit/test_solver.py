"""Unit tests for solver module."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import FlowResult, NetworkProblem, build_problem  # noqa: E402
from network_solver.exceptions import InvalidProblemError  # noqa: E402
from network_solver.solver import (  # noqa: E402
    load_problem,
    save_result,
    solve_min_cost_flow,
)


class TestSolveMinCostFlow:
    """Tests for solve_min_cost_flow() function."""

    def test_solve_min_cost_flow_simple_problem(self):
        """Test solve_min_cost_flow on a simple feasible problem."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem)

        assert result.status == "optimal"
        assert result.objective == pytest.approx(10.0)
        assert len(result.flows) > 0

    def test_solve_min_cost_flow_with_max_iterations(self):
        """Test solve_min_cost_flow with max_iterations parameter."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 5.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -5.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 10.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 10.0, "cost": 1.0},
                {"tail": "s", "head": "t", "capacity": 10.0, "cost": 3.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem, max_iterations=100)

        assert result.status == "optimal"
        assert result.iterations <= 100

    def test_solve_min_cost_flow_respects_iteration_limit(self):
        """Test that solve_min_cost_flow respects very low iteration limits."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 10.0},
                {"id": "m", "supply": 0.0},
                {"id": "t", "supply": -10.0},
            ],
            arcs=[
                {"tail": "s", "head": "m", "capacity": 15.0, "cost": 1.0},
                {"tail": "m", "head": "t", "capacity": 15.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem, max_iterations=1)

        # Should hit iteration limit
        assert result.status == "iteration_limit"
        assert result.iterations == 1

    def test_solve_min_cost_flow_with_none_iterations(self):
        """Test that solve_min_cost_flow accepts None for unlimited iterations."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 1.0},
                {"id": "t", "supply": -1.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 5.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        result = solve_min_cost_flow(problem, max_iterations=None)

        assert result.status == "optimal"

    def test_solve_min_cost_flow_propagates_validation_errors(self):
        """Test that solve_min_cost_flow propagates validation errors."""
        from network_solver.data import Arc, NetworkProblem, Node

        # Create problem that will fail validation
        nodes = {
            "s": Node(id="s", supply=10.0),
            "t": Node(id="t", supply=-5.0),  # Unbalanced
        }
        arcs = [Arc(tail="s", head="t", capacity=20.0, cost=1.0)]

        problem = NetworkProblem(
            directed=True,
            nodes=nodes,
            arcs=arcs,
            tolerance=1e-6,
        )

        # Validation should catch this unbalanced supply
        with pytest.raises(InvalidProblemError, match="Problem is unbalanced"):
            problem.validate()

    def test_solve_min_cost_flow_fresh_solver_instance(self):
        """Test that solve_min_cost_flow creates fresh solver instances."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 1.0},
                {"id": "t", "supply": -1.0},
            ],
            arcs=[
                {"tail": "s", "head": "t", "capacity": 5.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        # Solve twice and ensure results are independent
        result1 = solve_min_cost_flow(problem)
        result2 = solve_min_cost_flow(problem)

        assert result1.objective == result2.objective
        assert result1.status == result2.status


class TestLoadProblemWrapper:
    """Tests for load_problem() wrapper function."""

    def test_load_problem_calls_io_function(self, tmp_path: Path):
        """Test that load_problem wrapper delegates to io.load_problem."""
        import json

        problem_data = {
            "directed": True,
            "tolerance": 1e-6,
            "nodes": [
                {"id": "a", "supply": 1.0},
                {"id": "b", "supply": -1.0},
            ],
            "edges": [
                {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            ],
        }
        path = tmp_path / "problem.json"
        path.write_text(json.dumps(problem_data), encoding="utf-8")

        problem = load_problem(path)

        assert isinstance(problem, NetworkProblem)
        assert problem.directed is True
        assert len(problem.nodes) == 2
        assert len(problem.arcs) == 1

    def test_load_problem_accepts_string_path(self, tmp_path: Path):
        """Test that load_problem accepts string paths."""
        import json

        problem_data = {
            "nodes": [{"id": "a", "supply": 1.0}, {"id": "b", "supply": -1.0}],
            "edges": [{"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0}],
        }
        path = tmp_path / "problem.json"
        path.write_text(json.dumps(problem_data), encoding="utf-8")

        problem = load_problem(str(path))

        assert isinstance(problem, NetworkProblem)

    def test_load_problem_propagates_errors(self, tmp_path: Path):
        """Test that load_problem propagates IO errors."""
        nonexistent = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_problem(nonexistent)


class TestSaveResultWrapper:
    """Tests for save_result() wrapper function."""

    def test_save_result_calls_io_function(self, tmp_path: Path):
        """Test that save_result wrapper delegates to io.save_result."""
        import json

        result = FlowResult(
            objective=10.0,
            flows={("a", "b"): 5.0},
            status="optimal",
            iterations=3,
        )
        path = tmp_path / "result.json"

        save_result(path, result)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "optimal"
        assert data["objective"] == 10.0

    def test_save_result_accepts_string_path(self, tmp_path: Path):
        """Test that save_result accepts string paths."""
        result = FlowResult(objective=0.0, flows={}, status="optimal", iterations=0)
        path = tmp_path / "result.json"

        save_result(str(path), result)

        assert path.exists()

    def test_save_result_propagates_errors(self, monkeypatch, tmp_path: Path):
        """Test that save_result propagates IO errors."""
        result = FlowResult(objective=0.0, flows={}, status="optimal", iterations=0)
        path = tmp_path / "result.json"

        def fake_open(self, *_args, **_kwargs):
            raise PermissionError("cannot write")

        monkeypatch.setattr(Path, "open", fake_open)

        with pytest.raises(PermissionError):
            save_result(path, result)


class TestIntegration:
    """Integration tests for the solver module."""

    def test_round_trip_problem_solve_save(self, tmp_path: Path):
        """Test loading a problem, solving it, and saving the result."""
        import json

        # Create problem file
        problem_data = {
            "directed": True,
            "tolerance": 1e-6,
            "nodes": [
                {"id": "s", "supply": 10.0},
                {"id": "t", "supply": -10.0},
            ],
            "edges": [
                {"tail": "s", "head": "t", "capacity": 20.0, "cost": 2.0},
            ],
        }
        problem_path = tmp_path / "problem.json"
        problem_path.write_text(json.dumps(problem_data), encoding="utf-8")

        # Load and solve
        problem = load_problem(problem_path)
        result = solve_min_cost_flow(problem)

        # Save result
        result_path = tmp_path / "result.json"
        save_result(result_path, result)

        # Verify result was saved
        assert result_path.exists()
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
        assert result_data["status"] == "optimal"
        assert result_data["objective"] == pytest.approx(20.0)
