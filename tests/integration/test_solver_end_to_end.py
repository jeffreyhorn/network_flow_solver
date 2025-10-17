import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.solver import load_problem, save_result, solve_min_cost_flow  # noqa: E402


def test_solver_end_to_end(tmp_path: Path):
    # Exercise the public solver facade by round-tripping a tiny JSON instance.
    problem_payload = {
        "directed": True,
        "tolerance": 1e-6,
        "nodes": [
            {"id": "s", "supply": 4.0},
            {"id": "m", "supply": 0.0},
            {"id": "t", "supply": -4.0},
        ],
        "edges": [
            {"tail": "s", "head": "m", "capacity": 4.0, "cost": 1.0},
            {"tail": "m", "head": "t", "capacity": 4.0, "cost": 1.0},
            {"tail": "s", "head": "t", "capacity": 4.0, "cost": 3.0},
        ],
    }

    problem_path = tmp_path / "problem.json"
    problem_path.write_text(json.dumps(problem_payload), encoding="utf-8")

    problem = load_problem(problem_path)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    assert result.flows == {
        ("s", "m"): pytest.approx(4.0),
        ("m", "t"): pytest.approx(4.0),
    }
    assert result.objective == pytest.approx(8.0)

    result_path = tmp_path / "result.json"
    save_result(result_path, result)

    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["status"] == "optimal"
    assert saved["objective"] == pytest.approx(8.0)
    assert saved["iterations"] >= 1
    assert saved["flows"] == [
        {"tail": "m", "head": "t", "flow": 4.0},
        {"tail": "s", "head": "m", "flow": 4.0},
    ]


def test_dimacs_example_fixture(tmp_path: Path):
    fixture = Path("examples/dimacs_small_problem.json")
    artifact = tmp_path / "dimacs_solution.json"

    problem = load_problem(fixture)
    result = solve_min_cost_flow(problem, max_iterations=5000)
    save_result(artifact, result)

    assert result.status == "optimal"
    assert math.isclose(result.objective, 30.0, rel_tol=0.0, abs_tol=1e-9)
    expected = {
        ("u0", "u1"): 5.0,
        ("u1", "u2"): 5.0,
        ("u2", "u3"): 5.0,
        ("u3", "u4"): 5.0,
    }
    assert result.flows == expected

    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["status"] == "optimal"
    assert math.isclose(saved["objective"], 30.0, rel_tol=0.0, abs_tol=1e-9)


def test_textbook_transport_fixture(tmp_path: Path):
    fixture = Path("examples/textbook_transport_problem.json")
    artifact = tmp_path / "textbook_transport_solution.json"

    problem = load_problem(fixture)
    result = solve_min_cost_flow(problem, max_iterations=5000)
    save_result(artifact, result)

    assert result.status == "optimal"
    assert math.isclose(result.objective, 85.0, rel_tol=0.0, abs_tol=1e-9)
    expected = {
        ("A", "X"): 5.0,
        ("A", "Z"): 10.0,
        ("B", "Y"): 15.0,
    }
    for key, value in expected.items():
        assert math.isclose(result.flows[key], value, rel_tol=0.0, abs_tol=1e-9)

    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["status"] == "optimal"
    assert math.isclose(saved["objective"], 85.0, rel_tol=0.0, abs_tol=1e-9)


def test_large_transport_fixture(tmp_path: Path):
    fixture = Path("examples/large_transport_problem.json")
    artifact = tmp_path / "large_transport_solution.json"

    problem = load_problem(fixture)
    result = solve_min_cost_flow(problem, max_iterations=10000)
    save_result(artifact, result)

    assert result.status in {"optimal", "iteration_limit"}
    if result.status == "optimal":
        assert math.isclose(result.objective, 100.0, rel_tol=0.0, abs_tol=1e-9)
        for idx in range(10):
            key = (f"S{idx + 1}", f"D{idx + 1}")
            assert math.isclose(result.flows[key], 10.0, rel_tol=0.0, abs_tol=1e-9)

    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["status"] in {"optimal", "iteration_limit"}
    if saved["status"] == "optimal":
        assert math.isclose(saved["objective"], 100.0, rel_tol=0.0, abs_tol=1e-6)
