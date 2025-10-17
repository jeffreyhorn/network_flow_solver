import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import FlowResult, build_problem  # noqa: E402
from network_solver.io import load_problem, save_result  # noqa: E402

# These tests pin the JSON contract implemented by network_solver.io.


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_problem_prefers_edges_key(tmp_path: Path):
    # When both edges and arcs exist, the loader should prioritise the canonical edges field.
    payload = {
        "directed": True,
        "tolerance": 1e-4,
        "nodes": [
            {"id": "s", "supply": 5.0},
            {"id": "t", "supply": -5.0},
        ],
        "edges": [
            {"tail": "s", "head": "t", "capacity": 7.5, "cost": 3.0, "lower": 1.5},
        ],
        "arcs": [
            {"tail": "s", "head": "t", "capacity": 9.0, "cost": 4.0},
        ],
    }

    path = _write_payload(tmp_path, payload)
    problem = load_problem(path)

    assert problem.directed is True
    assert pytest.approx(problem.tolerance) == 1e-4
    assert len(problem.arcs) == 1
    arc = problem.arcs[0]
    assert arc.tail == "s"
    assert arc.head == "t"
    assert arc.capacity == pytest.approx(7.5)
    assert arc.cost == pytest.approx(3.0)
    assert arc.lower == pytest.approx(1.5)


def test_load_problem_requires_tail_and_head(tmp_path: Path):
    # Inputs lacking mandatory tail/head fields should surface a KeyError.
    payload = {
        "nodes": [
            {"id": "s", "supply": 1.0},
            {"id": "t", "supply": -1.0},
        ],
        "edges": [
            {"tail": "s", "capacity": 5.0},
        ],
    }

    path = _write_payload(tmp_path, payload)
    with pytest.raises(KeyError, match="tail and head identifiers"):
        load_problem(path)


def test_load_problem_requires_list_payloads(tmp_path: Path):
    # Non-list payloads violate the JSON schema and must be rejected.
    payload = {
        "nodes": {"id": "s", "supply": 1.0},
        "edges": [],
    }

    path = _write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="Problem JSON must include 'nodes' and 'edges' arrays"):
        load_problem(path)


def test_load_problem_rejects_missing_edges_key(tmp_path: Path):
    # Entirely missing the edge list should trigger the same schema error as wrong types.
    payload = {
        "nodes": [
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": -1.0},
        ],
    }

    path = _write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="Problem JSON must include 'nodes' and 'edges' arrays"):
        load_problem(path)


def test_build_problem_rejects_lower_greater_than_capacity():
    # Lower bounds greater than capacity are invalid and must raise a ValueError.
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 1.0, "cost": 0.0, "lower": 2.0},
    ]

    with pytest.raises(ValueError, match="Arc capacity must be >= lower bound"):
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def test_save_result_serializes_sorted_flows(tmp_path: Path):
    # Output should be deterministic to keep fixture diffs stable in version control.
    result = FlowResult(
        objective=12.5,
        flows={
            ("b", "a"): 3.0,
            ("a", "c"): 1.0,
            ("a", "b"): 2.0,
        },
        status="optimal",
        iterations=7,
    )

    path = tmp_path / "result.json"
    save_result(path, result)

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    assert data["status"] == "optimal"
    assert data["objective"] == pytest.approx(12.5)
    assert data["iterations"] == 7
    assert data["flows"] == [
        {"tail": "a", "head": "b", "flow": 2.0},
        {"tail": "a", "head": "c", "flow": 1.0},
        {"tail": "b", "head": "a", "flow": 3.0},
    ]
    # Indented JSON is easier to read; confirm indentation markers are present.
    assert '\n  "status"' in raw


def test_save_result_propagates_path_errors(monkeypatch, tmp_path: Path):
    # IOError scenarios should bubble out so callers can react appropriately.
    result = FlowResult(objective=1.0, flows={}, status="optimal", iterations=0)
    target = tmp_path / "blocked.json"

    # Simulate filesystem denial so we can assert the failure surfaces to callers.
    def fake_open(self, *_args, **_kwargs):
        raise PermissionError("cannot write")

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(PermissionError, match="cannot write"):
        save_result(target, result)


def test_load_problem_propagates_file_not_found(tmp_path: Path):
    """Test that load_problem raises FileNotFoundError for missing files."""
    nonexistent = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        load_problem(nonexistent)


def test_load_problem_propagates_json_decode_error(tmp_path: Path):
    """Test that load_problem raises JSONDecodeError for invalid JSON."""
    path = tmp_path / "invalid.json"
    path.write_text("{ this is not valid json }", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_problem(path)


def test_load_problem_handles_empty_file(tmp_path: Path):
    """Test that load_problem handles empty JSON files."""
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Problem JSON must include"):
        load_problem(path)


def test_load_problem_handles_non_boolean_directed(tmp_path: Path):
    """Test that load_problem converts directed field to bool."""
    payload = {
        "directed": "true",  # String instead of boolean
        "nodes": [{"id": "a", "supply": 1.0}, {"id": "b", "supply": -1.0}],
        "edges": [{"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0}],
    }
    path = _write_payload(tmp_path, payload)

    problem = load_problem(path)
    assert isinstance(problem.directed, bool)
    assert problem.directed is True


def test_load_problem_handles_missing_directed_field(tmp_path: Path):
    """Test that load_problem defaults directed to True when missing."""
    payload = {
        "nodes": [{"id": "a", "supply": 1.0}, {"id": "b", "supply": -1.0}],
        "edges": [{"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0}],
    }
    path = _write_payload(tmp_path, payload)

    problem = load_problem(path)
    assert problem.directed is True


def test_load_problem_handles_missing_tolerance_field(tmp_path: Path):
    """Test that load_problem defaults tolerance to 1e-3 when missing."""
    payload = {
        "nodes": [{"id": "a", "supply": 1.0}, {"id": "b", "supply": -1.0}],
        "edges": [{"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0}],
    }
    path = _write_payload(tmp_path, payload)

    problem = load_problem(path)
    assert problem.tolerance == pytest.approx(1e-3)


def test_normalize_edges_requires_head(tmp_path: Path):
    """Test that _normalize_edges requires head field."""
    payload = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"tail": "a"}],  # Missing head
    }
    path = _write_payload(tmp_path, payload)

    with pytest.raises(KeyError, match="tail and head"):
        load_problem(path)


def test_normalize_edges_provides_defaults(tmp_path: Path):
    """Test that _normalize_edges provides default values."""
    payload = {
        "nodes": [
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": -1.0},
        ],
        "edges": [
            {"tail": "a", "head": "b"}  # No capacity, cost, or lower
        ],
    }
    path = _write_payload(tmp_path, payload)

    problem = load_problem(path)
    arc = problem.arcs[0]
    assert arc.capacity is None
    assert arc.cost == 0.0
    assert arc.lower == 0.0


def test_save_result_handles_empty_flows(tmp_path: Path):
    """Test that save_result handles results with no flows."""
    result = FlowResult(objective=0.0, flows={}, status="infeasible", iterations=0)
    path = tmp_path / "empty_flows.json"

    save_result(path, result)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["flows"] == []
    assert data["status"] == "infeasible"


def test_save_result_handles_special_float_values(tmp_path: Path):
    """Test that save_result handles special float values in objective."""
    result = FlowResult(
        objective=float("inf"),
        flows={("a", "b"): 1.0},
        status="unbounded",
        iterations=5,
    )
    path = tmp_path / "special.json"

    save_result(path, result)

    # JSON should serialize successfully
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "unbounded"


def test_load_problem_accepts_string_path(tmp_path: Path):
    """Test that load_problem accepts string paths."""
    payload = {
        "nodes": [{"id": "a", "supply": 1.0}, {"id": "b", "supply": -1.0}],
        "edges": [{"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0}],
    }
    path = _write_payload(tmp_path, payload)

    # Pass as string instead of Path
    problem = load_problem(str(path))
    assert problem is not None


def test_save_result_accepts_string_path(tmp_path: Path):
    """Test that save_result accepts string paths."""
    result = FlowResult(objective=0.0, flows={}, status="optimal", iterations=0)
    path = tmp_path / "result.json"

    # Pass as string instead of Path
    save_result(str(path), result)
    assert path.exists()


def test_load_problem_with_invalid_node_structure(tmp_path: Path):
    """Test that load_problem handles invalid node structures."""
    payload = {
        "nodes": "not a list",  # Should be a list
        "edges": [],
    }
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="Problem JSON must include"):
        load_problem(path)
