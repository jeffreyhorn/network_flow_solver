import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _replicate_examples(tmp_path: Path) -> Path:
    # Copy the sample CLI assets into an isolated workspace the subprocess can mutate.
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"

    dest = tmp_path / "examples"
    dest.mkdir(parents=True, exist_ok=True)

    for name in (
        "solve_example.py",
        "sample_problem.json",
        "solve_dimacs_example.py",
        "dimacs_small_problem.json",
        "solve_textbook_transport.py",
        "textbook_transport_problem.json",
        "solve_large_transport.py",
        "large_transport_problem.json",
    ):
        shutil.copy2(examples_dir / name, dest / name)

    # Provide src/ so the example script can import using its relative path logic.
    src_symlink = tmp_path / "src"
    src_origin = repo_root / "src"
    if not src_symlink.exists():
        src_symlink.symlink_to(src_origin, target_is_directory=True)

    return dest


def test_example_cli_script_produces_solution(tmp_path: Path):
    # Run the example script as a subprocess to mimic the documented CLI usage.
    examples_dir = _replicate_examples(tmp_path)
    script_path = examples_dir / "solve_example.py"
    output_path = examples_dir / "sample_solution.json"

    if output_path.exists():
        output_path.unlink()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Solved sample_problem.json" in proc.stdout
    assert proc.stderr == ""

    assert output_path.exists()
    contents = json.loads(output_path.read_text(encoding="utf-8"))

    assert contents["status"] == "optimal"
    assert pytest.approx(contents["objective"]) == 15.0
    flows = {(entry["tail"], entry["head"]): entry["flow"] for entry in contents["flows"]}
    assert flows[("s", "a")] == pytest.approx(5.0)
    assert flows[("a", "t")] == pytest.approx(5.0)


def test_dimacs_cli_script_produces_solution(tmp_path: Path):
    # DIMACS adapter should still yield a solvable instance in the CLI flow.
    examples_dir = _replicate_examples(tmp_path)
    script_path = examples_dir / "solve_dimacs_example.py"
    output_path = examples_dir / "dimacs_small_solution.json"

    if output_path.exists():
        output_path.unlink()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Solved dimacs_small_problem.json" in proc.stdout
    assert proc.stderr == ""

    assert output_path.exists()
    contents = json.loads(output_path.read_text(encoding="utf-8"))

    assert contents["status"] == "optimal"
    assert pytest.approx(contents["objective"], abs=1e-6) == 30.0
    assert contents["flows"] == [
        {"tail": "u0", "head": "u1", "flow": 5.0},
        {"tail": "u1", "head": "u2", "flow": 5.0},
        {"tail": "u2", "head": "u3", "flow": 5.0},
        {"tail": "u3", "head": "u4", "flow": 5.0},
    ]


def test_textbook_cli_script_produces_solution(tmp_path: Path):
    # Textbook transportation sample validates handling of rectangular supply/demand grids.
    examples_dir = _replicate_examples(tmp_path)
    script_path = examples_dir / "solve_textbook_transport.py"
    output_path = examples_dir / "textbook_transport_solution.json"

    if output_path.exists():
        output_path.unlink()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Solved textbook_transport_problem.json" in proc.stdout
    assert proc.stderr == ""

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "optimal"
    assert pytest.approx(result["objective"], abs=1e-6) == 85.0
    assert result["flows"] == [
        {"tail": "A", "head": "X", "flow": 5.0},
        {"tail": "A", "head": "Z", "flow": 10.0},
        {"tail": "B", "head": "Y", "flow": 15.0},
    ]


def test_large_transport_cli_script(tmp_path: Path):
    # Large instance may hit iteration limits locally, so only assert guardrails.
    examples_dir = _replicate_examples(tmp_path)
    script_path = examples_dir / "solve_large_transport.py"
    output_path = examples_dir / "large_transport_solution.json"

    if output_path.exists():
        output_path.unlink()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Solved large_transport_problem.json" in proc.stdout
    assert proc.stderr == ""

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] in {"optimal", "iteration_limit"}
    if result["status"] == "optimal":
        assert pytest.approx(result["objective"], abs=1e-6) == 100.0
        diagonal = {(f"S{i + 1}", f"D{i + 1}"): 10.0 for i in range(10)}
        for entry in result["flows"]:
            tail, head, flow = entry["tail"], entry["head"], entry["flow"]
            if (tail, head) in diagonal:
                assert pytest.approx(flow, abs=1e-6) == diagonal[(tail, head)]


def test_cli_fails_with_missing_edges(tmp_path: Path):
    # Malformed JSON missing edges should cause the CLI to abort with a clear error.
    examples_dir = _replicate_examples(tmp_path)
    problem_path = examples_dir / "sample_problem.json"
    original = json.loads(problem_path.read_text(encoding="utf-8"))
    problem_path.write_text(json.dumps({"nodes": original["nodes"]}), encoding="utf-8")

    script_path = examples_dir / "solve_example.py"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "Problem JSON must include 'nodes' and 'edges' arrays" in proc.stderr


def test_cli_fails_when_lower_exceeds_capacity(tmp_path: Path):
    # Capacity vs lower bound mismatch must also bubble up as a failure in the CLI.
    examples_dir = _replicate_examples(tmp_path)
    problem_path = examples_dir / "sample_problem.json"
    payload = json.loads(problem_path.read_text(encoding="utf-8"))
    payload["edges"][0].update({"capacity": 1.0, "lower": 5.0})
    problem_path.write_text(json.dumps(payload), encoding="utf-8")

    script_path = examples_dir / "solve_example.py"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "Arc capacity must be >= lower bound" in proc.stderr
