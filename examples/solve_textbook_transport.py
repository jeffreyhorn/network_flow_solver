"""Solve the textbook transportation example and store the result."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import load_problem, save_result, solve_min_cost_flow  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    problem_path = base_dir / "textbook_transport_problem.json"
    output_path = base_dir / "textbook_transport_solution.json"

    problem = load_problem(problem_path)
    result = solve_min_cost_flow(problem, max_iterations=5000)
    save_result(output_path, result)
    print(
        f"Solved {problem_path.name}: status={result.status}, "
        f"objective={result.objective}"
    )


if __name__ == "__main__":
    main()
