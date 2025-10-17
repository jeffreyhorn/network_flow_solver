"""Example script demonstrating usage of the network simplex solver."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import load_problem, save_result, solve_min_cost_flow  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    problem_path = base_dir / "sample_problem.json"
    output_path = base_dir / "sample_solution.json"

    problem = load_problem(problem_path)
    result = solve_min_cost_flow(problem)
    save_result(output_path, result)

    print(f"Solved {problem_path.name}: status={result.status}, objective={result.objective}")

    # Display dual values (node potentials) for sensitivity analysis
    if result.duals:
        print("\nDual values (shadow prices):")
        for node_id, dual in sorted(result.duals.items()):
            print(f"  {node_id}: {dual:.6f}")


if __name__ == "__main__":
    main()
