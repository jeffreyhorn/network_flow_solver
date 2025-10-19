"""CLI script that solves the DIMACS-inspired transportation example."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import load_problem, save_result, solve_min_cost_flow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve DIMACS-inspired transportation problem")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    base_dir = Path(__file__).resolve().parent
    problem_path = base_dir / "dimacs_small_problem.json"
    output_path = base_dir / "dimacs_small_solution.json"

    problem = load_problem(problem_path)
    result = solve_min_cost_flow(problem)
    save_result(output_path, result)
    print(f"Solved {problem_path.name}: status={result.status}, objective={result.objective}")


if __name__ == "__main__":
    main()
