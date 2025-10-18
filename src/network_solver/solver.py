"""Public solver entrypoints."""

from __future__ import annotations

from pathlib import Path

from .data import FlowResult, NetworkProblem, ProgressCallback
from .io import load_problem as load_problem_file
from .io import save_result as save_result_file
from .simplex import NetworkSimplex


def solve_min_cost_flow(
    problem: NetworkProblem,
    max_iterations: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 100,
) -> FlowResult:
    """Run the network simplex solver on the provided problem definition.

    Args:
        problem: The network flow problem to solve.
        max_iterations: Maximum number of simplex iterations (default: max(100, 5*num_arcs)).
        progress_callback: Optional callback function to receive progress updates.
        progress_interval: Number of iterations between progress callbacks (default: 100).

    Returns:
        FlowResult containing solution, dual values, and solver statistics.
    """
    # Instantiate a fresh solver each call to avoid cross-run state sharing.
    solver = NetworkSimplex(problem)
    return solver.solve(
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )


def load_problem(path: str | Path) -> NetworkProblem:
    """Load a problem instance from disk."""
    # Reuse the IO helpers so callers interact with a single parsing implementation.
    return load_problem_file(path)


def save_result(path: str | Path, result: FlowResult) -> None:
    """Persist a solver result to disk."""
    # Mirror load_problem to keep round-trip logic encapsulated in the IO layer.
    save_result_file(path, result)
