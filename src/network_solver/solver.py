"""Public solver entrypoints."""

from __future__ import annotations

from pathlib import Path

from .data import FlowResult, NetworkProblem
from .io import load_problem as load_problem_file
from .io import save_result as save_result_file
from .simplex import NetworkSimplex


def solve_min_cost_flow(problem: NetworkProblem, max_iterations: int | None = None) -> FlowResult:
    """Run the network simplex solver on the provided problem definition."""
    # Instantiate a fresh solver each call to avoid cross-run state sharing.
    solver = NetworkSimplex(problem)
    return solver.solve(max_iterations=max_iterations)


def load_problem(path: str | Path) -> NetworkProblem:
    """Load a problem instance from disk."""
    # Reuse the IO helpers so callers interact with a single parsing implementation.
    return load_problem_file(path)


def save_result(path: str | Path, result: FlowResult) -> None:
    """Persist a solver result to disk."""
    # Mirror load_problem to keep round-trip logic encapsulated in the IO layer.
    save_result_file(path, result)
