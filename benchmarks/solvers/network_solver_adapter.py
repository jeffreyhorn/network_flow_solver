"""Adapter for our network_solver implementation."""

import time

from src.network_solver.data import NetworkProblem
from src.network_solver.solver import solve_min_cost_flow

from .base import SolverAdapter, SolverResult


class NetworkSolverAdapter(SolverAdapter):
    """Adapter for our network simplex implementation."""

    name = "network_solver"
    display_name = "Network Solver"
    description = "Our network simplex implementation with Devex/Dantzig pricing"

    @classmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using network_solver."""
        try:
            start = time.perf_counter()
            result = solve_min_cost_flow(problem, max_iterations=100000)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return SolverResult(
                solver_name=cls.name,
                problem_name="",  # Will be set by caller
                status=result.status,
                objective=result.objective if result.status == "optimal" else None,
                solve_time_ms=elapsed_ms,
                iterations=result.iterations,
                metadata={
                    "pricing_strategy": "auto-detected",
                    "has_duals": True,
                },
            )
        except Exception as e:
            return SolverResult(
                solver_name=cls.name,
                problem_name="",
                status="error",
                objective=None,
                solve_time_ms=0.0,
                iterations=None,
                error_message=str(e),
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if network_solver is available.

        network_solver is the core package, so this always returns True.
        This method exists for consistency with other adapter classes.
        """
        return True

    @classmethod
    def get_version(cls) -> str | None:
        """Get version of network_solver."""
        try:
            from src.network_solver import __version__

            return __version__
        except (ImportError, AttributeError):
            return "unknown"
