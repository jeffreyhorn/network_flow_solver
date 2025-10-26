"""Base classes for solver adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.network_solver.data import NetworkProblem


@dataclass
class SolverResult:
    """Results from a single solver run."""

    solver_name: str
    problem_name: str
    status: str  # 'optimal', 'infeasible', 'timeout', 'error'
    objective: float | None
    solve_time_ms: float
    iterations: int | None  # None if solver doesn't report
    error_message: str | None = None

    # Additional solver-specific metadata
    metadata: dict | None = None


class SolverAdapter(ABC):
    """Abstract base class for solver adapters.

    Each solver adapter implements a common interface for solving network
    flow problems, allowing fair comparison across different implementations.

    Attributes:
        name: Unique identifier for the solver (e.g., "network_solver", "ortools")
        display_name: Human-readable name for reports (e.g., "Network Solver", "Google OR-Tools")
        description: Brief description of the solver
    """

    name: str = "base"
    display_name: str = "Base Solver"
    description: str = "Abstract base solver"

    @classmethod
    @abstractmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve a network flow problem.

        Args:
            problem: NetworkProblem instance to solve
            timeout_s: Maximum time allowed for solving (seconds)

        Returns:
            SolverResult with solution information

        Note:
            Implementations should handle exceptions gracefully and return
            SolverResult with status='error' rather than raising.
        """
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        """Check if this solver is available (dependencies installed).

        Returns:
            True if solver can be used, False otherwise.

        Note:
            Base implementation returns True. Subclasses should override
            to check for required dependencies.
        """
        return True

    @classmethod
    def get_version(cls) -> str | None:
        """Get version of the solver library.

        Returns:
            Version string, or None if unavailable.
        """
        return None
