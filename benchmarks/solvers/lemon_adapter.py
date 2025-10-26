"""Adapter for LEMON C++ library.

LEMON (Library for Efficient Modeling and Optimization in Networks) is a
high-performance C++ library for network optimization. Using it from Python
requires either:
1. Python bindings (not widely available)
2. Command-line wrapper (requires compilation)
3. ctypes/cffi interface (complex setup)

For now, this adapter is a placeholder. Implementation would require:
- Compiling LEMON C++ library
- Creating Python bindings or CLI wrapper
- Installing on system

See: https://lemon.cs.elte.hu/
"""

from src.network_solver.data import NetworkProblem

from .base import SolverAdapter, SolverResult


class LEMONAdapter(SolverAdapter):
    """Adapter for LEMON library (placeholder).

    LEMON provides highly optimized network flow algorithms but requires
    C++ compilation and Python bindings. This adapter is not currently
    implemented but serves as a placeholder for future extension.
    """

    name = "lemon"
    display_name = "LEMON"
    description = "LEMON C++ library (requires manual installation - not available)"

    @classmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using LEMON (not implemented)."""
        return SolverResult(
            solver_name=cls.name,
            problem_name="",
            status="error",
            objective=None,
            solve_time_ms=0.0,
            iterations=None,
            error_message="LEMON adapter not implemented. Requires C++ compilation and Python bindings.",
        )

    @classmethod
    def is_available(cls) -> bool:
        """LEMON is not available via pip."""
        return False

    @classmethod
    def get_version(cls) -> str | None:
        """LEMON version not available."""
        return None
