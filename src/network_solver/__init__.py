"""High-level entrypoints for the network simplex solver library."""

from .data import ProgressCallback, ProgressInfo, build_problem
from .exceptions import (
    InfeasibleProblemError,
    InvalidProblemError,
    IterationLimitError,
    NetworkSolverError,
    NumericalInstabilityError,
    SolverConfigurationError,
    UnboundedProblemError,
)
from .solver import load_problem, save_result, solve_min_cost_flow

__version__ = "0.1.0"

__all__ = [
    # Main API
    "build_problem",
    "load_problem",
    "solve_min_cost_flow",
    "save_result",
    # Progress tracking
    "ProgressCallback",
    "ProgressInfo",
    # Exceptions
    "NetworkSolverError",
    "InvalidProblemError",
    "InfeasibleProblemError",
    "UnboundedProblemError",
    "NumericalInstabilityError",
    "IterationLimitError",
    "SolverConfigurationError",
    # Version
    "__version__",
]
