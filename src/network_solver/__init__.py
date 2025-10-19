"""High-level entrypoints for the network simplex solver library."""

from .data import Basis, ProgressCallback, ProgressInfo, SolverOptions, build_problem
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
from .utils import (
    BottleneckArc,
    FlowPath,
    ValidationResult,
    compute_bottleneck_arcs,
    extract_path,
    validate_flow,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "build_problem",
    "load_problem",
    "solve_min_cost_flow",
    "save_result",
    # Configuration
    "SolverOptions",
    "Basis",
    # Progress tracking
    "ProgressCallback",
    "ProgressInfo",
    # Utilities
    "extract_path",
    "validate_flow",
    "compute_bottleneck_arcs",
    "FlowPath",
    "ValidationResult",
    "BottleneckArc",
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
