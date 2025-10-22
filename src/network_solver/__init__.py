"""High-level entrypoints for the network simplex solver library."""

from .data import Basis, ProgressCallback, ProgressInfo, SolverOptions, build_problem
from .diagnostics import BasisHistory, ConvergenceMonitor
from .exceptions import (
    InfeasibleProblemError,
    InvalidProblemError,
    IterationLimitError,
    NetworkSolverError,
    NumericalInstabilityError,
    SolverConfigurationError,
    UnboundedProblemError,
)
from .preprocessing import PreprocessingResult, preprocess_and_solve, preprocess_problem
from .scaling import ScalingFactors, compute_scaling_factors, should_scale_problem
from .solver import load_problem, save_result, solve_min_cost_flow
from .specializations import NetworkType, analyze_network_structure, get_specialization_info
from .utils import (
    BottleneckArc,
    FlowPath,
    ValidationResult,
    compute_bottleneck_arcs,
    extract_path,
    validate_flow,
)
from .validation import (
    NumericAnalysis,
    NumericWarning,
    analyze_numeric_properties,
    validate_numeric_properties,
)
from .visualization import visualize_bottlenecks, visualize_flows, visualize_network

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
    # Preprocessing
    "preprocess_problem",
    "preprocess_and_solve",
    "PreprocessingResult",
    # Specializations
    "NetworkType",
    "analyze_network_structure",
    "get_specialization_info",
    # Scaling
    "should_scale_problem",
    "compute_scaling_factors",
    "ScalingFactors",
    # Utilities
    "extract_path",
    "validate_flow",
    "compute_bottleneck_arcs",
    "FlowPath",
    "ValidationResult",
    "BottleneckArc",
    # Numeric validation
    "analyze_numeric_properties",
    "validate_numeric_properties",
    "NumericAnalysis",
    "NumericWarning",
    # Diagnostics
    "ConvergenceMonitor",
    "BasisHistory",
    # Visualization
    "visualize_network",
    "visualize_flows",
    "visualize_bottlenecks",
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
