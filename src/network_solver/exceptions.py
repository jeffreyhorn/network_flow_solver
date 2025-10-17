"""Custom exceptions for the network solver library."""

from __future__ import annotations


class NetworkSolverError(Exception):
    """Base exception for all network solver errors.

    All custom exceptions in the network_solver package inherit from this class,
    allowing users to catch all solver-related errors with a single except clause.

    Example:
        try:
            result = solve_min_cost_flow(problem)
        except NetworkSolverError as e:
            print(f"Solver error: {e}")
    """


class InvalidProblemError(NetworkSolverError):
    """Raised when a problem definition is invalid or malformed.

    This includes:
    - Unbalanced supply/demand (total supply ≠ 0)
    - Missing nodes referenced in arcs
    - Invalid arc definitions (self-loops, capacity < lower bound)
    - Malformed JSON input
    - Type errors in problem specification

    Example:
        # Unbalanced problem
        InvalidProblemError("Problem is unbalanced: total supply 5.0 exceeds tolerance 1e-3")

        # Missing node
        InvalidProblemError("Arc tail 'missing_node' not found in node set")
    """


class InfeasibleProblemError(NetworkSolverError):
    """Raised when the problem has no feasible solution.

    This occurs when the network simplex algorithm completes Phase 1 but cannot
    drive all artificial variables to zero, indicating that the supply/demand
    constraints cannot be satisfied with the given arc capacities.

    The problem may be infeasible due to:
    - Insufficient arc capacity to satisfy demands
    - Disconnected components with non-zero supply
    - Conflicting constraints

    Example:
        # Supply cannot reach demand
        InfeasibleProblemError(
            "No feasible flow exists: artificial arcs have positive flow after Phase 1",
            iterations=42
        )
    """

    def __init__(self, message: str, iterations: int = 0):
        """Initialize with message and optional iteration count."""
        super().__init__(message)
        self.iterations = iterations


class UnboundedProblemError(NetworkSolverError):
    """Raised when the problem is unbounded (objective can decrease without limit).

    This occurs during a simplex pivot when the entering variable can increase
    indefinitely without violating any constraints. In network flow problems,
    this typically indicates:
    - A negative-cost cycle with no capacity restrictions
    - Malformed problem specification

    Unbounded problems are detected during the pivot operation when theta = ∞.

    Example:
        UnboundedProblemError(
            "Unbounded problem detected: entering arc can increase indefinitely",
            entering_arc=("A", "B"),
            reduced_cost=-5.0
        )
    """

    def __init__(
        self,
        message: str,
        entering_arc: tuple[str, str] | None = None,
        reduced_cost: float | None = None,
    ):
        """Initialize with message and optional diagnostic information."""
        super().__init__(message)
        self.entering_arc = entering_arc
        self.reduced_cost = reduced_cost


class NumericalInstabilityError(NetworkSolverError):
    """Raised when numerical issues prevent reliable computation.

    This can occur due to:
    - Ill-conditioned basis matrices (near-singular)
    - Extreme coefficient ranges (very large or very small values)
    - Accumulated floating-point errors
    - Factorization failures

    When this error is raised, consider:
    - Scaling the problem (normalize costs and capacities)
    - Adjusting the tolerance parameter
    - Using higher precision data types
    - Reformulating the problem

    Example:
        NumericalInstabilityError(
            "Basis matrix is singular: cannot compute node potentials",
            condition_number=1e15
        )
    """

    def __init__(self, message: str, condition_number: float | None = None):
        """Initialize with message and optional condition number."""
        super().__init__(message)
        self.condition_number = condition_number


class IterationLimitError(NetworkSolverError):
    """Raised when the solver reaches the iteration limit before converging.

    This is technically not an error condition - the solver returns a partial
    solution that may be feasible but not proven optimal. This exception is
    provided for users who want to treat iteration limits as errors.

    Note: By default, the solver returns a FlowResult with status="iteration_limit"
    rather than raising this exception. Users can check the status field instead.

    Example:
        IterationLimitError(
            "Iteration limit reached: 1000 iterations completed",
            iterations=1000,
            objective=123.45,
            status="feasible"
        )
    """

    def __init__(
        self,
        message: str,
        iterations: int = 0,
        objective: float | None = None,
        status: str = "unknown",
    ):
        """Initialize with message and solution state."""
        super().__init__(message)
        self.iterations = iterations
        self.objective = objective
        self.status = status


class SolverConfigurationError(NetworkSolverError):
    """Raised when solver configuration or options are invalid.

    This includes:
    - Invalid parameter values (negative iterations, invalid tolerance)
    - Incompatible option combinations
    - Unsupported features or modes

    Example:
        SolverConfigurationError("max_iterations must be positive, got -1")
    """
