"""Public solver entrypoints."""

from __future__ import annotations

from pathlib import Path

from .data import Basis, FlowResult, NetworkProblem, ProgressCallback, SolverOptions
from .io import load_problem as load_problem_file
from .io import save_result as save_result_file
from .simplex import NetworkSimplex


def solve_min_cost_flow(
    problem: NetworkProblem,
    options: SolverOptions | None = None,
    max_iterations: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 100,
    warm_start_basis: Basis | None = None,
) -> FlowResult:
    """Solve a minimum-cost flow problem using the network simplex algorithm.

    This is the main entry point for solving network flow problems. It uses a specialized
    network simplex algorithm that exploits the structure of flow problems for efficiency.

    Args:
        problem: The network flow problem to solve. Must be balanced (supplies = demands).
        options: Solver configuration options. If None, uses defaults.
                See SolverOptions for tuning parameters.
        max_iterations: Maximum number of simplex iterations. Overrides options.max_iterations if provided.
                       If None, defaults to max(100, 5*num_arcs).
        progress_callback: Optional callback function to receive progress updates.
                          Called every progress_interval iterations with ProgressInfo.
        progress_interval: Number of iterations between progress callbacks (default: 100).
        warm_start_basis: Optional basis from a previous solve to initialize the solver.
                         Provides a "warm start" by reusing the spanning tree structure,
                         which can significantly reduce iterations for similar problems.
                         Use result.basis from a previous solve.

    Returns:
        FlowResult containing:
        - objective: Total cost of the solution
        - flows: Flow values on each arc
        - status: 'optimal', 'infeasible', 'unbounded', or 'iteration_limit'
        - iterations: Number of iterations performed
        - duals: Node potentials (shadow prices) for sensitivity analysis
        - basis: Spanning tree basis for warm-starting future solves

    Raises:
        InvalidProblemError: If problem is malformed (unbalanced, invalid arcs, etc.).
        UnboundedProblemError: If problem has unbounded objective (negative-cost cycle).

    Time Complexity:
        - Best case: O(n²m) where n = nodes, m = arcs
        - Average case: O(nm log n) with good pivoting (Devex pricing)
        - Worst case: O(n²m²) in pathological cases
        - In practice: Often much faster, especially for sparse networks

    Space Complexity:
        O(n + m) for storing the network plus O(n²) for basis factorization

    Examples:
        >>> from network_solver import solve_min_cost_flow, build_problem
        >>> # Simple 2-node problem
        >>> nodes = [
        ...     {"id": "source", "supply": 100.0},
        ...     {"id": "sink", "supply": -100.0}
        ... ]
        >>> arcs = [{"tail": "source", "head": "sink", "capacity": 150.0, "cost": 2.5}]
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>> result = solve_min_cost_flow(problem)
        >>> print(f"Status: {result.status}, Cost: ${result.objective:.2f}")
        Status: optimal, Cost: $250.00
        >>>
        >>> # Warm-start with modified problem
        >>> nodes2 = [
        ...     {"id": "source", "supply": 120.0},  # Increased supply
        ...     {"id": "sink", "supply": -120.0}
        ... ]
        >>> problem2 = build_problem(nodes2, arcs, directed=True, tolerance=1e-6)
        >>> result2 = solve_min_cost_flow(problem2, warm_start_basis=result.basis)
        >>> print(f"Warm-start iterations: {result2.iterations}")
        Warm-start iterations: 2

    See Also:
        - NetworkProblem: Problem definition structure
        - SolverOptions: Configuration and tuning parameters
        - FlowResult: Solution output format
        - docs/algorithm.md: Network simplex algorithm details
        - docs/benchmarks.md: Performance characteristics and tuning

    Note:
        For convenience, max_iterations can be passed directly as a parameter, which will
        override the value in options if both are provided. For full control over solver
        behavior, use the SolverOptions parameter.
    """
    # Instantiate a fresh solver each call to avoid cross-run state sharing.
    solver = NetworkSimplex(problem, options=options)
    return solver.solve(
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        warm_start_basis=warm_start_basis,
    )


def load_problem(path: str | Path) -> NetworkProblem:
    """Load a network flow problem from a JSON file.

    Args:
        path: Path to JSON file containing problem definition.

    Returns:
        NetworkProblem instance ready to solve.

    Raises:
        FileNotFoundError: If file does not exist.
        InvalidProblemError: If JSON is malformed or problem is invalid.

    Time Complexity:
        O(n + m) where n = nodes, m = arcs (parsing and validation)

    Space Complexity:
        O(n + m) for storing the problem

    Examples:
        >>> from network_solver import load_problem, solve_min_cost_flow
        >>> problem = load_problem("examples/textbook_transport_problem.json")
        >>> result = solve_min_cost_flow(problem)
        >>> print(f"Loaded {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")
        Loaded 5 nodes, 6 arcs

    See Also:
        - save_result(): Save solution to JSON
        - build_problem(): Construct problem from dictionaries
    """
    # Reuse the IO helpers so callers interact with a single parsing implementation.
    return load_problem_file(path)


def save_result(path: str | Path, result: FlowResult) -> None:
    """Save a flow solution to a JSON file.

    Args:
        path: Path where JSON file will be written.
        result: FlowResult from solve_min_cost_flow().

    Raises:
        OSError: If file cannot be written.

    Time Complexity:
        O(m) where m = number of arcs with non-zero flow

    Space Complexity:
        O(m) for JSON serialization

    Examples:
        >>> from network_solver import solve_min_cost_flow, save_result
        >>> result = solve_min_cost_flow(problem)
        >>> save_result("solution.json", result)
        >>> # File contains objective, flows, status, iterations, duals

    See Also:
        - load_problem(): Load problem from JSON
    """
    # Mirror load_problem to keep round-trip logic encapsulated in the IO layer.
    save_result_file(path, result)
