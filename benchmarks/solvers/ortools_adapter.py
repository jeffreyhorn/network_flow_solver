"""Adapter for Google OR-Tools min_cost_flow.

Requires: pip install ortools

OR-Tools is Google's highly optimized operations research library.
It provides a fast C++ implementation of network flow algorithms.
"""

import time

from ortools.graph.python import min_cost_flow

from src.network_solver.data import NetworkProblem
from .base import SolverAdapter, SolverResult


class ORToolsAdapter(SolverAdapter):
    """Adapter for Google OR-Tools SimpleMinCostFlow.

    OR-Tools implements a highly optimized network simplex algorithm in C++.
    This is the primary performance baseline for comparison.
    """

    name = "ortools"
    display_name = "Google OR-Tools"
    description = "Google OR-Tools network simplex (highly optimized C++)"

    @classmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using OR-Tools SimpleMinCostFlow."""
        try:
            # Create OR-Tools solver
            smcf = min_cost_flow.SimpleMinCostFlow()

            # Build node ID mapping (OR-Tools requires integer IDs starting from 0)
            node_ids = list(problem.nodes.keys())
            node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

            # Add arcs
            expanded_arcs = problem.undirected_expansion()
            for arc in expanded_arcs:
                tail_idx = node_to_idx[arc.tail]
                head_idx = node_to_idx[arc.head]
                capacity = int(arc.capacity) if arc.capacity is not None else 2**31 - 1
                cost = int(arc.cost * 1000)  # Scale to handle fractional costs

                smcf.add_arc_with_capacity_and_unit_cost(tail_idx, head_idx, capacity, cost)

            # Add node supplies
            for node_id, node in problem.nodes.items():
                node_idx = node_to_idx[node_id]
                supply = int(node.supply)
                smcf.set_node_supply(node_idx, supply)

            # Solve
            start = time.perf_counter()
            status = smcf.solve()
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Map status
            if status == smcf.OPTIMAL:
                # Get objective (unscale cost)
                objective = smcf.optimal_cost() / 1000.0

                return SolverResult(
                    solver_name=cls.name,
                    problem_name="",
                    status="optimal",
                    objective=objective,
                    solve_time_ms=elapsed_ms,
                    iterations=None,  # OR-Tools doesn't expose iteration count
                    metadata={
                        "algorithm": "network_simplex",
                        "has_duals": False,
                    },
                )
            elif status == smcf.INFEASIBLE:
                return SolverResult(
                    solver_name=cls.name,
                    problem_name="",
                    status="infeasible",
                    objective=None,
                    solve_time_ms=elapsed_ms,
                    iterations=None,
                )
            else:
                return SolverResult(
                    solver_name=cls.name,
                    problem_name="",
                    status="error",
                    objective=None,
                    solve_time_ms=elapsed_ms,
                    iterations=None,
                    error_message=f"OR-Tools status: {status}",
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
        """Check if OR-Tools is installed."""
        try:
            import ortools.graph.python.min_cost_flow

            return True
        except ImportError:
            return False

    @classmethod
    def get_version(cls) -> str | None:
        """Get OR-Tools version."""
        try:
            import ortools

            return ortools.__version__
        except (ImportError, AttributeError):
            return None
