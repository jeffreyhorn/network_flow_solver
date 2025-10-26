"""Adapter for NetworkX min_cost_flow."""

import time

import networkx as nx

from src.network_solver.data import NetworkProblem

from .base import SolverAdapter, SolverResult


class NetworkXAdapter(SolverAdapter):
    """Adapter for NetworkX min_cost_flow.

    Note: NetworkX uses capacity scaling algorithm which is fast but may
    not always find the true optimal solution. Use for correctness validation
    and as a fast baseline, but not as the primary performance target.
    """

    name = "networkx"
    display_name = "NetworkX"
    description = "NetworkX capacity scaling algorithm (fast approximation)"

    @classmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using NetworkX."""
        try:
            # Convert problem to NetworkX format
            graph = nx.DiGraph()

            # Add nodes with demand (NetworkX uses opposite sign)
            for node_id, node in problem.nodes.items():
                graph.add_node(node_id, demand=-node.supply)

            # Add arcs
            expanded_arcs = problem.undirected_expansion()
            for arc in expanded_arcs:
                capacity = arc.capacity if arc.capacity is not None else float("inf")
                graph.add_edge(arc.tail, arc.head, weight=arc.cost, capacity=capacity)

            # Solve
            start = time.perf_counter()
            flow_dict = nx.min_cost_flow(graph)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Calculate objective
            objective = nx.cost_of_flow(graph, flow_dict)

            return SolverResult(
                solver_name=cls.name,
                problem_name="",
                status="optimal",
                objective=objective,
                solve_time_ms=elapsed_ms,
                iterations=None,  # NetworkX doesn't report iterations
                metadata={
                    "algorithm": "capacity_scaling",
                    "has_duals": False,
                },
            )
        except nx.NetworkXUnfeasible:
            return SolverResult(
                solver_name=cls.name,
                problem_name="",
                status="infeasible",
                objective=None,
                solve_time_ms=0.0,
                iterations=None,
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
        """Check if NetworkX is available.

        NetworkX is installed as a core dependency (required for visualization),
        so this always returns True. This method exists for consistency with
        other adapter classes.
        """
        return True

    @classmethod
    def get_version(cls) -> str | None:
        """Get NetworkX version."""
        try:
            return nx.__version__
        except AttributeError:
            return None
