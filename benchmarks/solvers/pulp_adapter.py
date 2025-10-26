"""Adapter for PuLP linear programming solver.

Requires: pip install pulp

PuLP is a linear programming modeling library that can formulate
min-cost flow as an LP and solve it with various backends (COIN-OR, GLPK, etc.).
"""

import time

import pulp

from src.network_solver.data import NetworkProblem
from .base import SolverAdapter, SolverResult


class PuLPAdapter(SolverAdapter):
    """Adapter for PuLP with network flow formulation.

    PuLP formulates the min-cost flow problem as a linear program and
    solves it using a general LP solver. This is typically slower than
    specialized network flow algorithms but provides good validation.
    """

    name = "pulp"
    display_name = "PuLP"
    description = "PuLP LP formulation with COIN-OR/GLPK backend"

    @classmethod
    def solve(cls, problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using PuLP LP formulation."""
        try:
            # Create LP problem
            lp_prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)

            # Create variables for arc flows
            expanded_arcs = problem.undirected_expansion()
            flow_vars = {}

            for i, arc in enumerate(expanded_arcs):
                var_name = f"flow_{arc.tail}_{arc.head}_{i}"
                lower = arc.lower if arc.lower is not None else 0
                upper = arc.capacity if arc.capacity is not None else None

                flow_vars[(arc.tail, arc.head, i)] = pulp.LpVariable(
                    var_name, lowBound=lower, upBound=upper, cat=pulp.LpContinuous
                )

            # Objective: minimize total cost
            lp_prob += pulp.lpSum(
                [
                    arc.cost * flow_vars[(arc.tail, arc.head, i)]
                    for i, arc in enumerate(expanded_arcs)
                ]
            )

            # Constraints: flow conservation at each node
            for node_id, node in problem.nodes.items():
                # Inflow - outflow = supply
                inflow = pulp.lpSum(
                    [
                        flow_vars[(arc.tail, arc.head, i)]
                        for i, arc in enumerate(expanded_arcs)
                        if arc.head == node_id
                    ]
                )
                outflow = pulp.lpSum(
                    [
                        flow_vars[(arc.tail, arc.head, i)]
                        for i, arc in enumerate(expanded_arcs)
                        if arc.tail == node_id
                    ]
                )

                lp_prob += inflow - outflow == node.supply, f"flow_conservation_{node_id}"

            # Solve
            start = time.perf_counter()
            lp_prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(timeout_s)))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Get status
            status_map = {
                pulp.LpStatusOptimal: "optimal",
                pulp.LpStatusInfeasible: "infeasible",
                pulp.LpStatusUnbounded: "unbounded",
                pulp.LpStatusNotSolved: "error",
            }
            status = status_map.get(lp_prob.status, "error")

            if status == "optimal":
                objective = pulp.value(lp_prob.objective)

                return SolverResult(
                    solver_name=cls.name,
                    problem_name="",
                    status=status,
                    objective=objective,
                    solve_time_ms=elapsed_ms,
                    iterations=None,  # PuLP doesn't expose iteration count
                    metadata={
                        "solver": "COIN-OR CBC",
                        "formulation": "LP",
                    },
                )
            else:
                return SolverResult(
                    solver_name=cls.name,
                    problem_name="",
                    status=status,
                    objective=None,
                    solve_time_ms=elapsed_ms,
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
        """Check if PuLP is installed."""
        try:
            import pulp

            return True
        except ImportError:
            return False

    @classmethod
    def get_version(cls) -> str | None:
        """Get PuLP version."""
        try:
            import pulp

            return pulp.__version__
        except (ImportError, AttributeError):
            return None
