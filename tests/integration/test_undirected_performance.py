import math
import sys
from pathlib import Path
from typing import Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import NetworkProblem, build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402


def _build_undirected_chain(
    node_count: int = 150,
    total_supply: float = 900.0,
) -> Tuple[NetworkProblem, float]:
    # Produce a large undirected chain to ensure the expansion path scales sensibly.
    if node_count < 2:
        raise ValueError("node_count must be at least 2.")

    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(node_count)]
    nodes[0]["supply"] = total_supply
    nodes[-1]["supply"] = -total_supply

    arcs = []
    base_capacity = total_supply + 100.0
    for idx in range(node_count - 1):
        arcs.append(
            {
                "tail": f"n{idx}",
                "head": f"n{idx + 1}",
                "capacity": base_capacity,
                "cost": 2.0 + (idx % 5),
            }
        )

    tolerance = 1e-6
    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=tolerance)
    return problem, total_supply


@pytest.mark.slow
def test_large_undirected_chain_expansion():
    # Verifies undirected graphs still achieve optimality after expansion into directed arcs.
    problem, total_supply = _build_undirected_chain()
    max_iterations = 4000

    result = solve_min_cost_flow(problem, max_iterations=max_iterations)

    assert result.status == "optimal"
    assert result.iterations < max_iterations
    assert len(result.flows) == len(problem.arcs)

    expected_objective = sum((2.0 + (idx % 5)) * total_supply for idx in range(len(problem.arcs)))
    assert math.isclose(result.objective, expected_objective, rel_tol=0.0, abs_tol=1e-6)

    for idx in range(len(problem.arcs)):
        key = (f"n{idx}", f"n{idx + 1}")
        assert key in result.flows
        assert math.isclose(result.flows[key], total_supply, rel_tol=0.0, abs_tol=1e-6)
