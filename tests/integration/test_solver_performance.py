import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import NetworkProblem, build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402


def _build_performance_chain(
    node_count: int = 160,
    total_supply: float = 1200.0,
    seed: int = 24,
) -> tuple[NetworkProblem, float]:
    # Generate a deterministic medium-sized chain with skip edges for performance sanity checks.
    import random

    if node_count < 3:
        raise ValueError("node_count must be at least 3.")

    rng = random.Random(seed)
    nodes = [{"id": f"n{i}", "supply": 0.0} for i in range(node_count)]
    nodes[0]["supply"] = total_supply
    nodes[-1]["supply"] = -total_supply

    arcs = []
    base_capacity = total_supply
    for idx in range(node_count - 1):
        capacity = base_capacity + rng.randint(0, 120)
        cost = 1.0 + (rng.randint(0, 9) / 5.0)
        arcs.append(
            {
                "tail": f"n{idx}",
                "head": f"n{idx + 1}",
                "capacity": float(capacity),
                "cost": cost,
            }
        )
        if idx + 2 < node_count and rng.random() < 0.35:
            capacity_skip = base_capacity + rng.randint(0, 120)
            cost_skip = 1.5 + (rng.randint(0, 9) / 4.0)
            arcs.append(
                {
                    "tail": f"n{idx}",
                    "head": f"n{idx + 2}",
                    "capacity": float(capacity_skip),
                    "cost": cost_skip,
                }
            )

    tolerance = 1e-6
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=tolerance)
    return problem, total_supply


@pytest.mark.slow
def test_medium_network_solves_within_budget():
    # Ensure a sizable directed instance completes within a reasonable iteration budget.
    problem, total_supply = _build_performance_chain()
    max_iterations = 2500

    result = solve_min_cost_flow(problem, max_iterations=max_iterations)

    assert result.status == "optimal"
    assert result.iterations < max_iterations

    sink_id = f"n{len(problem.nodes) - 1}"
    delivered = sum(flow for (tail, head), flow in result.flows.items() if head == sink_id)
    assert math.isclose(delivered, total_supply, rel_tol=0.0, abs_tol=1e-6)

    arc_lookup = {(arc.tail, arc.head): arc for arc in problem.arcs}
    for key, flow in result.flows.items():
        arc = arc_lookup[key]
        assert flow <= arc.capacity + problem.tolerance
        assert flow >= -problem.tolerance
