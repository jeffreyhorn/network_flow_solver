import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402

# These integration-style unit tests stress large directed graphs without the CLI overhead.

def test_large_chain_flow_distribution():
    node_count = 120
    total_supply = 750.0
    nodes = [{"id": f"v{i}", "supply": 0.0} for i in range(node_count)]
    nodes[0]["supply"] = total_supply
    nodes[-1]["supply"] = -total_supply

    arcs = [
        {
            "tail": f"v{i}",
            "head": f"v{i + 1}",
            "capacity": total_supply,
            "cost": 1 + (i % 9),
        }
        for i in range(node_count - 1)
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-4)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    assert len(result.flows) == node_count - 1

    for i in range(node_count - 1):
        edge = (f"v{i}", f"v{i + 1}")
        assert math.isclose(result.flows[edge], total_supply, rel_tol=0.0, abs_tol=1e-8)

    expected_objective = sum(arc["cost"] * total_supply for arc in arcs)
    assert math.isclose(result.objective, expected_objective, rel_tol=0.0, abs_tol=1e-6)


def test_large_chain_is_deterministic():
    node_count = 80
    total_supply = 500.0
    nodes = [{"id": f"p{i}", "supply": 0.0} for i in range(node_count)]
    nodes[0]["supply"] = total_supply
    nodes[-1]["supply"] = -total_supply

    arcs = [
        {
            "tail": f"p{i}",
            "head": f"p{i + 1}",
            "capacity": total_supply,
            "cost": 2 + (i % 5),
        }
        for i in range(node_count - 1)
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-4)
    first = solve_min_cost_flow(problem)
    second = solve_min_cost_flow(problem)

    assert first.status == "optimal"
    assert second.status == "optimal"
    assert math.isclose(first.objective, second.objective, rel_tol=0.0, abs_tol=1e-6)
    assert first.flows == second.flows


def test_multi_source_multi_sink_balancing():
    nodes = [
        {"id": "s1", "supply": 10.0},
        {"id": "s2", "supply": 5.0},
        {"id": "hub", "supply": 0.0},
        {"id": "t1", "supply": -6.0},
        {"id": "t2", "supply": -9.0},
    ]
    arcs = [
        {"tail": "s1", "head": "hub", "capacity": 10.0, "cost": 1.0},
        {"tail": "s2", "head": "hub", "capacity": 5.0, "cost": 1.0},
        {"tail": "hub", "head": "t1", "capacity": 10.0, "cost": 1.0},
        {"tail": "hub", "head": "t2", "capacity": 10.0, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-4)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    expected_flows = {
        ("s1", "hub"): 10.0,
        ("s2", "hub"): 5.0,
        ("hub", "t1"): 6.0,
        ("hub", "t2"): 9.0,
    }
    assert result.flows == expected_flows
    assert math.isclose(result.objective, 30.0, rel_tol=0.0, abs_tol=1e-6)


def test_large_undirected_chain_expansion():
    node_count = 75
    total_supply = 320.0
    tolerance = 1e-4

    nodes = [{"id": f"u{i}", "supply": 0.0} for i in range(node_count)]
    nodes[0]["supply"] = total_supply
    nodes[-1]["supply"] = -total_supply

    edges = [
        {
            "tail": f"u{i}",
            "head": f"u{i + 1}",
            "capacity": total_supply,
            "cost": 3 + (i % 4),
        }
        for i in range(node_count - 1)
    ]

    problem = build_problem(nodes=nodes, arcs=edges, directed=False, tolerance=tolerance)
    result = solve_min_cost_flow(problem)

    assert result.status == "optimal"
    assert len(result.flows) == node_count - 1
    expected_objective = sum(edge["cost"] * total_supply for edge in edges)
    assert math.isclose(result.objective, expected_objective, rel_tol=0.0, abs_tol=1e-5)
    for key, flow in result.flows.items():
        assert math.isclose(flow, total_supply, rel_tol=0.0, abs_tol=1e-8)
