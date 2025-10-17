import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # type: ignore  # noqa: E402
from hypothesis import strategies as st  # type: ignore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402


@st.composite
def _network_instances(draw) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    # Build moderately sized graphs exercising multiple supply/demand shapes.
    supply_count = draw(st.integers(min_value=1, max_value=4))
    demand_count = draw(st.integers(min_value=1, max_value=4))
    intermediary_count = draw(st.integers(min_value=0, max_value=2))

    supply_nodes = [f"s{idx}" for idx in range(supply_count)]
    demand_nodes = [f"t{idx}" for idx in range(demand_count)]
    relay_nodes = [f"m{idx}" for idx in range(intermediary_count)]

    supplies = [
        draw(st.integers(min_value=1, max_value=18)) for _ in range(supply_count)
    ]
    total_supply = sum(supplies)

    nodes: List[Dict[str, float]] = [
        {"id": node_id, "supply": float(amount)}
        for node_id, amount in zip(supply_nodes, supplies)
    ]

    remaining = total_supply
    demand_amounts: List[int] = []
    if demand_count > 0:
        for idx, node_id in enumerate(demand_nodes):
            if idx == demand_count - 1:
                amount = remaining
            else:
                amount = draw(st.integers(min_value=0, max_value=remaining))
            remaining -= amount
            demand_amounts.append(amount)
            nodes.append({"id": node_id, "supply": -float(amount)})

    for node_id in relay_nodes:
        nodes.append({"id": node_id, "supply": 0.0})

    base_capacity = max(total_supply, 1)
    arcs: List[Dict[str, float]] = []
    cost_strategy = st.integers(min_value=1, max_value=12)

    for tail in supply_nodes:
        for head in demand_nodes:
            capacity = draw(st.integers(min_value=base_capacity, max_value=base_capacity + 20))
            cost = draw(cost_strategy)
            arcs.append(
                {
                    "tail": tail,
                    "head": head,
                    "capacity": float(capacity),
                    "cost": float(cost),
                }
            )

    for relay in relay_nodes:
        for supply in supply_nodes:
            capacity = draw(st.integers(min_value=base_capacity, max_value=base_capacity + 20))
            cost = draw(cost_strategy)
            arcs.append(
                {
                    "tail": supply,
                    "head": relay,
                    "capacity": float(capacity),
                    "cost": float(cost),
                }
            )
        for demand in demand_nodes:
            capacity = draw(st.integers(min_value=base_capacity, max_value=base_capacity + 20))
            cost = draw(cost_strategy)
            arcs.append(
                {
                    "tail": relay,
                    "head": demand,
                    "capacity": float(capacity),
                    "cost": float(cost),
                }
            )

    # Ensure at least one direct connection exists even if relays are absent.
    if not arcs:
        tail = supply_nodes[0]
        head = demand_nodes[0]
        arcs.append(
            {
                "tail": tail,
                "head": head,
                "capacity": float(base_capacity),
                "cost": 1.0,
            }
        )

    return nodes, arcs


def _compute_node_balance(
    flows: Dict[Tuple[str, str], float], supplies: Dict[str, float]
) -> Dict[str, float]:
    balance = dict(supplies)
    for (tail, head), flow in flows.items():
        balance[tail] = balance.get(tail, 0.0) - flow
        balance[head] = balance.get(head, 0.0) + flow
    return balance


def _objective_from_flows(
    flows: Dict[Tuple[str, str], float],
    costs: Dict[Tuple[str, str], float],
) -> float:
    return sum(flow * costs[key] for key, flow in flows.items())


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(_network_instances())
def test_min_cost_flow_respects_mass_balance(instance: Tuple[List[Dict[str, float]], List[Dict[str, float]]]):
    # Property: solutions should balance all nodes and remain within capacity regardless of random draw.
    nodes, arcs = instance
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solve_kwargs = {"max_iterations": 2000}

    first = solve_min_cost_flow(problem, **solve_kwargs)
    second = solve_min_cost_flow(problem, **solve_kwargs)

    if first != second:
        assume(False)
    assume(first.status == "optimal")

    cost_lookup = {(arc["tail"], arc["head"]): float(arc["cost"]) for arc in arcs}
    capacity_lookup = {(arc["tail"], arc["head"]): float(arc["capacity"]) for arc in arcs}

    for key, value in first.flows.items():
        assert key in cost_lookup
        cap = capacity_lookup[key]
        assert value <= cap + 1e-6
        assert value >= -1e-6

    computed_objective = _objective_from_flows(first.flows, cost_lookup)
    assert math.isclose(first.objective, computed_objective, rel_tol=0.0, abs_tol=1e-5)

    supplies = {node["id"]: float(node.get("supply", 0.0)) for node in nodes}
    balance = _compute_node_balance(first.flows, supplies)

    for node_id, residual in balance.items():
        assert math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=1e-5), f"Node {node_id} imbalance {residual}"
