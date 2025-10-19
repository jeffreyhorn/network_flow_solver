"""Example demonstrating undirected graph handling in the network flow solver.

This example shows how to work with undirected graphs, understand the bidirectional
flow transformation, and interpret results correctly.
"""

from network_solver import solve_min_cost_flow
from network_solver.data import Arc, NetworkProblem, Node


def main() -> None:
    print("=" * 80)
    print("Undirected Graph Example: Campus Network Design")
    print("=" * 80)
    print()
    print("Problem: Design a data network connecting 4 campus buildings with")
    print("bidirectional fiber-optic cables. Find minimum cost routing of 100 Gbps")
    print("from the data center (Building A) to the research lab (Building D).")
    print()

    # Network topology:
    #     A (source: 100)
    #    / \
    #   B   C
    #    \ /
    #     D (demand: 100)
    #
    # Each link has symmetric capacity and cost (cable cost in thousands)

    nodes = {
        "A": Node(id="A", supply=100.0),  # Data center (source)
        "B": Node(id="B", supply=0.0),  # Building B (intermediate)
        "C": Node(id="C", supply=0.0),  # Building C (intermediate)
        "D": Node(id="D", supply=-100.0),  # Research lab (sink)
    }

    # Undirected edges: {A,B}, {A,C}, {B,D}, {C,D}
    # Note: For undirected graphs, edges are bidirectional with symmetric capacity/cost
    arcs = [
        Arc(tail="A", head="B", capacity=80.0, cost=10.0),  # A-B link: 80 Gbps, $10k
        Arc(tail="A", head="C", capacity=60.0, cost=8.0),  # A-C link: 60 Gbps, $8k
        Arc(tail="B", head="D", capacity=70.0, cost=12.0),  # B-D link: 70 Gbps, $12k
        Arc(tail="C", head="D", capacity=90.0, cost=15.0),  # C-D link: 90 Gbps, $15k
    ]

    problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

    print("Graph Structure (Undirected):")
    print("-" * 80)
    for arc in arcs:
        print(
            f"  {arc.tail} -- {arc.head}:  capacity={arc.capacity:5.0f} Gbps,  cost=${arc.cost:.0f}k"
        )
    print()

    # Show how undirected edges are transformed internally
    print("Internal Transformation (Undirected → Directed):")
    print("-" * 80)
    print("Each undirected edge {u, v} with capacity C becomes a directed arc (u, v) with:")
    print("  • Capacity: C (upper bound on flow)")
    print("  • Lower bound: -C (allows reverse flow)")
    print("  • Positive flow = tail→head, Negative flow = head→tail")
    print()

    expanded = problem.undirected_expansion()
    for arc in expanded:
        print(
            f"  {arc.tail} → {arc.head}:  lower={arc.lower:5.0f},  capacity={arc.capacity:5.0f},  cost={arc.cost:.0f}"
        )
    print()

    # Solve the problem
    result = solve_min_cost_flow(problem)

    print("Solution:")
    print("-" * 80)
    print(f"Status: {result.status}")
    print(f"Total Cost: ${result.objective:.2f}k")
    print(f"Iterations: {result.iterations}")
    print()

    # Interpret flows on undirected edges
    print("Flow on Each Link (Undirected Interpretation):")
    print("-" * 80)
    print("  Link        Flow (Gbps)  Direction         Cost Contribution")
    print("  " + "-" * 72)

    for arc in arcs:
        key = (arc.tail, arc.head)
        flow = result.flows.get(key, 0.0)

        if flow > 0:
            direction = f"{arc.tail} → {arc.head}"
        elif flow < 0:
            direction = f"{arc.head} → {arc.tail}"
        else:
            direction = "no flow"

        abs_flow = abs(flow)
        cost_contribution = abs_flow * arc.cost

        print(
            f"  {arc.tail}--{arc.head}      {abs_flow:6.1f}      {direction:12s}  ${cost_contribution:8.2f}k"
        )

    print()

    # Verify flow conservation
    print("Flow Conservation Check:")
    print("-" * 80)

    for node_id, node in nodes.items():
        inflow = sum(result.flows.get((t, node_id), 0.0) for t in nodes if t != node_id) + sum(
            -result.flows.get((node_id, h), 0.0)
            for h in nodes
            if h != node_id and result.flows.get((node_id, h), 0.0) < 0
        )

        outflow = sum(result.flows.get((node_id, h), 0.0) for h in nodes if h != node_id) + sum(
            -result.flows.get((t, node_id), 0.0)
            for t in nodes
            if t != node_id and result.flows.get((t, node_id), 0.0) < 0
        )

        balance = inflow - outflow
        expected = node.supply
        status = "✓" if abs(balance - expected) < 1e-6 else "✗"

        print(
            f"  Node {node_id}: supply={expected:6.1f}, inflow={inflow:6.1f}, "
            f"outflow={outflow:6.1f}, balance={balance:6.1f} {status}"
        )

    print()

    # Identify which path was used
    print("Optimal Routing:")
    print("-" * 80)

    flow_ab = result.flows.get(("A", "B"), 0.0)
    flow_ac = result.flows.get(("A", "C"), 0.0)
    flow_bd = result.flows.get(("B", "D"), 0.0)
    flow_cd = result.flows.get(("C", "D"), 0.0)

    if flow_ab > 0 and flow_bd > 0:
        print(f"  Route 1: A → B → D carries {min(flow_ab, flow_bd):.1f} Gbps")
        print(f"           Cost: {min(flow_ab, flow_bd) * (10.0 + 12.0):.2f}k")

    if flow_ac > 0 and flow_cd > 0:
        print(f"  Route 2: A → C → D carries {min(flow_ac, flow_cd):.1f} Gbps")
        print(f"           Cost: {min(flow_ac, flow_cd) * (8.0 + 15.0):.2f}k")

    if abs(flow_ab) < 1e-6 and abs(flow_bd) < 1e-6:
        print("  Route A → B → D: not used")

    if abs(flow_ac) < 1e-6 and abs(flow_cd) < 1e-6:
        print("  Route A → C → D: not used")

    print()

    # Key insights about undirected graphs
    print("Key Insights for Undirected Graphs:")
    print("-" * 80)
    print("✓ Undirected edges must have finite capacity (no infinite capacity)")
    print("✓ Cannot specify custom lower bounds (automatically set to -capacity)")
    print("✓ Positive flow = tail→head direction, Negative flow = head→tail")
    print("✓ Flow magnitude |f| shows amount, sign shows direction")
    print("✓ Costs are symmetric (same cost regardless of direction)")
    print("✓ In results, check flow sign to determine actual direction used")
    print()

    # Compare with directed alternative
    print("Comparison: Undirected vs Directed Representation:")
    print("-" * 80)
    print("Undirected (what we used):")
    print("  • 1 edge {A,B} with capacity 80")
    print("  • Transformed internally to 1 arc (A,B) with bounds [-80, 80]")
    print("  • Result: single flow value (positive or negative)")
    print()
    print("Directed alternative (manual):")
    print("  • 2 arcs: (A,B) and (B,A), each with capacity 80")
    print("  • Result: two separate flow values (one per arc)")
    print("  • More verbose but sometimes clearer")
    print()

    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
