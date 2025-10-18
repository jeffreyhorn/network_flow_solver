"""
Demonstrates utility functions for analyzing network flow solutions.

This example shows how to:
- Extract flow paths from source to target
- Validate flow solutions for correctness
- Identify bottleneck arcs limiting capacity
"""

from network_solver import (
    build_problem,
    compute_bottleneck_arcs,
    extract_path,
    solve_min_cost_flow,
    validate_flow,
)


def main():
    """Demonstrate utility functions on a supply chain network."""

    # Build a simple transportation problem
    # 2 factories -> 3 warehouses
    nodes = [
        # Factories (sources)
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 150.0},
        # Warehouses (sinks)
        {"id": "warehouse_1", "supply": -80.0},
        {"id": "warehouse_2", "supply": -120.0},
        {"id": "warehouse_3", "supply": -50.0},
    ]

    arcs = [
        # From factory A
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 100.0, "cost": 2.5},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "factory_a", "head": "warehouse_3", "capacity": 100.0, "cost": 1.5},
        # From factory B
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 150.0, "cost": 1.8},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 150.0, "cost": 2.2},
        {"tail": "factory_b", "head": "warehouse_3", "capacity": 150.0, "cost": 2.8},
    ]

    print("=" * 80)
    print("NETWORK FLOW UTILITY FUNCTIONS DEMONSTRATION")
    print("=" * 80)
    print()

    # Build and solve the problem
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    print("Solving supply chain network...")
    result = solve_min_cost_flow(problem)

    print(f"Status: {result.status}")
    print(f"Total cost: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()

    # ========================================================================
    # 1. VALIDATE FLOW SOLUTION
    # ========================================================================
    print("=" * 80)
    print("1. FLOW VALIDATION")
    print("=" * 80)
    print()

    validation = validate_flow(problem, result)

    if validation.is_valid:
        print("✓ Solution is VALID - all constraints satisfied")
        print()
        print("Flow balance at each node:")
        for node_id in sorted(validation.flow_balance.keys()):
            balance = validation.flow_balance[node_id]
            if abs(balance) < 1e-9:
                balance = 0.0
            print(f"  {node_id:20s}: {balance:+10.6f} (should be ~0)")
    else:
        print("✗ Solution is INVALID")
        print()
        print("Validation errors:")
        for error in validation.errors:
            print(f"  - {error}")
        if validation.capacity_violations:
            print()
            print("Capacity violations:")
            for tail, head in validation.capacity_violations:
                print(f"  - Arc ({tail}, {head})")
        if validation.lower_bound_violations:
            print()
            print("Lower bound violations:")
            for tail, head in validation.lower_bound_violations:
                print(f"  - Arc ({tail}, {head})")

    print()

    # ========================================================================
    # 2. EXTRACT FLOW PATHS
    # ========================================================================
    print("=" * 80)
    print("2. FLOW PATH EXTRACTION")
    print("=" * 80)
    print()

    # Find paths from factories to warehouses
    factories = ["factory_a", "factory_b"]
    warehouses = ["warehouse_1", "warehouse_2", "warehouse_3"]

    print("Sample flow paths from factories to warehouses:")
    print()

    for factory in factories:
        for warehouse in warehouses:
            path = extract_path(result, problem, factory, warehouse)
            if path is not None:
                print(f"Path from {factory} to {warehouse}:")
                print(f"  Route: {' -> '.join(path.nodes)}")
                print(f"  Flow: {path.flow:.2f} units")
                print(f"  Cost: ${path.cost:.2f}")
                print()

    # Demonstrate path not found
    path = extract_path(result, problem, "warehouse_1", "factory_a")
    if path is None:
        print("Path from warehouse_1 to factory_a: NOT FOUND (flow goes opposite direction)")
    print()

    # ========================================================================
    # 3. IDENTIFY BOTTLENECK ARCS
    # ========================================================================
    print("=" * 80)
    print("3. BOTTLENECK ANALYSIS")
    print("=" * 80)
    print()

    print("Identifying arcs at or near capacity (>= 90% utilization)...")
    print()

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.90)

    if bottlenecks:
        print(f"Found {len(bottlenecks)} bottleneck arc(s):")
        print()
        print(f"{'Arc':<40} {'Flow':>10} {'Capacity':>10} {'Util%':>8} {'Slack':>10} {'Cost':>8}")
        print("-" * 90)
        for bottleneck in bottlenecks:
            arc_name = f"({bottleneck.tail} -> {bottleneck.head})"
            util_pct = bottleneck.utilization * 100
            print(
                f"{arc_name:<40} "
                f"{bottleneck.flow:>10.2f} "
                f"{bottleneck.capacity:>10.2f} "
                f"{util_pct:>7.1f}% "
                f"{bottleneck.slack:>10.2f} "
                f"${bottleneck.cost:>7.2f}"
            )
        print()
        print("Bottleneck insights:")
        print("  - These arcs are limiting network throughput")
        print("  - Increasing their capacity would allow more flow")
        print("  - High-cost bottlenecks are priorities for capacity expansion")
        print()

        # Calculate potential benefit
        most_constrained = bottlenecks[0]
        print(f"Most constrained arc: ({most_constrained.tail} -> {most_constrained.head})")
        print(f"  Current capacity: {most_constrained.capacity:.2f} units")
        print(f"  Current flow: {most_constrained.flow:.2f} units")
        print(f"  Utilization: {most_constrained.utilization * 100:.1f}%")
        print(f"  Slack: {most_constrained.slack:.2f} units")
        print()
        if most_constrained.slack < 0.1:
            print("  ⚠ This arc is completely saturated!")
            print("  Recommendation: Increase capacity to enable additional flow")
    else:
        print("No bottlenecks found - all arcs have spare capacity")

    print()

    # ========================================================================
    # 4. DETAILED SOLUTION ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("4. DETAILED SOLUTION ANALYSIS")
    print("=" * 80)
    print()

    # Show all flows
    print("Active flows (sorted by magnitude):")
    print()
    sorted_flows = sorted(result.flows.items(), key=lambda x: -x[1])
    print(f"{'Arc':<50} {'Flow':>12} {'% of total':>12}")
    print("-" * 75)

    total_flow = sum(result.flows.values())
    for (tail, head), flow in sorted_flows[:15]:  # Show top 15
        arc_name = f"{tail} -> {head}"
        pct = (flow / total_flow * 100) if total_flow > 0 else 0
        print(f"{arc_name:<50} {flow:>12.2f} {pct:>11.1f}%")

    if len(sorted_flows) > 15:
        print(f"... and {len(sorted_flows) - 15} more arcs")

    print()

    # Factory and warehouse statistics
    print("Factory shipments:")
    print()
    for factory in factories:
        outflow = sum(flow for (tail, head), flow in result.flows.items() if tail == factory)
        supply = problem.nodes[factory].supply
        print(f"  {factory:15s}: {outflow:8.2f} shipped / {supply:8.2f} available")

    print()
    print("Warehouse receipts:")
    print()
    for warehouse in warehouses:
        inflow = sum(flow for (tail, head), flow in result.flows.items() if head == warehouse)
        demand = -problem.nodes[warehouse].supply
        print(f"  {warehouse:15s}: {inflow:8.2f} received / {demand:8.2f} needed")

    print()

    # ========================================================================
    # 5. SENSITIVITY ANALYSIS WITH BOTTLENECKS
    # ========================================================================
    print("=" * 80)
    print("5. SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()

    if bottlenecks:
        print("Impact of increasing bottleneck capacity:")
        print()

        for i, bottleneck in enumerate(bottlenecks[:3], 1):  # Analyze top 3
            # Use dual values to estimate marginal cost reduction
            tail_dual = result.duals.get(bottleneck.tail, 0.0)
            head_dual = result.duals.get(bottleneck.head, 0.0)
            reduced_cost = bottleneck.cost + tail_dual - head_dual

            print(f"{i}. Arc ({bottleneck.tail} -> {bottleneck.head}):")
            print(f"   Current capacity: {bottleneck.capacity:.2f}")
            print(f"   Current utilization: {bottleneck.utilization * 100:.1f}%")
            print(f"   Arc cost: ${bottleneck.cost:.2f}/unit")
            print(f"   Reduced cost: ${reduced_cost:.6f}")
            if abs(reduced_cost) < 1e-3:
                print("   → Increasing capacity may reduce total cost")
            print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
