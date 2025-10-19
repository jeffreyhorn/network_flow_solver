"""Example demonstrating incremental resolving with problem modifications.

This example shows how to efficiently re-solve network flow problems after:
1. Capacity changes (infrastructure expansion)
2. Cost changes (pricing updates)
3. Supply/demand changes (demand fluctuations)
4. Adding/removing arcs (network topology changes)

The solver supports warm-starting (see warm_start_example.py and docs), which can
dramatically reduce iterations for similar problems. This example focuses on
demonstrating various problem modification scenarios. Incremental resolving is
valuable for:
- Scenario analysis (what-if modeling)
- Iterative optimization (gradually improving solutions)
- Sensitivity analysis validation
- Real-time adaptation to changing conditions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import build_problem, solve_min_cost_flow  # noqa: E402


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def print_solution(result, label: str = "Solution") -> None:
    """Print solution summary."""
    print(f"\n{label}:")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:,.2f}")
    print(f"  Iterations: {result.iterations}")
    if result.flows:
        print(f"  Flows:")
        for (tail, head), flow in sorted(result.flows.items()):
            print(f"    {tail} -> {head}: {flow:.2f} units")


def scenario_1_capacity_expansion() -> None:
    """Scenario 1: Incremental capacity expansion analysis."""
    print_section_header("SCENARIO 1: CAPACITY EXPANSION")

    print("\nUse case: Transportation network needs more capacity")
    print("  Question: How much does cost decrease as we expand capacity?")

    # Base problem: Limited capacity causing high costs
    nodes = [
        {"id": "warehouse", "supply": 100.0},
        {"id": "store_a", "supply": -60.0},
        {"id": "store_b", "supply": -40.0},
    ]

    base_arcs = [
        {"tail": "warehouse", "head": "store_a", "capacity": 50.0, "cost": 2.0},
        {"tail": "warehouse", "head": "store_b", "capacity": 50.0, "cost": 3.0},
    ]

    print_subsection("Base Problem: Limited Capacity")
    problem_base = build_problem(nodes=nodes, arcs=base_arcs, directed=True, tolerance=1e-6)
    result_base = solve_min_cost_flow(problem_base)
    print_solution(result_base, "Initial Solution")

    # Incremental capacity increases
    capacities = [50, 60, 70, 80, 100]
    print_subsection("Incremental Capacity Expansion")
    print("\nExpanding warehouse -> store_a capacity:")
    print(f"{'Capacity':<12} {'Objective':<15} {'Iterations':<12} {'Improvement'}")
    print("-" * 70)

    prev_objective = result_base.objective
    for cap in capacities:
        modified_arcs = [
            {"tail": "warehouse", "head": "store_a", "capacity": float(cap), "cost": 2.0},
            {"tail": "warehouse", "head": "store_b", "capacity": 50.0, "cost": 3.0},
        ]
        problem = build_problem(nodes=nodes, arcs=modified_arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)

        improvement = prev_objective - result.objective
        print(f"{cap:<12} ${result.objective:<14.2f} {result.iterations:<12} ${improvement:>6.2f}")
        prev_objective = result.objective

    print("\nInsight: Marginal benefit decreases as capacity increases")
    print("         (diminishing returns on capacity expansion)")


def scenario_2_cost_changes() -> None:
    """Scenario 2: Re-solving with updated costs (pricing changes)."""
    print_section_header("SCENARIO 2: COST UPDATES")

    print("\nUse case: Fuel prices change, need to update shipping costs")
    print("  Question: How does the optimal solution change?")

    nodes = [
        {"id": "factory", "supply": 100.0},
        {"id": "hub", "supply": 0.0},
        {"id": "customer", "supply": -100.0},
    ]

    # Route 1: Direct (expensive but fast)
    # Route 2: Via hub (cheaper but slower)

    print_subsection("Original Costs")
    arcs_original = [
        {"tail": "factory", "head": "customer", "capacity": 50.0, "cost": 10.0},  # Direct
        {"tail": "factory", "head": "hub", "capacity": 100.0, "cost": 3.0},  # To hub
        {"tail": "hub", "head": "customer", "capacity": 100.0, "cost": 4.0},  # From hub
    ]

    problem_original = build_problem(nodes=nodes, arcs=arcs_original, directed=True, tolerance=1e-6)
    result_original = solve_min_cost_flow(problem_original)
    print_solution(result_original, "Original Solution")

    print_subsection("After Fuel Price Increase (+50%)")
    # Fuel price increase affects direct route more
    arcs_increased = [
        {"tail": "factory", "head": "customer", "capacity": 50.0, "cost": 15.0},  # Direct +50%
        {"tail": "factory", "head": "hub", "capacity": 100.0, "cost": 4.0},  # To hub +33%
        {"tail": "hub", "head": "customer", "capacity": 100.0, "cost": 5.0},  # From hub +25%
    ]

    problem_increased = build_problem(
        nodes=nodes, arcs=arcs_increased, directed=True, tolerance=1e-6
    )
    result_increased = solve_min_cost_flow(problem_increased)
    print_solution(result_increased, "After Cost Increase")

    cost_increase = result_increased.objective - result_original.objective
    pct_increase = (cost_increase / result_original.objective) * 100
    print(f"\nCost Impact:")
    print(f"  Absolute increase: ${cost_increase:.2f}")
    print(f"  Percentage increase: {pct_increase:.1f}%")

    # Compare flow patterns
    print(f"\nFlow Pattern Changes:")
    direct_flow_before = result_original.flows.get(("factory", "customer"), 0.0)
    direct_flow_after = result_increased.flows.get(("factory", "customer"), 0.0)
    print(f"  Direct route flow: {direct_flow_before:.2f} -> {direct_flow_after:.2f} units")

    if direct_flow_after < direct_flow_before:
        print(f"  ✓ Solver shifted to cheaper hub route after price increase")


def scenario_3_demand_fluctuations() -> None:
    """Scenario 3: Re-solving with changing demand patterns."""
    print_section_header("SCENARIO 3: DEMAND FLUCTUATIONS")

    print("\nUse case: Daily demand varies, need to adjust shipments")
    print("  Question: How to efficiently handle demand changes?")

    base_nodes = [
        {"id": "supplier", "supply": 100.0},
        {"id": "customer_a", "supply": -60.0},
        {"id": "customer_b", "supply": -40.0},
    ]

    arcs = [
        {"tail": "supplier", "head": "customer_a", "capacity": 100.0, "cost": 2.0},
        {"tail": "supplier", "head": "customer_b", "capacity": 100.0, "cost": 3.0},
    ]

    print_subsection("Week 1: Base Demand")
    problem_week1 = build_problem(nodes=base_nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result_week1 = solve_min_cost_flow(problem_week1)
    print_solution(result_week1, "Week 1")

    # Week 2: Demand shift
    print_subsection("Week 2: Demand Shift (Customer A +20%, Customer B -20%)")
    nodes_week2 = [
        {"id": "supplier", "supply": 100.0},
        {"id": "customer_a", "supply": -72.0},  # +20%
        {"id": "customer_b", "supply": -28.0},  # -30% to maintain balance
    ]

    problem_week2 = build_problem(nodes=nodes_week2, arcs=arcs, directed=True, tolerance=1e-6)
    result_week2 = solve_min_cost_flow(problem_week2)
    print_solution(result_week2, "Week 2")

    # Week 3: Demand surge
    print_subsection("Week 3: Overall Demand Surge (+20%)")
    nodes_week3 = [
        {"id": "supplier", "supply": 120.0},  # +20%
        {"id": "customer_a", "supply": -72.0},  # +20%
        {"id": "customer_b", "supply": -48.0},  # +20%
    ]

    problem_week3 = build_problem(nodes=nodes_week3, arcs=arcs, directed=True, tolerance=1e-6)
    result_week3 = solve_min_cost_flow(problem_week3)
    print_solution(result_week3, "Week 3")

    print(f"\nCost Progression:")
    print(f"  Week 1: ${result_week1.objective:,.2f}")
    print(
        f"  Week 2: ${result_week2.objective:,.2f} ({result_week2.objective - result_week1.objective:+.2f})"
    )
    print(
        f"  Week 3: ${result_week3.objective:,.2f} ({result_week3.objective - result_week1.objective:+.2f})"
    )


def scenario_4_network_topology_changes() -> None:
    """Scenario 4: Adding/removing arcs (network topology changes)."""
    print_section_header("SCENARIO 4: NETWORK TOPOLOGY CHANGES")

    print("\nUse case: Opening new distribution routes or closing old ones")
    print("  Question: What's the value of adding a new route?")

    nodes = [
        {"id": "plant", "supply": 100.0},
        {"id": "dist_center", "supply": 0.0},
        {"id": "market", "supply": -100.0},
    ]

    print_subsection("Current Network: Single Route via Distribution Center")
    arcs_current = [
        {"tail": "plant", "head": "dist_center", "capacity": 100.0, "cost": 5.0},
        {"tail": "dist_center", "head": "market", "capacity": 100.0, "cost": 4.0},
    ]

    problem_current = build_problem(nodes=nodes, arcs=arcs_current, directed=True, tolerance=1e-6)
    result_current = solve_min_cost_flow(problem_current)
    print_solution(result_current, "Current Network")

    print_subsection("Proposed: Add Direct Route (Plant -> Market)")
    print("  Cost: $8/unit (versus $9/unit via distribution center)")

    arcs_with_direct = [
        {"tail": "plant", "head": "dist_center", "capacity": 100.0, "cost": 5.0},
        {"tail": "dist_center", "head": "market", "capacity": 100.0, "cost": 4.0},
        {"tail": "plant", "head": "market", "capacity": 60.0, "cost": 8.0},  # New route
    ]

    problem_with_direct = build_problem(
        nodes=nodes, arcs=arcs_with_direct, directed=True, tolerance=1e-6
    )
    result_with_direct = solve_min_cost_flow(problem_with_direct)
    print_solution(result_with_direct, "Network with Direct Route")

    savings = result_current.objective - result_with_direct.objective
    print(f"\nValue of Direct Route:")
    print(f"  Annual savings: ${savings:,.2f} per period")
    if savings > 0:
        print(f"  ✓ Direct route is cost-effective")
    else:
        print(f"  ✗ Direct route doesn't reduce costs (existing routes are better)")


def scenario_5_iterative_optimization() -> None:
    """Scenario 5: Iterative optimization with incremental improvements."""
    print_section_header("SCENARIO 5: ITERATIVE OPTIMIZATION")

    print("\nUse case: Gradually improve network by adding capacity where needed")
    print("  Strategy: Identify bottlenecks, expand, re-solve, repeat")

    nodes = [
        {"id": "source", "supply": 150.0},
        {"id": "node_a", "supply": 0.0},
        {"id": "node_b", "supply": 0.0},
        {"id": "sink", "supply": -150.0},
    ]

    # Initial network with bottlenecks
    iteration = 0
    arcs = [
        {"tail": "source", "head": "node_a", "capacity": 60.0, "cost": 1.0},
        {"tail": "source", "head": "node_b", "capacity": 60.0, "cost": 2.0},
        {"tail": "node_a", "head": "sink", "capacity": 50.0, "cost": 2.0},  # Bottleneck
        {"tail": "node_b", "head": "sink", "capacity": 50.0, "cost": 1.0},  # Bottleneck
    ]

    print_subsection(f"Iteration {iteration}: Initial Network")
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)
    print_solution(result)

    # Identify bottleneck and expand
    print("\nBottleneck Analysis:")
    for (tail, head), flow in sorted(result.flows.items()):
        capacity = next(a["capacity"] for a in arcs if a["tail"] == tail and a["head"] == head)
        utilization = (flow / capacity) * 100 if capacity > 0 else 0
        if utilization > 95:
            print(f"  ⚠ {tail} -> {head}: {utilization:.1f}% utilized (BOTTLENECK)")

    # Iteration 1: Expand bottleneck arcs
    iteration += 1
    print_subsection(f"Iteration {iteration}: Expand Bottleneck Arcs (+30 units)")
    arcs = [
        {"tail": "source", "head": "node_a", "capacity": 60.0, "cost": 1.0},
        {"tail": "source", "head": "node_b", "capacity": 60.0, "cost": 2.0},
        {"tail": "node_a", "head": "sink", "capacity": 80.0, "cost": 2.0},  # Expanded
        {"tail": "node_b", "head": "sink", "capacity": 80.0, "cost": 1.0},  # Expanded
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result_iter1 = solve_min_cost_flow(problem)
    print_solution(result_iter1)

    improvement = result.objective - result_iter1.objective
    print(f"\nImprovement: ${improvement:.2f} cost reduction")

    # Iteration 2: Check for new bottlenecks
    iteration += 1
    print_subsection(f"Iteration {iteration}: Check for New Bottlenecks")

    print("\nBottleneck Analysis:")
    for (tail, head), flow in sorted(result_iter1.flows.items()):
        capacity = next(a["capacity"] for a in arcs if a["tail"] == tail and a["head"] == head)
        utilization = (flow / capacity) * 100 if capacity > 0 else 0
        if utilization > 95:
            print(f"  ⚠ {tail} -> {head}: {utilization:.1f}% utilized (NEW BOTTLENECK)")
        else:
            print(f"  ✓ {tail} -> {head}: {utilization:.1f}% utilized")

    print("\nIterative Optimization Strategy:")
    print("  1. Solve current network")
    print("  2. Identify bottlenecks (high utilization arcs)")
    print("  3. Expand bottleneck capacity")
    print("  4. Re-solve and measure improvement")
    print("  5. Repeat until cost/benefit threshold met")


def main() -> None:
    """Run all incremental resolving scenarios."""
    print_section_header("INCREMENTAL RESOLVING EXAMPLES")
    print("\nDemonstrating efficient re-solving with problem modifications")

    start_time = time.time()

    scenario_1_capacity_expansion()
    scenario_2_cost_changes()
    scenario_3_demand_fluctuations()
    scenario_4_network_topology_changes()
    scenario_5_iterative_optimization()

    elapsed = time.time() - start_time

    print_section_header("SUMMARY")
    print("\nKey Takeaways:")
    print("  1. Incremental resolving enables scenario analysis")
    print("  2. Small problem modifications typically solve quickly")
    print("  3. Useful for:")
    print("     - Capacity planning (what-if analysis)")
    print("     - Cost sensitivity analysis")
    print("     - Demand forecasting and adaptation")
    print("     - Network design decisions")
    print("     - Iterative optimization strategies")
    print("\n  Note: The solver supports warm-starting for even faster re-solving.")
    print("        See warm_start_example.py for demonstrations.")

    print(f"\nTotal time for all scenarios: {elapsed:.2f}s")
    print("\n" + "=" * 70)
    print("For more details, see docs/examples.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
