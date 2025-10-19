"""Example demonstrating sensitivity analysis using dual values (shadow prices).

This example shows how to use dual values for:
1. Understanding marginal costs of supply/demand changes
2. Verifying complementary slackness (optimality condition)
3. Predicting cost changes without re-solving
4. Identifying binding capacity constraints
5. Production planning decisions
"""

from __future__ import annotations

import sys
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


def main() -> None:
    """Demonstrate how dual values enable sensitivity analysis."""

    print_section_header("DUAL VALUES & SENSITIVITY ANALYSIS")

    # Original problem: Ship goods from supplier to customer
    nodes = [
        {"id": "supplier", "supply": 100.0},
        {"id": "warehouse", "supply": 0.0},
        {"id": "customer", "supply": -100.0},
    ]
    arcs = [
        {"tail": "supplier", "head": "warehouse", "capacity": 150.0, "cost": 2.0},
        {"tail": "warehouse", "head": "customer", "capacity": 150.0, "cost": 3.0},
    ]

    print("\nOriginal Problem:")
    print("  Supply at 'supplier': 100 units")
    print("  Demand at 'customer': 100 units")
    print("  Arc costs: supplier->warehouse = $2, warehouse->customer = $3")

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print("\nOptimal Solution:")
    print(f"  Status: {result.status}")
    print(f"  Total cost: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")

    print("\nFlows:")
    for (tail, head), flow in sorted(result.flows.items()):
        print(f"  {tail} -> {head}: {flow:.2f} units")

    print("\nDual Values (Shadow Prices):")
    for node_id, dual in sorted(result.duals.items()):
        print(f"  {node_id}: {dual:.6f}")

    # Interpret dual values
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    print("\nWhat do dual values tell us?")
    print("  - Dual values represent the marginal value of resources at each node")
    print("  - They indicate how much the objective would change if we increase")
    print("    supply/demand at that node by one unit")

    # Verify by actually changing supply
    print("\n" + "-" * 70)
    print("Experiment: Increase supply by 10 units")
    print("-" * 70)

    nodes_modified = [
        {"id": "supplier", "supply": 110.0},
        {"id": "warehouse", "supply": 0.0},
        {"id": "customer", "supply": -110.0},
    ]

    problem2 = build_problem(nodes=nodes_modified, arcs=arcs, directed=True, tolerance=1e-6)
    result2 = solve_min_cost_flow(problem2)

    cost_change = result2.objective - result.objective
    expected_change_per_unit = result.duals["supplier"] - result.duals["customer"]

    print(f"\nNew total cost: ${result2.objective:.2f}")
    print(f"Actual cost change: ${cost_change:.2f}")
    print(f"Change per unit: ${cost_change / 10:.2f}")
    print(f"\nExpected from duals: ${expected_change_per_unit:.6f} per unit")
    print(
        f"  (dual[supplier] - dual[customer] = {result.duals['supplier']:.6f} - {result.duals['customer']:.6f})"
    )

    # Check complementary slackness
    print("\n" + "-" * 70)
    print("Complementary Slackness Verification")
    print("-" * 70)

    print("\nFor each arc with positive flow, reduced cost should be zero:")
    print("  Reduced cost = cost + dual[tail] - dual[head]")

    for (tail, head), flow in sorted(result.flows.items()):
        if flow > 1e-6:
            cost = next(arc["cost"] for arc in arcs if arc["tail"] == tail and arc["head"] == head)
            reduced_cost = cost + result.duals[tail] - result.duals[head]
            print(f"\n  Arc {tail} -> {head}:")
            print(f"    Flow: {flow:.2f}")
            print(f"    Cost: ${cost:.2f}")
            print(f"    Reduced cost: {reduced_cost:.10f} (should be ≈ 0)")

    # Additional use case: Multi-factory production planning
    print_section_header("USE CASE: PRODUCTION PLANNING")

    print("\nScenario: Two factories can produce goods to meet customer demand")
    print("  Factory A: Lower variable cost ($3/unit) but limited capacity")
    print("  Factory B: Higher variable cost ($5/unit) but unlimited capacity")
    print("  Question: Which factory should we expand?")

    nodes_production = [
        {"id": "factory_a", "supply": 50.0},  # Limited production
        {"id": "factory_b", "supply": 50.0},  # Supplemental production
        {"id": "customer", "supply": -100.0},  # Total demand
    ]

    arcs_production = [
        {"tail": "factory_a", "head": "customer", "capacity": 60.0, "cost": 3.0},
        {"tail": "factory_b", "head": "customer", "capacity": 150.0, "cost": 5.0},
    ]

    problem_prod = build_problem(
        nodes=nodes_production, arcs=arcs_production, directed=True, tolerance=1e-6
    )
    result_prod = solve_min_cost_flow(problem_prod)

    print("\nOptimal Solution:")
    print(f"  Total cost: ${result_prod.objective:.2f}")
    for (tail, head), flow in sorted(result_prod.flows.items()):
        cost = next(
            arc["cost"] for arc in arcs_production if arc["tail"] == tail and arc["head"] == head
        )
        print(f"  {tail} -> {head}: {flow:.2f} units @ ${cost}/unit")

    print("\nDual Values (Shadow Prices):")
    for node_id, dual in sorted(result_prod.duals.items()):
        print(f"  {node_id}: ${dual:.6f}")

    print("\nDecision Analysis:")
    factory_a_value = -result_prod.duals["factory_a"]
    factory_b_value = -result_prod.duals["factory_b"]

    print(f"  Increasing Factory A capacity by 1 unit saves: ${factory_a_value:.2f}")
    print(f"  Increasing Factory B capacity by 1 unit saves: ${factory_b_value:.2f}")

    if factory_a_value > factory_b_value:
        print(f"\n  ✓ RECOMMENDATION: Expand Factory A (higher marginal value)")
        print(f"    Marginal benefit: ${factory_a_value:.2f}/unit vs ${factory_b_value:.2f}/unit")
    else:
        print(f"\n  ✓ RECOMMENDATION: Expand Factory B (higher marginal value)")

    # Capacity constraint analysis
    print_section_header("CAPACITY CONSTRAINT ANALYSIS")

    print("\nWhich arcs are capacity-constrained?")
    print("  - If an arc is at capacity, its capacity has shadow value")
    print("  - Dual values help identify bottlenecks")

    for (tail, head), flow in sorted(result_prod.flows.items()):
        capacity = next(
            arc["capacity"]
            for arc in arcs_production
            if arc["tail"] == tail and arc["head"] == head
        )
        utilization = (flow / capacity) * 100 if capacity > 0 else 0
        at_capacity = abs(flow - capacity) < 1e-3

        print(f"\n  Arc {tail} -> {head}:")
        print(f"    Flow: {flow:.2f} / Capacity: {capacity:.2f} ({utilization:.1f}%)")
        if at_capacity:
            print(f"    ⚠ BOTTLENECK: Arc is at full capacity!")
            print(f"    Increasing capacity here would reduce cost by ~${factory_a_value:.2f}/unit")
        else:
            slack = capacity - flow
            print(f"    Available slack: {slack:.2f} units")

    # Summary of key concepts
    print_section_header("KEY CONCEPTS SUMMARY")

    print("\n1. DUAL VALUES (Shadow Prices):")
    print("   - Marginal cost of increasing supply at a node by 1 unit")
    print("   - Marginal benefit of decreasing demand at a node by 1 unit")
    print("   - Negative dual = it costs to supply more at that node")
    print("   - Positive dual = it benefits to consume less at that node")

    print("\n2. COMPLEMENTARY SLACKNESS:")
    print("   - For arcs with positive flow:")
    print("     reduced_cost = cost + dual[tail] - dual[head] ≈ 0")
    print("   - This is an optimality condition")
    print("   - Violated reduced cost → solution is not optimal")

    print("\n3. SENSITIVITY ANALYSIS:")
    print("   - Predict cost change: Δcost ≈ Δsupply × dual[node]")
    print("   - Valid for small changes (within basis)")
    print("   - Large changes may require re-solving")

    print("\n4. PRACTICAL APPLICATIONS:")
    print("   - Production planning: Which facility to expand?")
    print("   - Logistics: Value of warehouse capacity increases?")
    print("   - Pricing: How much to charge for expedited delivery?")
    print("   - Bottleneck identification: Which constraints are binding?")

    print("\n" + "=" * 70)
    print("For more details, see docs/algorithm.md and docs/api.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
