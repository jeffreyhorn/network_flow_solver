"""Example demonstrating sensitivity analysis using dual values."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import build_problem, solve_min_cost_flow  # noqa: E402


def main() -> None:
    """Demonstrate how dual values enable sensitivity analysis."""
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS WITH DUAL VALUES")
    print("=" * 70)
    
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
    
    print(f"\nOptimal Solution:")
    print(f"  Status: {result.status}")
    print(f"  Total cost: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")
    
    print(f"\nFlows:")
    for (tail, head), flow in sorted(result.flows.items()):
        print(f"  {tail} -> {head}: {flow:.2f} units")
    
    print(f"\nDual Values (Shadow Prices):")
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
    expected_change_per_unit = (
        result.duals["supplier"] - result.duals["customer"]
    )
    
    print(f"\nNew total cost: ${result2.objective:.2f}")
    print(f"Actual cost change: ${cost_change:.2f}")
    print(f"Change per unit: ${cost_change / 10:.2f}")
    print(f"\nExpected from duals: ${expected_change_per_unit:.6f} per unit")
    print(f"  (dual[supplier] - dual[customer] = {result.duals['supplier']:.6f} - {result.duals['customer']:.6f})")
    
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
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
