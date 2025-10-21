"""
Demonstrates automatic problem scaling for improved numerical stability.

This example shows how automatic scaling helps when problem values span
many orders of magnitude, which can cause numerical stability issues.

The solver automatically:
- Detects when values differ by >6 orders of magnitude
- Scales costs, capacities, and supplies to a reasonable range
- Solves the scaled problem
- Unscales the solution back to original units
"""

from network_solver import (
    SolverOptions,
    build_problem,
    compute_scaling_factors,
    should_scale_problem,
    solve_min_cost_flow,
)


def main():
    """Demonstrate automatic problem scaling."""

    print("=" * 70)
    print("AUTOMATIC PROBLEM SCALING DEMONSTRATION")
    print("=" * 70)
    print()

    # Example 1: Problem with extreme value ranges (benefits from scaling)
    print("1. Problem with extreme value ranges")
    print("-" * 70)
    print("   Supply: 100 million units")
    print("   Capacity: 200 million units")
    print("   Cost: $0.001 per unit")
    print("   Range: 11 orders of magnitude (0.001 to 200,000,000)")
    print()

    nodes_extreme = [
        {"id": "source", "supply": 1e8},
        {"id": "sink", "supply": -1e8},
    ]
    arcs_extreme = [
        {"tail": "source", "head": "sink", "capacity": 2e8, "cost": 0.001},
    ]
    problem_extreme = build_problem(
        nodes=nodes_extreme, arcs=arcs_extreme, directed=True, tolerance=1e-6
    )

    # Check if scaling is recommended
    print(f"   Should scale: {should_scale_problem(problem_extreme)}")

    # Show scaling factors
    factors = compute_scaling_factors(problem_extreme)
    print(f"   Cost scale factor: {factors.cost_scale:.2e}")
    print(f"   Capacity scale factor: {factors.capacity_scale:.2e}")
    print(f"   Supply scale factor: {factors.supply_scale:.2e}")
    print()

    # Solve with automatic scaling (default)
    print("   Solving WITH automatic scaling (default)...")
    result_scaled = solve_min_cost_flow(problem_extreme)
    print(f"   Status: {result_scaled.status}")
    print(f"   Objective: ${result_scaled.objective:,.2f}")
    print(f"   Flow: {result_scaled.flows[('source', 'sink')]:,.0f} units")
    print(f"   Iterations: {result_scaled.iterations}")
    print()

    # Solve without scaling for comparison
    print("   Solving WITHOUT automatic scaling...")
    options_no_scale = SolverOptions(auto_scale=False)
    result_no_scale = solve_min_cost_flow(problem_extreme, options=options_no_scale)
    print(f"   Status: {result_no_scale.status}")
    print(f"   Objective: ${result_no_scale.objective:,.2f}")
    print(f"   Flow: {result_no_scale.flows[('source', 'sink')]:,.0f} units")
    print(f"   Iterations: {result_no_scale.iterations}")
    print()
    print("   Note: Both produce the same correct answer!")
    print()

    # Example 2: Balanced problem (no scaling needed)
    print("2. Well-balanced problem (no scaling needed)")
    print("-" * 70)
    print("   Supply: 100 units")
    print("   Capacity: 200 units")
    print("   Cost: $5 per unit")
    print("   Range: <2 orders of magnitude")
    print()

    nodes_balanced = [
        {"id": "source", "supply": 100.0},
        {"id": "sink", "supply": -100.0},
    ]
    arcs_balanced = [
        {"tail": "source", "head": "sink", "capacity": 200.0, "cost": 5.0},
    ]
    problem_balanced = build_problem(
        nodes=nodes_balanced, arcs=arcs_balanced, directed=True, tolerance=1e-6
    )

    print(f"   Should scale: {should_scale_problem(problem_balanced)}")
    print("   Solver will skip scaling (not needed)")
    print()

    result_balanced = solve_min_cost_flow(problem_balanced)
    print(f"   Status: {result_balanced.status}")
    print(f"   Objective: ${result_balanced.objective:,.2f}")
    print(f"   Flow: {result_balanced.flows[('source', 'sink')]:,.0f} units")
    print()

    # Example 3: Multi-commodity flow with mixed scales
    print("3. Transportation problem with mixed value scales")
    print("-" * 70)
    print("   Combining micro-costs with macro-supplies")
    print()

    nodes_transport = [
        {"id": "factory_a", "supply": 5_000_000.0},  # 5 million units
        {"id": "factory_b", "supply": 3_000_000.0},  # 3 million units
        {"id": "warehouse_1", "supply": -4_000_000.0},
        {"id": "warehouse_2", "supply": -2_500_000.0},
        {"id": "warehouse_3", "supply": -1_500_000.0},
    ]

    # Shipping costs in dollars per unit (very small)
    arcs_transport = [
        # From factory A (costs: $0.0001 - $0.0005 per unit)
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 5e6, "cost": 0.00025},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 5e6, "cost": 0.0003},
        {"tail": "factory_a", "head": "warehouse_3", "capacity": 5e6, "cost": 0.00015},
        # From factory B
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 3e6, "cost": 0.00018},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 3e6, "cost": 0.00022},
        {"tail": "factory_b", "head": "warehouse_3", "capacity": 3e6, "cost": 0.00028},
    ]

    problem_transport = build_problem(
        nodes=nodes_transport, arcs=arcs_transport, directed=True, tolerance=1e-6
    )

    print(f"   Should scale: {should_scale_problem(problem_transport)}")

    if should_scale_problem(problem_transport):
        factors_transport = compute_scaling_factors(problem_transport)
        print(f"   Cost scale factor: {factors_transport.cost_scale:.2e}")
        print(f"   Capacity scale factor: {factors_transport.capacity_scale:.2e}")
        print(f"   Supply scale factor: {factors_transport.supply_scale:.2e}")
    print()

    result_transport = solve_min_cost_flow(problem_transport)
    print(f"   Status: {result_transport.status}")
    print(f"   Total cost: ${result_transport.objective:,.2f}")
    print(f"   Iterations: {result_transport.iterations}")
    print()

    print("   Optimal shipments:")
    for (tail, head), flow in sorted(result_transport.flows.items()):
        if flow > 1e-3:
            # Calculate unit cost for this route
            arc = next(a for a in arcs_transport if a["tail"] == tail and a["head"] == head)
            route_cost = flow * arc["cost"]
            print(f"   {tail:12s} -> {head:12s}: {flow:10,.0f} units (${route_cost:8.2f})")
    print()

    # Example 4: Disabling automatic scaling
    print("4. Manually controlling scaling behavior")
    print("-" * 70)
    print("   You can disable auto-scaling with SolverOptions:")
    print()
    print("   options = SolverOptions(auto_scale=False)")
    print("   result = solve_min_cost_flow(problem, options=options)")
    print()
    print("   This is useful when:")
    print("   - Testing specific numerical behaviors")
    print("   - Working with pre-scaled problems")
    print("   - Debugging scaling-related issues")
    print()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ Automatic scaling is ENABLED by default")
    print("✓ Scaling activates when values differ by >1 million (6 orders of magnitude)")
    print("✓ The solver uses geometric mean to compute scaling factors")
    print("✓ Solutions are automatically unscaled to original units")
    print("✓ Scaling improves numerical stability for extreme value ranges")
    print("✓ Well-balanced problems are not affected (scaling is skipped)")
    print("✓ You can disable scaling with SolverOptions(auto_scale=False)")
    print()


if __name__ == "__main__":
    main()
