"""
Demonstrates configuring the network simplex solver using SolverOptions.

This example shows how to customize solver behavior including:
- Tolerance for numerical precision
- Pricing strategy (Devex vs Dantzig)
- Block size for pricing blocks
- Forrest-Tomlin update limit for basis refactorization
- Maximum iteration limit
"""

from network_solver import SolverOptions, build_problem, solve_min_cost_flow


def main():
    """Run solver with different configuration options."""

    # Create a sample transportation problem
    # Supply nodes: Factory A (100 units), Factory B (150 units)
    # Demand nodes: Warehouse 1 (80 units), Warehouse 2 (120 units), Warehouse 3 (50 units)
    nodes = [
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 150.0},
        {"id": "warehouse_1", "supply": -80.0},
        {"id": "warehouse_2", "supply": -120.0},
        {"id": "warehouse_3", "supply": -50.0},
    ]

    # Shipping costs and capacities
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

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    print("=" * 70)
    print("SOLVER OPTIONS DEMONSTRATION")
    print("=" * 70)
    print()

    # Example 1: Default settings
    print("1. Default settings (Devex pricing, default tolerance)")
    print("-" * 70)
    result = solve_min_cost_flow(problem)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.2f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 2: High-precision solve with tight tolerance
    print("2. High precision (tolerance=1e-10)")
    print("-" * 70)
    options = SolverOptions(tolerance=1e-10)
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.10f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 3: Dantzig pricing (simpler, may be slower)
    print("3. Dantzig pricing strategy (first eligible arc)")
    print("-" * 70)
    options = SolverOptions(pricing_strategy="dantzig")
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.2f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 4: Custom block size for pricing
    print("4. Custom block size (block_size=2)")
    print("-" * 70)
    options = SolverOptions(block_size=2)
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.2f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 5: Aggressive basis refactorization
    print("5. Frequent basis rebuilds (ft_update_limit=10)")
    print("-" * 70)
    options = SolverOptions(ft_update_limit=10)
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.2f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 6: Limited iterations
    print("6. Limited iterations (max_iterations=5)")
    print("-" * 70)
    options = SolverOptions(max_iterations=5)
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.2f}")
    print(f"   Iterations: {result.iterations}")
    if result.status == "iteration_limit":
        print("   Note: Hit iteration limit before reaching optimality")
    print()

    # Example 7: Combining multiple options
    print("7. Combined settings (tight tolerance + Dantzig pricing)")
    print("-" * 70)
    options = SolverOptions(
        tolerance=1e-9,
        pricing_strategy="dantzig",
        block_size=3,
        ft_update_limit=20,
    )
    result = solve_min_cost_flow(problem, options=options)
    print(f"   Status: {result.status}")
    print(f"   Objective: ${result.objective:,.9f}")
    print(f"   Iterations: {result.iterations}")
    print()

    # Example 8: Using max_iterations parameter to override options
    print("8. Parameter override (options has max_iterations=1000, param=3)")
    print("-" * 70)
    options = SolverOptions(max_iterations=1000)
    # The max_iterations parameter overrides the options value
    result = solve_min_cost_flow(problem, options=options, max_iterations=3)
    print(f"   Status: {result.status}")
    print(f"   Iterations: {result.iterations} (should be <= 3)")
    print()

    # Display final optimal solution
    print("=" * 70)
    print("OPTIMAL SOLUTION (using default settings)")
    print("=" * 70)
    result = solve_min_cost_flow(problem)

    print(f"\nTotal shipping cost: ${result.objective:,.2f}\n")
    print("Shipments:")
    for (tail, head), flow in sorted(result.flows.items()):
        if flow > 1e-6:  # Only show non-zero flows
            print(f"  {tail:12s} -> {head:12s}: {flow:6.1f} units")

    print("\nNode potentials (dual values):")
    for node_id, dual in sorted(result.duals.items()):
        dual_type = "supply" if nodes[0]["id"] <= node_id <= nodes[1]["id"] else "demand"
        print(f"  {node_id:12s}: ${dual:8.2f} (shadow price for {dual_type})")


if __name__ == "__main__":
    main()
