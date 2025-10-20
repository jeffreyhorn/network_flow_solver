#!/usr/bin/env python
"""
Warm-Start Example

This example demonstrates how to use warm-starting to accelerate solving a
sequence of related network flow problems. Warm-starting reuses the basis
(spanning tree structure) from a previous solve to initialize the solver,
which can dramatically reduce iterations when problems are similar.

Scenarios covered:
1. Supply/demand changes - Adjust node supplies while keeping network structure
2. Cost changes - Update arc costs while preserving network and supplies
3. Capacity changes - Modify arc capacities with similar optimal routes
4. Sequential optimization - Solve a series of increasingly constrained problems
5. What-if analysis - Rapid evaluation of multiple scenarios

Key benefits of warm-starting:
- Reduced iterations (often 50-90% reduction)
- Faster solve times for similar problems
- Efficient sensitivity analysis
- Real-time scenario evaluation
"""

import time

from network_solver import Basis, build_problem, solve_min_cost_flow


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


# ==============================================================================
# Scenario 1: Supply/Demand Changes
# ==============================================================================


def scenario_1_supply_demand_changes() -> None:
    """Demonstrate warm-starting with changing supply/demand."""
    print_section_header("SCENARIO 1: SUPPLY/DEMAND CHANGES")

    print("Problem: Transportation network with 3 warehouses and 3 stores")
    print("We'll solve multiple instances with varying demand levels\n")

    # Define network structure (stays constant)
    base_arcs = [
        # From warehouse A
        {"tail": "warehouse_a", "head": "store_1", "capacity": 100.0, "cost": 2.0},
        {"tail": "warehouse_a", "head": "store_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "warehouse_a", "head": "store_3", "capacity": 100.0, "cost": 4.0},
        # From warehouse B
        {"tail": "warehouse_b", "head": "store_1", "capacity": 100.0, "cost": 3.0},
        {"tail": "warehouse_b", "head": "store_2", "capacity": 100.0, "cost": 2.0},
        {"tail": "warehouse_b", "head": "store_3", "capacity": 100.0, "cost": 3.0},
        # From warehouse C
        {"tail": "warehouse_c", "head": "store_1", "capacity": 100.0, "cost": 4.0},
        {"tail": "warehouse_c", "head": "store_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "warehouse_c", "head": "store_3", "capacity": 100.0, "cost": 2.0},
    ]

    # Scenario: Demand gradually increases at all stores
    demand_levels = [30.0, 40.0, 50.0, 60.0, 70.0]

    print("Solving with increasing demand levels:")
    print("┌────────────┬─────────────┬────────────┬──────────────┬─────────────┐")
    print("│ Demand     │ Cost        │ Iterations │ Time (ms)    │ Speedup     │")
    print("├────────────┼─────────────┼────────────┼──────────────┼─────────────┤")

    basis: Basis | None = None
    baseline_iters = None

    for demand in demand_levels:
        # Each warehouse supplies 1/3 of total demand
        supply_per_warehouse = demand
        nodes = [
            {"id": "warehouse_a", "supply": supply_per_warehouse},
            {"id": "warehouse_b", "supply": supply_per_warehouse},
            {"id": "warehouse_c", "supply": supply_per_warehouse},
            {"id": "store_1", "supply": -demand},
            {"id": "store_2", "supply": -demand},
            {"id": "store_3", "supply": -demand},
        ]

        problem = build_problem(nodes=nodes, arcs=base_arcs, directed=True, tolerance=1e-6)

        # Solve with warm-start if we have a basis
        start = time.perf_counter()
        result = solve_min_cost_flow(problem, warm_start_basis=basis)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Store basis for next iteration
        basis = result.basis

        # Calculate speedup
        if baseline_iters is None:
            baseline_iters = result.iterations
            speedup_str = "baseline"
        else:
            speedup = baseline_iters / result.iterations if result.iterations > 0 else float("inf")
            speedup_str = f"{speedup:.1f}x"

        print(
            f"│ {demand:>10.0f} │ ${result.objective:>10.2f} │ {result.iterations:>10d} │ "
            f"{elapsed_ms:>12.2f} │ {speedup_str:>11s} │"
        )

    print("└────────────┴─────────────┴────────────┴──────────────┴─────────────┘")

    print("\n✓ Warm-starting reduced iterations by reusing the basis from previous solves")
    print("✓ The optimal routing structure remains similar as demand scales")


# ==============================================================================
# Scenario 2: Cost Changes
# ==============================================================================


def scenario_2_cost_changes() -> None:
    """Demonstrate warm-starting with changing costs."""
    print_section_header("SCENARIO 2: COST CHANGES")

    print("Problem: Fuel price changes affect transportation costs")
    print("We'll evaluate different fuel price scenarios\n")

    # Fixed supply/demand
    nodes = [
        {"id": "factory", "supply": 100.0},
        {"id": "warehouse_1", "supply": -60.0},
        {"id": "warehouse_2", "supply": -40.0},
    ]

    # Fuel price multipliers
    fuel_prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    base_cost = 5.0

    print("Evaluating fuel price scenarios:")
    print("┌──────────────┬─────────────┬────────────┬──────────────┬─────────────┐")
    print("│ Fuel Price   │ Total Cost  │ Iterations │ Time (ms)    │ Speedup     │")
    print("├──────────────┼─────────────┼────────────┼──────────────┼─────────────┤")

    basis: Basis | None = None
    baseline_iters = None

    for multiplier in fuel_prices:
        # Update costs based on fuel price
        arcs = [
            {
                "tail": "factory",
                "head": "warehouse_1",
                "capacity": 100.0,
                "cost": base_cost * multiplier * 1.0,
            },  # Closer warehouse
            {
                "tail": "factory",
                "head": "warehouse_2",
                "capacity": 100.0,
                "cost": base_cost * multiplier * 1.5,
            },  # Farther warehouse
        ]

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        start = time.perf_counter()
        result = solve_min_cost_flow(problem, warm_start_basis=basis)
        elapsed_ms = (time.perf_counter() - start) * 1000

        basis = result.basis

        if baseline_iters is None:
            baseline_iters = result.iterations
            speedup_str = "baseline"
        else:
            speedup = baseline_iters / result.iterations if result.iterations > 0 else float("inf")
            speedup_str = f"{speedup:.1f}x"

        print(
            f"│ {multiplier:>12.1f}x │ ${result.objective:>10.2f} │ {result.iterations:>10d} │ "
            f"{elapsed_ms:>12.2f} │ {speedup_str:>11s} │"
        )

    print("└──────────────┴─────────────┴────────────┴──────────────┴─────────────┘")

    print("\n✓ Cost changes don't affect the optimal routing when structure is preserved")
    print("✓ Warm-start enables rapid 'what-if' analysis for different cost scenarios")


# ==============================================================================
# Scenario 3: Capacity Changes
# ==============================================================================


def scenario_3_capacity_changes() -> None:
    """Demonstrate warm-starting with capacity adjustments."""
    print_section_header("SCENARIO 3: CAPACITY EXPANSION ANALYSIS")

    print("Problem: Evaluating the value of expanding arc capacities")
    print("We'll incrementally increase capacity on a bottleneck arc\n")

    nodes = [
        {"id": "source", "supply": 100.0},
        {"id": "intermediate", "supply": 0.0},
        {"id": "sink", "supply": -100.0},
    ]

    # Test increasing capacities on the bottleneck arc
    capacities = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    print("Capacity expansion evaluation:")
    print("┌────────────┬─────────────┬──────────┬────────────┬──────────────┬─────────────┐")
    print("│ Capacity   │ Cost        │ Utilized │ Iterations │ Time (ms)    │ Speedup     │")
    print("├────────────┼─────────────┼──────────┼────────────┼──────────────┼─────────────┤")

    basis: Basis | None = None
    baseline_iters = None

    for capacity in capacities:
        arcs = [
            # Route 1: Direct but limited capacity
            {"tail": "source", "head": "sink", "capacity": capacity, "cost": 1.0},
            # Route 2: Via intermediate (always available, higher cost)
            {"tail": "source", "head": "intermediate", "capacity": 100.0, "cost": 2.0},
            {"tail": "intermediate", "head": "sink", "capacity": 100.0, "cost": 2.0},
        ]

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        start = time.perf_counter()
        result = solve_min_cost_flow(problem, warm_start_basis=basis)
        elapsed_ms = (time.perf_counter() - start) * 1000

        basis = result.basis

        # Calculate utilization of direct arc
        direct_flow = result.flows.get(("source", "sink"), 0.0)
        utilization = (direct_flow / capacity * 100) if capacity > 0 else 0

        if baseline_iters is None:
            baseline_iters = result.iterations
            speedup_str = "baseline"
        else:
            speedup = baseline_iters / result.iterations if result.iterations > 0 else float("inf")
            speedup_str = f"{speedup:.1f}x"

        print(
            f"│ {capacity:>10.0f} │ ${result.objective:>10.2f} │ {utilization:>7.1f}% │ "
            f"{result.iterations:>10d} │ {elapsed_ms:>12.2f} │ {speedup_str:>11s} │"
        )

    print("└────────────┴─────────────┴──────────┴────────────┴──────────────┴─────────────┘")

    print("\n✓ Warm-start accelerates evaluation of capacity expansion scenarios")
    print("✓ Useful for infrastructure investment analysis")


# ==============================================================================
# Scenario 4: Sequential Optimization
# ==============================================================================


def scenario_4_sequential_optimization() -> None:
    """Demonstrate warm-starting for sequential problem-solving."""
    print_section_header("SCENARIO 4: ROLLING HORIZON PLANNING")

    print("Problem: Weekly production planning over 5 weeks")
    print("Each week's solution provides the basis for next week\n")

    weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    weekly_demands = [80.0, 90.0, 85.0, 95.0, 100.0]

    print("Weekly planning with warm-start:")
    print("┌──────────┬────────────┬─────────────┬────────────┬──────────────┬─────────────┐")
    print("│ Week     │ Demand     │ Cost        │ Iterations │ Time (ms)    │ Speedup     │")
    print("├──────────┼────────────┼─────────────┼────────────┼──────────────┼─────────────┤")

    basis: Basis | None = None
    baseline_iters = None

    for week, demand in zip(weeks, weekly_demands, strict=True):
        nodes = [
            {"id": "production", "supply": demand},
            {"id": "customer_a", "supply": -demand * 0.6},
            {"id": "customer_b", "supply": -demand * 0.4},
        ]

        arcs = [
            {"tail": "production", "head": "customer_a", "capacity": 100.0, "cost": 3.0},
            {"tail": "production", "head": "customer_b", "capacity": 100.0, "cost": 4.0},
        ]

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        start = time.perf_counter()
        result = solve_min_cost_flow(problem, warm_start_basis=basis)
        elapsed_ms = (time.perf_counter() - start) * 1000

        basis = result.basis

        if baseline_iters is None:
            baseline_iters = result.iterations
            speedup_str = "baseline"
        else:
            speedup = baseline_iters / result.iterations if result.iterations > 0 else float("inf")
            speedup_str = f"{speedup:.1f}x"

        print(
            f"│ {week:>8s} │ {demand:>10.0f} │ ${result.objective:>10.2f} │ {result.iterations:>10d} │ "
            f"{elapsed_ms:>12.2f} │ {speedup_str:>11s} │"
        )

    print("└──────────┴────────────┴─────────────┴────────────┴──────────────┴─────────────┘")

    print("\n✓ Rolling horizon planning benefits greatly from warm-starting")
    print("✓ Each week's solution provides a good starting basis for the next")


# ==============================================================================
# Scenario 5: Performance Comparison
# ==============================================================================


def scenario_5_performance_comparison() -> None:
    """Compare cold start vs warm start performance."""
    print_section_header("SCENARIO 5: COLD START VS WARM START COMPARISON")

    print("Problem: 10×10 grid network with increasing supply")
    print("Compare performance with and without warm-starting\n")

    # Generate a grid network
    nodes = []
    arcs = []

    grid_size = 10
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = f"n_{i}_{j}"
            if i == 0 and j == 0 or i == grid_size - 1 and j == grid_size - 1:
                nodes.append({"id": node_id, "supply": 0.0})  # Will vary
            else:
                nodes.append({"id": node_id, "supply": 0.0})

            # Add edges to neighbors
            if j < grid_size - 1:
                neighbor = f"n_{i}_{j + 1}"
                arcs.append({"tail": node_id, "head": neighbor, "cost": 1.0, "capacity": 1000.0})
                arcs.append({"tail": neighbor, "head": node_id, "cost": 1.0, "capacity": 1000.0})

            if i < grid_size - 1:
                neighbor = f"n_{i + 1}_{j}"
                arcs.append({"tail": node_id, "head": neighbor, "cost": 1.0, "capacity": 1000.0})
                arcs.append({"tail": neighbor, "head": node_id, "cost": 1.0, "capacity": 1000.0})

    supply_levels = [100.0, 150.0, 200.0, 250.0, 300.0]

    print("Performance comparison:")
    print("┌───────────┬─────────────────────┬──────────────────────┬────────────┐")
    print("│ Supply    │ Cold Start          │ Warm Start           │ Speedup    │")
    print("│           │ (iters / time)      │ (iters / time)       │            │")
    print("├───────────┼─────────────────────┼──────────────────────┼────────────┤")

    basis: Basis | None = None

    for supply in supply_levels:
        # Update supply/demand
        nodes[0]["supply"] = supply
        nodes[-1]["supply"] = -supply

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        # Cold start
        start = time.perf_counter()
        result_cold = solve_min_cost_flow(problem)
        time_cold = (time.perf_counter() - start) * 1000

        # Warm start
        start = time.perf_counter()
        result_warm = solve_min_cost_flow(problem, warm_start_basis=basis)
        time_warm = (time.perf_counter() - start) * 1000

        basis = result_warm.basis

        speedup = (
            result_cold.iterations / result_warm.iterations
            if result_warm.iterations > 0
            else float("inf")
        )

        print(
            f"│ {supply:>9.0f} │ {result_cold.iterations:>4d} / {time_cold:>7.2f} ms │ "
            f"{result_warm.iterations:>4d} / {time_warm:>7.2f} ms │ {speedup:>9.1f}x │"
        )

    print("└───────────┴─────────────────────┴──────────────────────┴────────────┘")

    print("\n✓ Warm-starting provides significant speedup (typically 2-10x)")
    print("✓ Benefits increase with problem size and similarity between solves")


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    """Run all warm-start scenarios."""
    print("\n" + "=" * 80)
    print("  Warm-Start Examples")
    print("=" * 80)
    print("\nWarm-starting reuses the basis from a previous solve to accelerate solving")
    print("similar problems. This is essential for:")
    print("  • Sensitivity analysis (what-if scenarios)")
    print("  • Rolling horizon planning")
    print("  • Real-time optimization")
    print("  • Parameter tuning and calibration")

    scenario_1_supply_demand_changes()
    scenario_2_cost_changes()
    scenario_3_capacity_changes()
    scenario_4_sequential_optimization()
    scenario_5_performance_comparison()

    print_section_header("SUMMARY")
    print("Key takeaways:")
    print("\n1. Extract basis from result:")
    print("   basis = result.basis")
    print("\n2. Use basis for next solve:")
    print("   result = solve_min_cost_flow(problem, warm_start_basis=basis)")
    print("\n3. Warm-starting works best when:")
    print("   • Network structure is similar (same nodes/arcs)")
    print("   • Supply/demand changes are moderate")
    print("   • Cost or capacity changes don't drastically alter optimal routes")
    print("\n4. Performance benefits:")
    print("   • Typical speedup: 2-10x reduction in iterations")
    print("   • Enables real-time scenario evaluation")
    print("   • Essential for interactive optimization applications")
    print()


if __name__ == "__main__":
    main()
