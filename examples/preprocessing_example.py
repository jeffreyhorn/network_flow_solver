"""
Demonstrates problem preprocessing for faster solving.

Preprocessing reduces problem size by:
- Removing redundant parallel arcs with same cost
- Detecting disconnected components
- Simplifying series arcs (merging consecutive arcs)
- Removing zero-supply nodes with single arc

This example shows how preprocessing can significantly reduce solve time
for large problems with structural redundancy.
"""

import argparse
import logging
import time

from network_solver import (
    build_problem,
    preprocess_and_solve,
    preprocess_problem,
    solve_min_cost_flow,
)


def setup_logging(verbose: int = 0):
    """Configure logging based on verbosity level."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(message)s",
    )


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def print_subsection(title: str):
    """Print a subsection header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def example1_redundant_arcs():
    """Example 1: Remove redundant parallel arcs."""
    print_subsection("Example 1: Redundant Parallel Arcs")
    print("Problem with multiple arcs between same nodes with identical costs")
    print()

    nodes = [
        {"id": "warehouse", "supply": 1000.0},
        {"id": "store", "supply": -1000.0},
    ]

    # Multiple delivery routes with same cost (can be merged)
    arcs = [
        {"tail": "warehouse", "head": "store", "capacity": 300.0, "cost": 5.0},
        {"tail": "warehouse", "head": "store", "capacity": 300.0, "cost": 5.0},  # Duplicate
        {"tail": "warehouse", "head": "store", "capacity": 400.0, "cost": 5.0},  # Duplicate
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print(f"Original problem: {len(arcs)} arcs")

    result = preprocess_problem(problem)

    print(f"After preprocessing: {len(result.problem.arcs)} arc")
    print(f"  Removed {result.redundant_arcs} redundant arcs")
    print(f"  Combined capacity: {result.problem.arcs[0].capacity} units")
    print(f"  Preprocessing time: {result.preprocessing_time_ms:.2f}ms")
    print()
    print("✓ Redundant arcs merged into single arc with combined capacity")


def example2_series_arcs():
    """Example 2: Simplify series arcs through transshipment nodes."""
    print_subsection("Example 2: Series Arc Simplification")
    print("Problem with zero-supply transshipment nodes in series")
    print()

    nodes = [
        {"id": "factory", "supply": 500.0},
        {"id": "hub1", "supply": 0.0},  # Transshipment
        {"id": "hub2", "supply": 0.0},  # Transshipment
        {"id": "hub3", "supply": 0.0},  # Transshipment
        {"id": "customer", "supply": -500.0},
    ]

    arcs = [
        {"tail": "factory", "head": "hub1", "capacity": 600.0, "cost": 10.0},
        {"tail": "hub1", "head": "hub2", "capacity": 550.0, "cost": 5.0},
        {"tail": "hub2", "head": "hub3", "capacity": 500.0, "cost": 3.0},
        {"tail": "hub3", "head": "customer", "capacity": 600.0, "cost": 7.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print(f"Original problem: {len(nodes)} nodes, {len(arcs)} arcs")

    result = preprocess_problem(problem)

    print(
        f"After preprocessing: {len(result.problem.nodes)} nodes, {len(result.problem.arcs)} arcs"
    )
    print(f"  Removed {result.removed_nodes} transshipment nodes")
    print(f"  Merged {result.merged_arcs} series arcs")
    print(f"  Preprocessing time: {result.preprocessing_time_ms:.2f}ms")
    print()
    print("✓ Series arcs merged: factory → customer (capacity: min, cost: sum)")


def example3_disconnected_components():
    """Example 3: Detect disconnected components."""
    print_subsection("Example 3: Disconnected Component Detection")
    print("Problem with separate, disconnected subnetworks")
    print()

    nodes = [
        # Component 1
        {"id": "factory_A", "supply": 100.0},
        {"id": "store_A", "supply": -100.0},
        # Component 2 (disconnected)
        {"id": "factory_B", "supply": 50.0},
        {"id": "store_B", "supply": -50.0},
    ]

    arcs = [
        {"tail": "factory_A", "head": "store_A", "capacity": 200.0, "cost": 5.0},
        {"tail": "factory_B", "head": "store_B", "capacity": 100.0, "cost": 3.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print(f"Problem with {len(nodes)} nodes, {len(arcs)} arcs")

    result = preprocess_problem(problem)

    print(f"Detected {result.disconnected_components} disconnected components")
    print()
    if result.disconnected_components > 1:
        print("⚠ Multiple components detected - each must be balanced independently")
    else:
        print("✓ Single connected component")


def example4_combined_preprocessing():
    """Example 4: Large problem with multiple optimizations."""
    print_subsection("Example 4: Combined Preprocessing (Large Problem)")
    print("Realistic supply chain with redundancy and transshipment nodes")
    print()

    nodes = [
        # Suppliers
        {"id": "supplier_1", "supply": 500.0},
        {"id": "supplier_2", "supply": 300.0},
        # Distribution centers (transshipment)
        {"id": "dc_east", "supply": 0.0},
        {"id": "dc_west", "supply": 0.0},
        {"id": "dc_central", "supply": 0.0},
        # Retailers
        {"id": "retail_1", "supply": -200.0},
        {"id": "retail_2", "supply": -250.0},
        {"id": "retail_3", "supply": -150.0},
        {"id": "retail_4", "supply": -200.0},
    ]

    arcs = [
        # Supplier to DCs (with redundant routes)
        {"tail": "supplier_1", "head": "dc_east", "capacity": 300.0, "cost": 10.0},
        {"tail": "supplier_1", "head": "dc_east", "capacity": 200.0, "cost": 10.0},  # Redundant
        {"tail": "supplier_1", "head": "dc_west", "capacity": 300.0, "cost": 15.0},
        {"tail": "supplier_2", "head": "dc_central", "capacity": 300.0, "cost": 8.0},
        # DC series (some DCs in series)
        {"tail": "dc_east", "head": "retail_1", "capacity": 300.0, "cost": 5.0},
        {"tail": "dc_west", "head": "retail_2", "capacity": 300.0, "cost": 7.0},
        {"tail": "dc_central", "head": "retail_3", "capacity": 200.0, "cost": 6.0},
        {"tail": "dc_central", "head": "retail_4", "capacity": 200.0, "cost": 9.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print("Original problem:")
    print(f"  Nodes: {len(problem.nodes)}")
    print(f"  Arcs:  {len(problem.arcs)}")
    print()

    # Preprocess
    result = preprocess_problem(problem)

    print("After preprocessing:")
    print(f"  Nodes: {len(result.problem.nodes)} (removed {result.removed_nodes})")
    print(f"  Arcs:  {len(result.problem.arcs)} (removed {result.removed_arcs})")
    print()
    print("Optimizations applied:")
    print(f"  Redundant arcs removed: {result.redundant_arcs}")
    print(f"  Series arcs merged: {result.merged_arcs}")
    print(f"  Disconnected components: {result.disconnected_components}")
    print(f"  Preprocessing time: {result.preprocessing_time_ms:.2f}ms")
    print()
    print("✓ Problem size reduced before solving")


def example5_performance_comparison():
    """Example 5: Compare solve time with and without preprocessing."""
    print_subsection("Example 5: Performance Comparison")
    print("Benchmark: solve with vs without preprocessing")
    print()

    # Create a larger problem with redundancy
    nodes = []
    arcs = []

    # Suppliers
    for i in range(5):
        nodes.append({"id": f"supplier_{i}", "supply": 100.0})

    # Transshipment hubs (in series chains)
    for i in range(10):
        nodes.append({"id": f"hub_{i}", "supply": 0.0})

    # Customers
    for i in range(5):
        nodes.append({"id": f"customer_{i}", "supply": -100.0})

    # Redundant arcs from suppliers to first hub
    for i in range(5):
        arcs.append({"tail": f"supplier_{i}", "head": "hub_0", "capacity": 150.0, "cost": 5.0})
        arcs.append(
            {"tail": f"supplier_{i}", "head": "hub_0", "capacity": 150.0, "cost": 5.0}
        )  # Duplicate

    # Series arcs through hubs
    for i in range(9):
        arcs.append({"tail": f"hub_{i}", "head": f"hub_{i + 1}", "capacity": 500.0, "cost": 2.0})

    # Arcs from last hub to customers
    for i in range(5):
        arcs.append({"tail": "hub_9", "head": f"customer_{i}", "capacity": 150.0, "cost": 8.0})

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print(f"Problem size: {len(nodes)} nodes, {len(arcs)} arcs")
    print()

    # Solve without preprocessing
    print("Solving WITHOUT preprocessing:")
    start = time.time()
    result_no_preproc = solve_min_cost_flow(problem)
    time_no_preproc = (time.time() - start) * 1000

    print(f"  Status: {result_no_preproc.status}")
    print(f"  Iterations: {result_no_preproc.iterations}")
    print(f"  Time: {time_no_preproc:.2f}ms")
    print()

    # Solve with preprocessing
    print("Solving WITH preprocessing:")
    start = time.time()
    preproc_result, result_preproc = preprocess_and_solve(problem)
    time_with_preproc = (time.time() - start) * 1000

    print(f"  Preprocessing: {preproc_result.preprocessing_time_ms:.2f}ms")
    print(f"    Removed {preproc_result.removed_arcs} arcs, {preproc_result.removed_nodes} nodes")
    print(f"  Solving: {time_with_preproc - preproc_result.preprocessing_time_ms:.2f}ms")
    print(f"  Status: {result_preproc.status}")
    print(f"  Iterations: {result_preproc.iterations}")
    print(f"  Total time: {time_with_preproc:.2f}ms")
    print()

    # Compare
    speedup = time_no_preproc / time_with_preproc
    print(f"Speedup: {speedup:.2f}x")
    print(f"Objectives match: {abs(result_no_preproc.objective - result_preproc.objective) < 1e-4}")
    print()

    # Show that solutions are equivalent (flows translated back to original problem)
    print("Solution details:")
    print("  Original problem flow on supplier_0 → hub_0: ", end="")
    flow_sum = sum(
        result_no_preproc.flows.get(("supplier_0", "hub_0"), 0.0)
        for _ in range(1)  # Just accessing once since dict aggregates
    )
    print(f"{flow_sum:.1f}")
    print("  Preprocessed solution (translated back):      ", end="")
    flow_translated = result_preproc.flows.get(("supplier_0", "hub_0"), 0.0)
    print(f"{flow_translated:.1f}")
    print(f"  Flows match: {abs(flow_sum - flow_translated) < 1e-4}")
    print()

    if speedup > 1.1:
        print("✓ Preprocessing provided significant speedup!")
    else:
        print("ℹ For this problem size, preprocessing overhead similar to savings")
        print("  (Preprocessing shows larger benefits for bigger problems)")


def example6_selective_preprocessing():
    """Example 6: Selective preprocessing (enable/disable specific optimizations)."""
    print_subsection("Example 6: Selective Preprocessing")
    print("Control which optimizations to apply")
    print()

    nodes = [
        {"id": "A", "supply": 100.0},
        {"id": "B", "supply": 0.0},
        {"id": "C", "supply": -100.0},
    ]

    arcs = [
        {"tail": "A", "head": "B", "capacity": 100.0, "cost": 5.0},
        {"tail": "A", "head": "B", "capacity": 100.0, "cost": 5.0},  # Redundant
        {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    # Only remove redundant arcs (skip series simplification)
    result = preprocess_problem(
        problem,
        remove_redundant=True,
        simplify_series=False,
        detect_disconnected=True,
        remove_zero_supply=False,
    )

    print("Preprocessing with remove_redundant=True, simplify_series=False:")
    print(f"  Nodes: {len(result.problem.nodes)} (kept transshipment node)")
    print(f"  Arcs: {len(result.problem.arcs)} (merged redundant arcs)")
    print(f"  Redundant arcs removed: {result.redundant_arcs}")
    print(f"  Series arcs merged: {result.merged_arcs}")
    print()
    print("✓ Granular control over preprocessing steps")


def main():
    """Run all preprocessing examples."""
    parser = argparse.ArgumentParser(description="Preprocessing example")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    print_section("PROBLEM PREPROCESSING DEMONSTRATION")

    example1_redundant_arcs()
    example2_series_arcs()
    example3_disconnected_components()
    example4_combined_preprocessing()
    example5_performance_comparison()
    example6_selective_preprocessing()

    print_section("SUMMARY: WHEN TO USE PREPROCESSING")

    print("✓ USE preprocessing when:")
    print("  • Large problems with redundant structure")
    print("  • Multiple parallel arcs between same nodes")
    print("  • Long chains of transshipment nodes")
    print("  • Problem has dead-end nodes (single arc, zero supply)")
    print("  • Preprocessing time << solve time (typically true for n > 100)")
    print()
    print("✗ SKIP preprocessing when:")
    print("  • Very small problems (< 10 nodes, < 20 arcs)")
    print("  • Problem already optimized/minimal")
    print()
    print("BENEFITS:")
    print("  • Fewer nodes/arcs → faster simplex pivots")
    print("  • Smaller basis matrix → faster refactorization")
    print("  • Typical speedup: 2-10x for problems with redundancy")
    print("  • Preserves optimal solution (safe transformation)")
    print("  • Solutions automatically translated back to original problem structure")
    print()
    print("USAGE:")
    print("  from network_solver import preprocess_problem, preprocess_and_solve")
    print()
    print("  # Preprocess then solve separately")
    print("  result = preprocess_problem(problem)")
    print("  flow_result = solve_min_cost_flow(result.problem)")
    print()
    print("  # Or use convenience function (automatically translates solution)")
    print("  preproc_result, flow_result = preprocess_and_solve(problem)")
    print("  # flow_result contains flows/duals for ORIGINAL problem arcs/nodes")
    print()
    print("RESULT TRANSLATION:")
    print("  • preprocess_and_solve() automatically translates solutions")
    print("  • Flow values mapped back to original arcs (including removed arcs)")
    print("  • Dual values computed for removed nodes based on adjacent arcs")
    print("  • Redundant arcs: flows distributed proportionally by capacity")
    print("  • Series arcs: all arcs in series carry same flow")
    print("  • Removed arcs: assigned zero flow")
    print()


if __name__ == "__main__":
    main()
