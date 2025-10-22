"""Visualization Examples for Network Flow Problems.

This script demonstrates the visualization utilities for network structures,
flow solutions, and bottleneck analysis.

Requires optional visualization dependencies:
    pip install 'network_solver[visualization]'

Examples demonstrated:
1. Basic network structure visualization
2. Flow solution visualization
3. Bottleneck identification and highlighting
4. Transportation problem with multiple routes
5. Supply chain with transshipment nodes
6. Large network with selective bottleneck analysis
"""

import sys

# Check for visualization dependencies
try:
    from network_solver import (
        build_problem,
        solve_min_cost_flow,
        visualize_bottlenecks,
        visualize_flows,
        visualize_network,
    )
except ImportError as e:
    print("Error: Visualization dependencies not installed")
    print("Install with: pip install 'network_solver[visualization]'")
    print(f"Details: {e}")
    sys.exit(1)


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def example_1_basic_network():
    """Example 1: Basic network structure visualization."""
    print_section_header("Example 1: Basic Network Structure")

    # Simple transportation problem
    nodes = [
        {"id": "warehouse", "supply": 100.0},
        {"id": "store_a", "supply": -60.0},
        {"id": "store_b", "supply": -40.0},
    ]
    arcs = [
        {"tail": "warehouse", "head": "store_a", "capacity": 80.0, "cost": 2.5},
        {"tail": "warehouse", "head": "store_b", "capacity": 50.0, "cost": 1.8},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    print("Problem: Warehouse distributing to 2 stores")
    print(f"  Nodes: {len(problem.nodes)}")
    print(f"  Arcs: {len(problem.arcs)}")

    # Visualize network structure
    fig = visualize_network(
        problem,
        layout="spring",
        figsize=(10, 7),
        title="Example 1: Basic Distribution Network",
    )
    fig.savefig("viz_example_1_network.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_1_network.png")
    print("  Shows: Network structure with supply/demand nodes and arc costs")


def example_2_flow_visualization():
    """Example 2: Flow solution visualization."""
    print_section_header("Example 2: Flow Solution Visualization")

    # Transportation problem
    nodes = [
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 80.0},
        {"id": "warehouse_1", "supply": -70.0},
        {"id": "warehouse_2", "supply": -60.0},
        {"id": "warehouse_3", "supply": -50.0},
    ]
    arcs = [
        # From factory A
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 70.0, "cost": 2.0},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 70.0, "cost": 3.0},
        {"tail": "factory_a", "head": "warehouse_3", "capacity": 70.0, "cost": 4.0},
        # From factory B
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 60.0, "cost": 1.5},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 60.0, "cost": 2.5},
        {"tail": "factory_b", "head": "warehouse_3", "capacity": 60.0, "cost": 3.5},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print(f"Solved: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")

    # Visualize flow solution
    fig = visualize_flows(
        problem,
        result,
        layout="spring",
        figsize=(12, 8),
        show_zero_flows=False,
        title="Example 2: Optimal Flow Solution",
    )
    fig.savefig("viz_example_2_flows.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_2_flows.png")
    print("  Shows: Flow values on arcs (thickness = flow magnitude)")
    print("  Note: Zero flows hidden for clarity")


def example_3_bottleneck_highlighting():
    """Example 3: Bottleneck identification."""
    print_section_header("Example 3: Bottleneck Highlighting")

    # Create problem with bottleneck
    nodes = [
        {"id": "supplier", "supply": 200.0},
        {"id": "hub", "supply": 0.0},
        {"id": "customer_a", "supply": -120.0},
        {"id": "customer_b", "supply": -80.0},
    ]
    arcs = [
        # Supplier to hub (bottleneck - tight capacity)
        {"tail": "supplier", "head": "hub", "capacity": 200.0, "cost": 1.0},
        # Hub to customers (plenty of capacity)
        {"tail": "hub", "head": "customer_a", "capacity": 150.0, "cost": 2.0},
        {"tail": "hub", "head": "customer_b", "capacity": 100.0, "cost": 1.5},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print(f"Solved: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")

    # Visualize with bottleneck highlighting
    fig = visualize_flows(
        problem,
        result,
        layout="spring",
        figsize=(12, 8),
        highlight_bottlenecks=True,
        bottleneck_threshold=0.95,
        title="Example 3: Flow with Bottleneck Highlighting (≥95% utilization)",
    )
    fig.savefig("viz_example_3_bottleneck_flows.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_3_bottleneck_flows.png")
    print("  Shows: Bottleneck arcs highlighted in red")
    print("  Bottleneck: supplier → hub (100% utilization)")


def example_4_transportation_network():
    """Example 4: Multi-source multi-sink transportation."""
    print_section_header("Example 4: Transportation Network")

    # 3×3 transportation problem
    nodes = [
        {"id": "plant_1", "supply": 150.0},
        {"id": "plant_2", "supply": 120.0},
        {"id": "plant_3", "supply": 100.0},
        {"id": "depot_1", "supply": -140.0},
        {"id": "depot_2", "supply": -130.0},
        {"id": "depot_3", "supply": -100.0},
    ]
    arcs = [
        # Plant 1
        {"tail": "plant_1", "head": "depot_1", "capacity": 200.0, "cost": 8.0},
        {"tail": "plant_1", "head": "depot_2", "capacity": 200.0, "cost": 6.0},
        {"tail": "plant_1", "head": "depot_3", "capacity": 200.0, "cost": 10.0},
        # Plant 2
        {"tail": "plant_2", "head": "depot_1", "capacity": 200.0, "cost": 9.0},
        {"tail": "plant_2", "head": "depot_2", "capacity": 200.0, "cost": 12.0},
        {"tail": "plant_2", "head": "depot_3", "capacity": 200.0, "cost": 13.0},
        # Plant 3
        {"tail": "plant_3", "head": "depot_1", "capacity": 200.0, "cost": 14.0},
        {"tail": "plant_3", "head": "depot_2", "capacity": 200.0, "cost": 7.0},
        {"tail": "plant_3", "head": "depot_3", "capacity": 200.0, "cost": 9.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    # Visualize network structure
    fig1 = visualize_network(
        problem,
        layout="kamada_kawai",
        figsize=(14, 10),
        title="Example 4a: Transportation Network Structure",
    )
    fig1.savefig("viz_example_4a_network.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: viz_example_4a_network.png")
    print("  Shows: 3 plants × 3 depots network structure")

    # Solve and visualize flows
    result = solve_min_cost_flow(problem)
    print(f"\nSolved: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")

    fig2 = visualize_flows(
        problem,
        result,
        layout="kamada_kawai",
        figsize=(14, 10),
        show_zero_flows=False,
        title="Example 4b: Optimal Transportation Routes",
    )
    fig2.savefig("viz_example_4b_flows.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_4b_flows.png")
    print("  Shows: Optimal routing (only active routes shown)")


def example_5_supply_chain():
    """Example 5: Supply chain with transshipment."""
    print_section_header("Example 5: Supply Chain with Transshipment")

    # Supply chain: factories → distribution centers → stores
    nodes = [
        # Factories
        {"id": "factory_east", "supply": 200.0},
        {"id": "factory_west", "supply": 150.0},
        # Distribution centers (transshipment)
        {"id": "dc_central", "supply": 0.0},
        {"id": "dc_north", "supply": 0.0},
        # Stores
        {"id": "store_a", "supply": -120.0},
        {"id": "store_b", "supply": -100.0},
        {"id": "store_c", "supply": -80.0},
        {"id": "store_d", "supply": -50.0},
    ]
    arcs = [
        # Factories to DCs
        {"tail": "factory_east", "head": "dc_central", "capacity": 200.0, "cost": 5.0},
        {"tail": "factory_east", "head": "dc_north", "capacity": 200.0, "cost": 8.0},
        {"tail": "factory_west", "head": "dc_central", "capacity": 150.0, "cost": 7.0},
        {"tail": "factory_west", "head": "dc_north", "capacity": 150.0, "cost": 4.0},
        # DCs to Stores
        {"tail": "dc_central", "head": "store_a", "capacity": 150.0, "cost": 3.0},
        {"tail": "dc_central", "head": "store_b", "capacity": 150.0, "cost": 2.5},
        {"tail": "dc_north", "head": "store_c", "capacity": 100.0, "cost": 3.5},
        {"tail": "dc_north", "head": "store_d", "capacity": 100.0, "cost": 2.0},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    # Visualize network
    fig1 = visualize_network(
        problem,
        layout="spring",
        figsize=(14, 10),
        title="Example 5a: Multi-Stage Supply Chain",
    )
    fig1.savefig("viz_example_5a_network.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: viz_example_5a_network.png")
    print("  Shows: Factories → DCs → Stores (multi-stage)")

    # Solve and visualize flows
    result = solve_min_cost_flow(problem)
    print(f"\nSolved: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")

    fig2 = visualize_flows(
        problem,
        result,
        layout="spring",
        figsize=(14, 10),
        highlight_bottlenecks=True,
        bottleneck_threshold=0.85,
        title="Example 5b: Supply Chain Flow (bottlenecks ≥85%)",
    )
    fig2.savefig("viz_example_5b_flows.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_5b_flows.png")
    print("  Shows: Flow through supply chain with bottleneck detection")


def example_6_bottleneck_analysis():
    """Example 6: Focused bottleneck analysis."""
    print_section_header("Example 6: Detailed Bottleneck Analysis")

    # Create problem with multiple bottlenecks
    nodes = [
        {"id": "source_a", "supply": 100.0},
        {"id": "source_b", "supply": 80.0},
        {"id": "hub_1", "supply": 0.0},
        {"id": "hub_2", "supply": 0.0},
        {"id": "sink_x", "supply": -90.0},
        {"id": "sink_y", "supply": -90.0},
    ]
    arcs = [
        # Sources to hubs
        {"tail": "source_a", "head": "hub_1", "capacity": 100.0, "cost": 1.0},
        {"tail": "source_b", "head": "hub_2", "capacity": 80.0, "cost": 1.0},
        # Hubs to sinks (some tight capacities)
        {"tail": "hub_1", "head": "sink_x", "capacity": 50.0, "cost": 2.0},  # Bottleneck
        {"tail": "hub_1", "head": "sink_y", "capacity": 60.0, "cost": 2.5},  # Bottleneck
        {"tail": "hub_2", "head": "sink_x", "capacity": 100.0, "cost": 3.0},
        {"tail": "hub_2", "head": "sink_y", "capacity": 100.0, "cost": 2.8},
    ]

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print(f"Solved: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")

    # Focused bottleneck visualization
    fig = visualize_bottlenecks(
        problem,
        result,
        threshold=0.8,
        layout="spring",
        figsize=(12, 8),
        title="Example 6: Bottleneck Analysis (≥80% utilization)",
    )
    fig.savefig("viz_example_6_bottlenecks.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: viz_example_6_bottlenecks.png")
    print("  Shows: Utilization heatmap with color gradient")
    print("  Red = high utilization, Yellow = medium")

    # Print bottleneck details
    from network_solver import compute_bottleneck_arcs

    bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.8)
    if bottlenecks:
        print("\n  Identified bottlenecks:")
        for b in bottlenecks:
            print(f"    {b.tail} → {b.head}: {b.utilization * 100:.1f}% utilization")
            print(f"      Flow: {b.flow:.0f} / Capacity: {b.capacity:.0f}")
            print(f"      Slack: {b.slack:.0f} units")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 80)
    print("Network Flow Visualization Examples".center(80))
    print("=" * 80)
    print("\nGenerating visualizations...")
    print("(Figures will be saved as PNG files in current directory)")

    try:
        example_1_basic_network()
        example_2_flow_visualization()
        example_3_bottleneck_highlighting()
        example_4_transportation_network()
        example_5_supply_chain()
        example_6_bottleneck_analysis()

        print("\n" + "=" * 80)
        print("Summary".center(80))
        print("=" * 80)
        print("\n✓ All visualizations completed successfully!")
        print("\nGenerated files:")
        print("  1. viz_example_1_network.png - Basic network structure")
        print("  2. viz_example_2_flows.png - Flow solution visualization")
        print("  3. viz_example_3_bottleneck_flows.png - Bottleneck highlighting")
        print("  4. viz_example_4a_network.png - Transportation network")
        print("  5. viz_example_4b_flows.png - Transportation flows")
        print("  6. viz_example_5a_network.png - Supply chain structure")
        print("  7. viz_example_5b_flows.png - Supply chain flows")
        print("  8. viz_example_6_bottlenecks.png - Bottleneck analysis")

        print("\nKey Features Demonstrated:")
        print("  • Network structure visualization (nodes, arcs, costs)")
        print("  • Flow solution visualization (flow values, utilization)")
        print("  • Bottleneck identification and highlighting")
        print("  • Different layout algorithms (spring, circular, kamada_kawai)")
        print("  • Customizable appearance (colors, sizes, labels)")
        print("  • Multiple problem types (transportation, supply chain)")

        print("\nUsage Tips:")
        print("  • Use visualize_network() to understand problem structure")
        print("  • Use visualize_flows() to see optimal flow patterns")
        print("  • Use visualize_bottlenecks() to identify capacity constraints")
        print("  • Set show_zero_flows=False for cleaner flow visualizations")
        print("  • Adjust bottleneck_threshold to focus on tightest constraints")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
