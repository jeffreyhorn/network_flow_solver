#!/usr/bin/env python
"""
NetworkX Comparison Example

This example compares the network_solver package with NetworkX's min-cost flow
implementation. It demonstrates:

1. API differences between the two libraries
2. Performance comparison on various problem sizes
3. When to use each library
4. Feature comparison (dual values, solver options, etc.)

NetworkX is a widely-used graph library with built-in network flow algorithms,
while network_solver is a specialized implementation focused on performance
and detailed solver control.
"""

import time
from typing import Any

import networkx as nx

from network_solver import build_problem, solve_min_cost_flow, SolverOptions


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


# ==============================================================================
# Example 1: Basic API Comparison
# ==============================================================================


def example_1_api_comparison() -> None:
    """Compare the basic API of both libraries."""
    print_section_header("EXAMPLE 1: API COMPARISON")

    # Define the same problem for both libraries
    # Transportation problem: 2 warehouses, 2 stores
    print("Problem: Ship goods from 2 warehouses to 2 stores at minimum cost\n")
    print("Warehouses (supply):")
    print("  - Warehouse A: 60 units")
    print("  - Warehouse B: 40 units")
    print("\nStores (demand):")
    print("  - Store 1: 50 units")
    print("  - Store 2: 50 units")
    print("\nShipping costs:")
    print("  - Warehouse A → Store 1: $2/unit")
    print("  - Warehouse A → Store 2: $3/unit")
    print("  - Warehouse B → Store 1: $4/unit")
    print("  - Warehouse B → Store 2: $1/unit")

    # -------------------------
    # network_solver approach
    # -------------------------
    print_subsection("network_solver Approach")

    nodes = [
        {"id": "warehouse_a", "supply": 60.0},
        {"id": "warehouse_b", "supply": 40.0},
        {"id": "store_1", "supply": -50.0},
        {"id": "store_2", "supply": -50.0},
    ]

    arcs = [
        {"tail": "warehouse_a", "head": "store_1", "cost": 2.0},
        {"tail": "warehouse_a", "head": "store_2", "cost": 3.0},
        {"tail": "warehouse_b", "head": "store_1", "cost": 4.0},
        {"tail": "warehouse_b", "head": "store_2", "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print("Code:")
    print("  nodes = [")
    print('    {"id": "warehouse_a", "supply": 60.0},')
    print('    {"id": "store_1", "supply": -50.0},  # negative = demand')
    print("    ...]")
    print("  arcs = [")
    print('    {"tail": "warehouse_a", "head": "store_1", "cost": 2.0},')
    print("    ...]")
    print("  problem = build_problem(nodes, arcs, directed=True)")
    print("  result = solve_min_cost_flow(problem)")
    print(f"\nTotal cost: ${result.objective:.2f}")
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")
    print("\nOptimal flows:")
    for (tail, head), flow in sorted(result.flows.items()):
        if flow > 1e-6:
            print(f"  {tail} → {head}: {flow:.1f} units")

    # -------------------------
    # NetworkX approach
    # -------------------------
    print_subsection("NetworkX Approach")

    G = nx.DiGraph()
    # Add nodes with demand (positive = receive, negative = send)
    # Note: NetworkX uses opposite sign convention!
    G.add_node("warehouse_a", demand=-60)  # Negative = supply in NetworkX
    G.add_node("warehouse_b", demand=-40)
    G.add_node("store_1", demand=50)  # Positive = demand in NetworkX
    G.add_node("store_2", demand=50)

    # Add edges with costs (weight)
    G.add_edge("warehouse_a", "store_1", weight=2)
    G.add_edge("warehouse_a", "store_2", weight=3)
    G.add_edge("warehouse_b", "store_1", weight=4)
    G.add_edge("warehouse_b", "store_2", weight=1)

    flow_dict = nx.min_cost_flow(G)
    cost = nx.cost_of_flow(G, flow_dict)

    print("Code:")
    print("  G = nx.DiGraph()")
    print('  G.add_node("warehouse_a", demand=-60)  # negative = supply')
    print('  G.add_node("store_1", demand=50)       # positive = demand')
    print('  G.add_edge("warehouse_a", "store_1", weight=2)')
    print("  flow_dict = nx.min_cost_flow(G)")
    print("  cost = nx.cost_of_flow(G, flow_dict)")
    print(f"\nTotal cost: ${cost:.2f}")
    print("\nOptimal flows:")
    for u in sorted(flow_dict.keys()):
        for v in sorted(flow_dict[u].keys()):
            if flow_dict[u][v] > 1e-6:
                print(f"  {u} → {v}: {flow_dict[u][v]:.1f} units")

    # -------------------------
    # Key differences
    # -------------------------
    print_subsection("Key API Differences")
    print("1. Sign convention:")
    print("   - network_solver: supply is positive, demand is negative")
    print("   - NetworkX: supply is negative (demand=-60), demand is positive (demand=50)")
    print("\n2. Return format:")
    print("   - network_solver: FlowResult object with flows dict, objective, duals, etc.")
    print("   - NetworkX: flow_dict[u][v] nested dictionary")
    print("\n3. Cost calculation:")
    print("   - network_solver: result.objective (included)")
    print("   - NetworkX: nx.cost_of_flow(G, flow_dict) (separate function)")
    print("\n4. Solver information:")
    print("   - network_solver: iteration count, status, solve time in result")
    print("   - NetworkX: not available")


# ==============================================================================
# Example 2: Feature Comparison
# ==============================================================================


def example_2_feature_comparison() -> None:
    """Compare advanced features of both libraries."""
    print_section_header("EXAMPLE 2: FEATURE COMPARISON")

    # Create a simple problem
    nodes = [
        {"id": "source", "supply": 100.0},
        {"id": "sink", "supply": -100.0},
    ]
    arcs = [
        {"tail": "source", "head": "sink", "cost": 1.0, "capacity": 100.0},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # NetworkX equivalent
    G = nx.DiGraph()
    G.add_node("source", demand=-100)
    G.add_node("sink", demand=100)
    G.add_edge("source", "sink", weight=1, capacity=100)

    # -------------------------
    # Dual values (shadow prices)
    # -------------------------
    print_subsection("Feature: Dual Values (Shadow Prices)")

    result = solve_min_cost_flow(problem)
    print("network_solver:")
    print("  ✓ Dual values available in result.duals")
    print(f"    Dual values: {result.duals}")
    print("  ✓ Use for sensitivity analysis (marginal costs)")

    print("\nNetworkX:")
    print("  ✗ Dual values not directly available")
    print("  ✗ Cannot perform sensitivity analysis without re-solving")

    # -------------------------
    # Solver configuration
    # -------------------------
    print_subsection("Feature: Solver Configuration")

    options = SolverOptions(
        max_iterations=1000,
        tolerance=1e-9,
        pricing_strategy="devex",
        ft_update_limit=64,
    )
    result = solve_min_cost_flow(problem, options=options)

    print("network_solver:")
    print("  ✓ Configurable solver options:")
    print("    - max_iterations (iteration limit)")
    print("    - tolerance (numerical precision)")
    print("    - pricing_strategy ('devex' or 'dantzig')")
    print("    - ft_update_limit (basis refactorization frequency)")
    print("    - block_size (pricing block size)")

    print("\nNetworkX:")
    print("  ✗ No configurable solver options")
    print("  ✗ Uses default capacity scaling algorithm")

    # -------------------------
    # Solver diagnostics
    # -------------------------
    print_subsection("Feature: Solver Diagnostics")

    print("network_solver:")
    print(f"  ✓ Iteration count: {result.iterations}")
    print(f"  ✓ Status: {result.status}")
    print("  ✓ Detailed logging available (INFO/DEBUG levels)")
    print("  ✓ Structured logging with JSON output")

    print("\nNetworkX:")
    print("  ✗ No iteration count")
    print("  ✗ No status information")
    print("  ✗ No solver logging")

    # -------------------------
    # Lower bounds on arcs
    # -------------------------
    print_subsection("Feature: Lower Bounds on Arcs")

    nodes_lb = [
        {"id": "a", "supply": 50.0},
        {"id": "b", "supply": -50.0},
    ]
    arcs_lb = [
        {"tail": "a", "head": "b", "cost": 1.0, "lower": 20.0, "capacity": 100.0},
    ]
    problem_lb = build_problem(nodes=nodes_lb, arcs=arcs_lb, directed=True, tolerance=1e-6)
    result_lb = solve_min_cost_flow(problem_lb)

    print("network_solver:")
    print("  ✓ Supports lower bounds on arcs")
    print(f"    Flow on arc with lower=20: {result_lb.flows[('a', 'b')]:.1f}")
    print("  ✓ Useful for minimum flow requirements")

    print("\nNetworkX:")
    print("  ✗ No native support for lower bounds")
    print("  ✗ Would require problem transformation")

    # -------------------------
    # Undirected graphs
    # -------------------------
    print_subsection("Feature: Undirected Graphs")

    print("network_solver:")
    print("  ✓ Native support for undirected graphs")
    print("    Set directed=False in build_problem()")
    print("  ✓ Automatic transformation (each edge → arc with lower=-C, upper=C)")

    print("\nNetworkX:")
    print("  ✓ Supports undirected graphs via nx.Graph()")
    print("  ✓ Can convert to DiGraph if needed")

    # -------------------------
    # Summary table
    # -------------------------
    print_subsection("Feature Summary")
    print("┌─────────────────────────────┬──────────────────┬──────────────┐")
    print("│ Feature                     │ network_solver   │ NetworkX     │")
    print("├─────────────────────────────┼──────────────────┼──────────────┤")
    print("│ Min-cost flow               │ ✓                │ ✓            │")
    print("│ Dual values / shadow prices │ ✓                │ ✗            │")
    print("│ Solver configuration        │ ✓                │ ✗            │")
    print("│ Iteration count / status    │ ✓                │ ✗            │")
    print("│ Lower bounds on arcs        │ ✓                │ ✗            │")
    print("│ Structured logging          │ ✓                │ ✗            │")
    print("│ Undirected graphs           │ ✓                │ ✓            │")
    print("│ Other graph algorithms      │ ✗                │ ✓            │")
    print("│ Graph analysis tools        │ ✗                │ ✓            │")
    print("│ Community / ecosystem       │ Small            │ Large        │")
    print("└─────────────────────────────┴──────────────────┴──────────────┘")


# ==============================================================================
# Example 3: Performance Comparison
# ==============================================================================


def example_3_performance_comparison() -> None:
    """Compare performance on various problem sizes."""
    print_section_header("EXAMPLE 3: PERFORMANCE COMPARISON")

    print("Comparing solve times on grid networks of increasing size...\n")
    print("┌────────┬───────┬──────────────────────┬──────────────────────┬─────────────┐")
    print("│ Size   │ Arcs  │ network_solver (ms)  │ NetworkX (ms)        │ Speedup     │")
    print("├────────┼───────┼──────────────────────┼──────────────────────┼─────────────┤")

    sizes = [(3, 3), (5, 5), (8, 8), (10, 10), (15, 15)]

    for rows, cols in sizes:
        # Generate grid problem for network_solver
        nodes, arcs = generate_grid_network(rows, cols)
        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

        # Solve with network_solver
        start = time.perf_counter()
        result_ns = solve_min_cost_flow(problem)
        time_ns = (time.perf_counter() - start) * 1000

        # Generate grid for NetworkX
        G = create_networkx_grid(rows, cols)

        # Solve with NetworkX
        start = time.perf_counter()
        try:
            flow_dict = nx.min_cost_flow(G)
            time_nx = (time.perf_counter() - start) * 1000
            speedup = time_nx / time_ns if time_ns > 0 else 0
            speedup_str = f"{speedup:.2f}x"
        except Exception as e:
            time_nx = float("inf")
            speedup_str = "N/A"

        num_arcs = len(arcs)
        print(
            f"│ {rows:2d}×{cols:<2d}  │ {num_arcs:5d} │ {time_ns:18.2f} │ "
            f"{time_nx:18.2f} │ {speedup_str:11s} │"
        )

    print("└────────┴───────┴──────────────────────┴──────────────────────┴─────────────┘")

    print("\nNotes:")
    print("- Both libraries solve the same problem to optimality")
    print("- network_solver uses network simplex (primal method)")
    print("- NetworkX uses capacity scaling algorithm")
    print("- Performance depends on problem structure and size")


# ==============================================================================
# Example 4: When to Use Each Library
# ==============================================================================


def example_4_when_to_use() -> None:
    """Guidance on when to use each library."""
    print_section_header("EXAMPLE 4: WHEN TO USE EACH LIBRARY")

    print_subsection("Use network_solver when:")
    print("✓ You need dual values (shadow prices) for sensitivity analysis")
    print("✓ You want fine-grained control over the solver (iterations, tolerance, strategy)")
    print("✓ You need detailed solver diagnostics (iteration counts, status, logging)")
    print("✓ You have lower bounds on arcs (minimum flow requirements)")
    print("✓ You want structured logging for monitoring/analytics")
    print("✓ Performance on large problems is critical")
    print("✓ You're focused specifically on min-cost flow problems")

    print_subsection("Use NetworkX when:")
    print("✓ You need a wide variety of graph algorithms (shortest paths, centrality, etc.)")
    print("✓ You're doing general graph analysis and min-cost flow is just one part")
    print("✓ You want to visualize graphs easily (nx.draw)")
    print("✓ You need community support and extensive documentation")
    print("✓ You want to leverage the large NetworkX ecosystem")
    print("✓ You're prototyping and don't need fine-grained solver control")
    print("✓ You're already using NetworkX for other graph operations")

    print_subsection("Use both when:")
    print("✓ You need NetworkX for graph construction/visualization")
    print("✓ But want network_solver for actual optimization (better control/diagnostics)")
    print("✓ You can convert between the two formats easily")

    print_subsection("Example workflow combining both:")
    print("1. Use NetworkX to build and visualize your network")
    print("2. Convert to network_solver format for optimization")
    print("3. Analyze dual values and sensitivity with network_solver")
    print("4. Convert results back to NetworkX for visualization")


# ==============================================================================
# Helper Functions
# ==============================================================================


def generate_grid_network(rows: int, cols: int) -> tuple[list[dict], list[dict]]:
    """Generate a 4-connected grid network for testing."""
    nodes = []
    arcs = []

    # Source at top-left, sink at bottom-right
    total_supply = rows * cols * 10.0

    for i in range(rows):
        for j in range(cols):
            node_id = f"n_{i}_{j}"
            if i == 0 and j == 0:
                nodes.append({"id": node_id, "supply": total_supply})
            elif i == rows - 1 and j == cols - 1:
                nodes.append({"id": node_id, "supply": -total_supply})
            else:
                nodes.append({"id": node_id, "supply": 0.0})

            # Add edges to neighbors (right, down)
            if j < cols - 1:
                neighbor = f"n_{i}_{j + 1}"
                arcs.append(
                    {
                        "tail": node_id,
                        "head": neighbor,
                        "cost": 1.0,
                        "capacity": total_supply,
                    }
                )
                arcs.append(
                    {
                        "tail": neighbor,
                        "head": node_id,
                        "cost": 1.0,
                        "capacity": total_supply,
                    }
                )

            if i < rows - 1:
                neighbor = f"n_{i + 1}_{j}"
                arcs.append(
                    {
                        "tail": node_id,
                        "head": neighbor,
                        "cost": 1.0,
                        "capacity": total_supply,
                    }
                )
                arcs.append(
                    {
                        "tail": neighbor,
                        "head": node_id,
                        "cost": 1.0,
                        "capacity": total_supply,
                    }
                )

    return nodes, arcs


def create_networkx_grid(rows: int, cols: int) -> nx.DiGraph:
    """Create a NetworkX grid network matching generate_grid_network."""
    G = nx.DiGraph()
    total_supply = rows * cols * 10.0

    # Add nodes with demand
    for i in range(rows):
        for j in range(cols):
            node_id = f"n_{i}_{j}"
            if i == 0 and j == 0:
                G.add_node(node_id, demand=-total_supply)  # Source (negative in NetworkX)
            elif i == rows - 1 and j == cols - 1:
                G.add_node(node_id, demand=total_supply)  # Sink (positive in NetworkX)
            else:
                G.add_node(node_id, demand=0)

    # Add edges
    for i in range(rows):
        for j in range(cols):
            node_id = f"n_{i}_{j}"

            if j < cols - 1:
                neighbor = f"n_{i}_{j + 1}"
                G.add_edge(node_id, neighbor, weight=1, capacity=total_supply)
                G.add_edge(neighbor, node_id, weight=1, capacity=total_supply)

            if i < rows - 1:
                neighbor = f"n_{i + 1}_{j}"
                G.add_edge(node_id, neighbor, weight=1, capacity=total_supply)
                G.add_edge(neighbor, node_id, weight=1, capacity=total_supply)

    return G


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  NetworkX vs network_solver Comparison")
    print("=" * 80)
    print("\nThis example compares network_solver with NetworkX's min-cost flow solver.")
    print("NetworkX is a popular graph library with many algorithms, while network_solver")
    print("is specialized for min-cost flow with advanced features and solver control.")

    example_1_api_comparison()
    example_2_feature_comparison()
    example_3_performance_comparison()
    example_4_when_to_use()

    print_section_header("CONCLUSION")
    print("Both libraries are excellent choices depending on your needs:")
    print("\n• network_solver: Specialized, configurable, detailed diagnostics")
    print("• NetworkX: General-purpose, rich ecosystem, easy visualization")
    print("\nChoose based on your requirements, or use both together!")
    print()


if __name__ == "__main__":
    main()
