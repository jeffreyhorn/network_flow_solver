"""Performance profiling example for network flow solver.

This example demonstrates how to:
1. Profile solver performance for different problem sizes
2. Analyze iteration counts and solve times
3. Compare pricing strategies (Devex vs Dantzig)
4. Identify performance bottlenecks
5. Generate performance reports

Useful for:
- Understanding solver scaling characteristics
- Choosing optimal solver configuration
- Benchmarking problem instances
- Identifying performance regressions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import SolverOptions, build_problem, solve_min_cost_flow  # noqa: E402


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


def generate_grid_network(rows: int, cols: int) -> tuple[list[dict], list[dict]]:
    """Generate a grid network for testing.

    Creates a grid where:
    - Top-left supplies flow
    - Bottom-right demands flow
    - All other nodes are transshipment
    - Arcs connect adjacent nodes (4-connected)

    Args:
        rows: Number of rows in grid
        cols: Number of columns in grid

    Returns:
        Tuple of (nodes, arcs) for build_problem
    """
    total_supply = rows * cols * 10.0
    nodes = []
    arcs = []

    # Create nodes
    for r in range(rows):
        for c in range(cols):
            node_id = f"node_{r}_{c}"
            if r == 0 and c == 0:
                # Top-left: source
                nodes.append({"id": node_id, "supply": total_supply})
            elif r == rows - 1 and c == cols - 1:
                # Bottom-right: sink
                nodes.append({"id": node_id, "supply": -total_supply})
            else:
                # Transshipment
                nodes.append({"id": node_id, "supply": 0.0})

    # Create arcs (4-connected grid)
    for r in range(rows):
        for c in range(cols):
            node_id = f"node_{r}_{c}"

            # Right neighbor
            if c < cols - 1:
                neighbor = f"node_{r}_{c + 1}"
                cost = 1.0 + (r + c) * 0.1  # Varying costs
                arcs.append(
                    {
                        "tail": node_id,
                        "head": neighbor,
                        "capacity": 100.0,
                        "cost": cost,
                    }
                )

            # Down neighbor
            if r < rows - 1:
                neighbor = f"node_{r + 1}_{c}"
                cost = 1.0 + (r + c) * 0.1
                arcs.append(
                    {
                        "tail": node_id,
                        "head": neighbor,
                        "capacity": 100.0,
                        "cost": cost,
                    }
                )

    return nodes, arcs


def generate_bipartite_network(sources: int, sinks: int) -> tuple[list[dict], list[dict]]:
    """Generate a bipartite (assignment) network.

    Args:
        sources: Number of source nodes
        sinks: Number of sink nodes

    Returns:
        Tuple of (nodes, arcs)
    """
    nodes = []
    arcs = []

    # Total supply must equal total demand
    total_flow = 100.0 * min(sources, sinks)
    supply_per_source = total_flow / sources
    demand_per_sink = total_flow / sinks

    # Source nodes
    for i in range(sources):
        nodes.append({"id": f"source_{i}", "supply": supply_per_source})

    # Sink nodes
    for j in range(sinks):
        nodes.append({"id": f"sink_{j}", "supply": -demand_per_sink})

    # Complete bipartite graph
    for i in range(sources):
        for j in range(sinks):
            cost = (i + 1) * (j + 1) * 0.5  # Varying costs
            arcs.append(
                {
                    "tail": f"source_{i}",
                    "head": f"sink_{j}",
                    "capacity": 200.0,
                    "cost": cost,
                }
            )

    return nodes, arcs


def profile_problem(
    name: str,
    nodes: list[dict],
    arcs: list[dict],
    options: SolverOptions | None = None,
) -> dict[str, Any]:
    """Profile a single problem instance.

    Args:
        name: Problem name for display
        nodes: Node list
        arcs: Arc list
        options: Solver options

    Returns:
        Dictionary with profiling results
    """
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    start_time = time.perf_counter()
    result = solve_min_cost_flow(problem, options=options)
    elapsed = time.perf_counter() - start_time

    return {
        "name": name,
        "nodes": len(nodes),
        "arcs": len(arcs),
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "elapsed_ms": elapsed * 1000,
        "iter_per_sec": result.iterations / elapsed if elapsed > 0 else 0,
    }


def profile_scaling() -> None:
    """Profile solver scaling with problem size."""
    print_section_header("SCALING ANALYSIS")

    print("\nGrid Network Scaling (increasing size):")
    print(f"{'Size':<10} {'Nodes':<8} {'Arcs':<8} {'Iters':<8} {'Time (ms)':<12} {'Iters/sec':<12}")
    print("-" * 70)

    grid_sizes = [(3, 3), (5, 5), (7, 7), (10, 10), (15, 15), (20, 20)]

    for rows, cols in grid_sizes:
        nodes, arcs = generate_grid_network(rows, cols)
        result = profile_problem(f"{rows}×{cols}", nodes, arcs)

        print(
            f"{result['name']:<10} {result['nodes']:<8} {result['arcs']:<8} "
            f"{result['iterations']:<8} {result['elapsed_ms']:<12.2f} "
            f"{result['iter_per_sec']:<12.0f}"
        )

    print("\nObservations:")
    print("  - Solve time grows roughly quadratically with problem size")
    print("  - Iteration count increases with network complexity")
    print("  - Iterations/sec decreases for larger problems (more work per iteration)")


def profile_pricing_strategies() -> None:
    """Compare Devex vs Dantzig pricing strategies."""
    print_section_header("PRICING STRATEGY COMPARISON")

    print("\nComparing Devex vs Dantzig on various problem sizes:")
    print(f"{'Problem':<20} {'Strategy':<10} {'Iters':<8} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    test_problems = [
        ("Grid 7×7", *generate_grid_network(7, 7)),
        ("Grid 10×10", *generate_grid_network(10, 10)),
        ("Bipartite 10×10", *generate_bipartite_network(10, 10)),
        ("Bipartite 15×15", *generate_bipartite_network(15, 15)),
    ]

    for problem_name, nodes, arcs in test_problems:
        # Devex (default)
        devex_opts = SolverOptions(pricing_strategy="devex")
        devex_result = profile_problem(problem_name, nodes, arcs, devex_opts)

        # Dantzig
        dantzig_opts = SolverOptions(pricing_strategy="dantzig")
        dantzig_result = profile_problem(problem_name, nodes, arcs, dantzig_opts)

        speedup = dantzig_result["elapsed_ms"] / devex_result["elapsed_ms"]

        print(
            f"{problem_name:<20} {'Devex':<10} {devex_result['iterations']:<8} "
            f"{devex_result['elapsed_ms']:<12.2f} {'1.0x':<10}"
        )
        print(
            f"{'':<20} {'Dantzig':<10} {dantzig_result['iterations']:<8} "
            f"{dantzig_result['elapsed_ms']:<12.2f} {speedup:<10.2f}x"
        )
        print()

    print("Observations:")
    print("  - Devex typically requires fewer iterations (better pivot selection)")
    print("  - Devex may have slightly higher per-iteration cost (weight updates)")
    print("  - For most problems, Devex is faster overall")
    print("  - Dantzig can be competitive on very sparse or simple problems")


def profile_solver_options() -> None:
    """Profile different solver configuration options."""
    print_section_header("SOLVER OPTIONS IMPACT")

    nodes, arcs = generate_grid_network(12, 12)

    print("\nTesting different configurations on 12×12 grid:")
    print(f"{'Configuration':<30} {'Iters':<8} {'Time (ms)':<12} {'FT Rebuilds':<12}")
    print("-" * 70)

    configs = [
        ("Default", SolverOptions()),
        ("Large FT limit (128)", SolverOptions(ft_update_limit=128)),
        ("Small FT limit (32)", SolverOptions(ft_update_limit=32)),
        ("Very small FT limit (16)", SolverOptions(ft_update_limit=16)),
        ("Large block size (200)", SolverOptions(block_size=200)),
        ("Small block size (50)", SolverOptions(block_size=50)),
    ]

    for config_name, options in configs:
        result = profile_problem(config_name, nodes, arcs, options)
        # Note: FT rebuilds not in result dict, so we show placeholder
        print(
            f"{config_name:<30} {result['iterations']:<8} {result['elapsed_ms']:<12.2f} {'N/A':<12}"
        )

    print("\nObservations:")
    print("  - ft_update_limit: Lower values = more rebuilds = more stable but slower")
    print("  - ft_update_limit: Higher values = fewer rebuilds = faster but less stable")
    print("  - block_size: Affects Devex pricing granularity")
    print("  - Default settings work well for most problems")


def profile_problem_types() -> None:
    """Profile different problem structures."""
    print_section_header("PROBLEM STRUCTURE ANALYSIS")

    print("\nComparing different network structures:")
    print(
        f"{'Problem Type':<25} {'Nodes':<8} {'Arcs':<8} {'Density':<10} {'Iters':<8} {'Time (ms)':<12}"
    )
    print("-" * 85)

    # Sparse (grid)
    grid_nodes, grid_arcs = generate_grid_network(10, 10)
    grid_result = profile_problem("Sparse (Grid 10×10)", grid_nodes, grid_arcs)
    grid_density = len(grid_arcs) / (len(grid_nodes) * len(grid_nodes))

    # Dense (bipartite)
    bip_nodes, bip_arcs = generate_bipartite_network(10, 10)
    bip_result = profile_problem("Dense (Bipartite 10×10)", bip_nodes, bip_arcs)
    bip_density = len(bip_arcs) / (len(bip_nodes) * len(bip_nodes))

    # Medium (grid with extra arcs)
    medium_nodes, medium_arcs = generate_grid_network(10, 10)
    # Add some diagonal connections
    for r in range(9):
        for c in range(9):
            medium_arcs.append(
                {
                    "tail": f"node_{r}_{c}",
                    "head": f"node_{r + 1}_{c + 1}",
                    "capacity": 50.0,
                    "cost": 1.5,
                }
            )
    medium_result = profile_problem("Medium (Grid+Diag)", medium_nodes, medium_arcs)
    medium_density = len(medium_arcs) / (len(medium_nodes) * len(medium_nodes))

    for result, density in [
        (grid_result, grid_density),
        (medium_result, medium_density),
        (bip_result, bip_density),
    ]:
        print(
            f"{result['name']:<25} {result['nodes']:<8} {result['arcs']:<8} "
            f"{density:<10.3f} {result['iterations']:<8} {result['elapsed_ms']:<12.2f}"
        )

    print("\nObservations:")
    print("  - Sparse networks: Fewer arcs → faster pivot selection")
    print("  - Dense networks: More arcs → more options but slower per iteration")
    print("  - Structure matters: Grid vs bipartite affects iteration count")


def generate_performance_summary() -> None:
    """Generate a comprehensive performance summary."""
    print_section_header("PERFORMANCE SUMMARY")

    # Quick benchmark on standard problem
    nodes, arcs = generate_grid_network(10, 10)
    result = profile_problem("Standard 10×10 Grid", nodes, arcs)

    print("\nBenchmark Problem (10×10 Grid):")
    print(f"  Nodes: {result['nodes']}")
    print(f"  Arcs: {result['arcs']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Solve time: {result['elapsed_ms']:.2f} ms")
    print(f"  Throughput: {result['iter_per_sec']:.0f} iterations/sec")

    print("\nPerformance Characteristics:")
    print("  - Small problems (<100 nodes): Solve in <10ms")
    print("  - Medium problems (100-1000 nodes): Solve in 10-100ms")
    print("  - Large problems (1000-10000 nodes): Solve in 100ms-2s")
    print("  - Very large problems (>10000 nodes): May take several seconds")

    print("\nOptimization Tips:")
    print("  1. Use Devex pricing for most problems (default)")
    print("  2. Adjust ft_update_limit if numerical issues occur")
    print("  3. For very dense networks, consider Dantzig pricing")
    print("  4. Profile your specific problem structure to tune settings")
    print("  5. Use progress callbacks to monitor long-running solves")

    print("\nWhen to Profile:")
    print("  - New problem types or structures")
    print("  - Unusually slow solve times")
    print("  - Before/after code changes (regression testing)")
    print("  - Capacity planning for production systems")


def profile_with_structured_logging() -> None:
    """Demonstrate profiling with structured logging."""
    print_section_header("STRUCTURED LOGGING FOR PROFILING")

    print("\nUsing structured logging to capture detailed metrics...")
    print("(See structured_logging_example.py for JSON formatter)")

    import logging

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    nodes, arcs = generate_grid_network(8, 8)

    print("\nSolving 8×8 grid with INFO logging:")
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    result = solve_min_cost_flow(problem)

    print(
        f"\nResult: {result.status}, objective={result.objective:.2f}, iterations={result.iterations}"
    )

    print("\nStructured log fields available:")
    print("  - nodes, arcs, max_iterations, pricing_strategy")
    print("  - iterations, total_iterations, elapsed_ms")
    print("  - objective, status, tree_arcs, nonzero_flows, ft_rebuilds")
    print("\nThese fields can be captured in JSON logs for analysis")


def main() -> None:
    """Run all performance profiling scenarios."""
    import logging

    # Suppress solver logging during profiling (except for the dedicated logging section)
    logging.getLogger("network_solver").setLevel(logging.CRITICAL)

    print_section_header("PERFORMANCE PROFILING EXAMPLES")
    print("\nComprehensive performance analysis for network flow solver")

    overall_start = time.perf_counter()

    profile_scaling()
    profile_pricing_strategies()
    profile_solver_options()
    profile_problem_types()
    generate_performance_summary()
    profile_with_structured_logging()

    overall_elapsed = time.perf_counter() - overall_start

    print_section_header("PROFILING COMPLETE")
    print(f"\nTotal profiling time: {overall_elapsed:.2f}s")
    print("\nFor more details:")
    print("  - See docs/benchmarks.md for detailed performance analysis")
    print("  - See docs/examples.md for usage examples")
    print("  - Use --verbose flag to see detailed solver logging")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
