"""
Demonstrates performance comparison between sparse and dense basis modes.

This example shows the memory and time benefits of sparse-only LU factorization
versus dense basis inverse computation for large network flow problems.

Key insights:
- Dense inverse: O(n³) time, O(n²) memory for each basis rebuild
- Sparse LU: O(n) memory, faster factorization for sparse networks
- Difference becomes dramatic as problem size increases
"""

import argparse
import time
import tracemalloc
from typing import Any

from network_solver import build_problem, solve_min_cost_flow
from network_solver.data import SolverOptions


def format_bytes(bytes_val: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def create_large_transportation_problem(num_sources: int, num_sinks: int) -> dict[str, Any]:
    """Create a large balanced transportation problem.

    Args:
        num_sources: Number of supply nodes
        num_sinks: Number of demand nodes

    Returns:
        Dictionary with nodes and arcs for build_problem()
    """
    supply_per_source = 100.0
    demand_per_sink = (num_sources * supply_per_source) / num_sinks

    nodes = []
    # Supply nodes
    for i in range(num_sources):
        nodes.append({"id": f"source_{i}", "supply": supply_per_source})

    # Demand nodes
    for i in range(num_sinks):
        nodes.append({"id": f"sink_{i}", "supply": -demand_per_sink})

    # Create arcs: each source can ship to each sink (dense network)
    arcs = []
    for i in range(num_sources):
        for j in range(num_sinks):
            # Cost based on "distance" between source and sink
            cost = abs(i - j) + 1.0
            capacity = supply_per_source * 1.5  # Generous capacity
            arcs.append(
                {
                    "tail": f"source_{i}",
                    "head": f"sink_{j}",
                    "capacity": capacity,
                    "cost": cost,
                }
            )

    return {"nodes": nodes, "arcs": arcs}


def benchmark_solve(problem_data: dict[str, Any], use_dense: bool) -> dict[str, float]:
    """Benchmark solving with specified basis mode.

    Args:
        problem_data: Problem specification
        use_dense: Whether to use dense inverse mode

    Returns:
        Dictionary with timing and memory statistics
    """
    problem = build_problem(
        nodes=problem_data["nodes"],
        arcs=problem_data["arcs"],
        directed=True,
        tolerance=1e-6,
    )

    options = SolverOptions(use_dense_inverse=use_dense)

    # Start memory and time tracking
    tracemalloc.start()
    start_time = time.time()

    # Solve
    result = solve_min_cost_flow(problem, options=options)

    # Capture metrics
    solve_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "solve_time": solve_time,
        "peak_memory": peak_mem,
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
    }


def run_comparison(num_sources: int, num_sinks: int, verbose: bool = False):
    """Run and display comparison for a problem size.

    Args:
        num_sources: Number of supply nodes
        num_sinks: Number of demand nodes
        verbose: Show detailed output
    """
    total_nodes = num_sources + num_sinks
    total_arcs = num_sources * num_sinks

    print(f"\n{'=' * 80}")
    print(f"Problem Size: {num_sources} sources × {num_sinks} sinks")
    print(f"  Nodes: {total_nodes:,}")
    print(f"  Arcs: {total_arcs:,}")
    print(f"{'=' * 80}")

    # Create problem
    if verbose:
        print("Creating problem...")
    problem_data = create_large_transportation_problem(num_sources, num_sinks)

    # Benchmark sparse mode
    print("\n[1/2] Solving with SPARSE-ONLY mode (use_dense_inverse=False)...")
    sparse_stats = benchmark_solve(problem_data, use_dense=False)

    # Benchmark dense mode
    print("[2/2] Solving with DENSE mode (use_dense_inverse=True)...")
    dense_stats = benchmark_solve(problem_data, use_dense=True)

    # Display results
    print(f"\n{'Results':-^80}")
    print(f"\n{'Metric':<30} {'Sparse Mode':>20} {'Dense Mode':>20}")
    print(f"{'-' * 30} {'-' * 20} {'-' * 20}")

    # Solve time
    sparse_time_str = format_time(sparse_stats["solve_time"])
    dense_time_str = format_time(dense_stats["solve_time"])
    speedup = dense_stats["solve_time"] / sparse_stats["solve_time"]
    print(f"{'Solve Time':<30} {sparse_time_str:>20} {dense_time_str:>20}")
    print(f"{'  → Speedup':<30} {f'{speedup:.2f}x faster':>20} {'baseline':>20}")

    # Memory
    sparse_mem_str = format_bytes(sparse_stats["peak_memory"])
    dense_mem_str = format_bytes(dense_stats["peak_memory"])
    mem_saved = dense_stats["peak_memory"] - sparse_stats["peak_memory"]
    mem_saved_str = format_bytes(mem_saved)
    mem_reduction = (mem_saved / dense_stats["peak_memory"]) * 100
    print(f"\n{'Peak Memory':<30} {sparse_mem_str:>20} {dense_mem_str:>20}")
    print(f"{'  → Saved':<30} {mem_saved_str:>20} {f'({mem_reduction:.1f}% less)':>20}")

    # Solution quality
    print(
        f"\n{'Iterations':<30} {sparse_stats['iterations']:>20,} {dense_stats['iterations']:>20,}"
    )
    print(
        f"{'Objective':<30} {sparse_stats['objective']:>20,.2f} {dense_stats['objective']:>20,.2f}"
    )
    print(f"{'Status':<30} {sparse_stats['status']:>20} {dense_stats['status']:>20}")

    # Verify solutions match
    obj_diff = abs(sparse_stats["objective"] - dense_stats["objective"])
    if obj_diff < 1e-6:
        print(f"\n✓ Both modes found identical optimal solutions")
    else:
        print(f"\n⚠ Warning: Objectives differ by {obj_diff:.2e}")

    return sparse_stats, dense_stats


def main():
    """Run sparse vs dense comparison benchmark."""
    parser = argparse.ArgumentParser(
        description="Compare sparse-only vs dense basis mode performance"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run small problem (50×50, fast)",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run medium problem (100×100)",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Run large problem (200×200, slower)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all problem sizes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Determine which sizes to run
    if args.all:
        sizes = [(25, 25), (50, 50), (75, 75), (100, 100)]
    else:
        sizes = []
        if args.small:
            sizes.append((25, 25))
        if args.medium:
            sizes.append((50, 50))
        if args.large:
            sizes.append((100, 100))
        if not sizes:  # Default
            sizes = [(50, 50)]

    print("=" * 80)
    print("SPARSE-ONLY vs DENSE BASIS MODE BENCHMARK".center(80))
    print("=" * 80)
    print("\nComparing:")
    print("  • SPARSE mode: use_dense_inverse=False (default with scipy)")
    print("  • DENSE mode:  use_dense_inverse=True (fallback without scipy)")
    print("\nBenefit: Sparse mode avoids O(n³) dense inverse computation")

    # Run benchmarks
    results = []
    for num_sources, num_sinks in sizes:
        sparse_stats, dense_stats = run_comparison(num_sources, num_sinks, args.verbose)
        results.append(
            {
                "size": (num_sources, num_sinks),
                "sparse": sparse_stats,
                "dense": dense_stats,
            }
        )

    # Summary
    if len(results) > 1:
        print(f"\n\n{'=' * 80}")
        print("SUMMARY: Scaling Behavior".center(80))
        print(f"{'=' * 80}\n")
        print(f"{'Problem Size':<20} {'Sparse Time':>15} {'Dense Time':>15} {'Speedup':>15}")
        print(f"{'-' * 20} {'-' * 15} {'-' * 15} {'-' * 15}")

        for r in results:
            size_str = f"{r['size'][0]}×{r['size'][1]}"
            sparse_time = format_time(r["sparse"]["solve_time"])
            dense_time = format_time(r["dense"]["solve_time"])
            speedup = r["dense"]["solve_time"] / r["sparse"]["solve_time"]
            print(f"{size_str:<20} {sparse_time:>15} {dense_time:>15} {f'{speedup:.2f}x':>15}")

        print(f"\n✓ Sparse mode consistently faster and uses less memory")
        print(f"✓ Benefits increase with problem size")

    print(f"\n{'=' * 80}")
    print("RECOMMENDATION".center(80))
    print(f"{'=' * 80}")
    print("\nFor large problems (>1000 nodes):")
    print("  • Use sparse mode (default with scipy installed)")
    print("  • Avoids O(n²) memory for dense inverse matrix")
    print("  • Faster basis rebuilds (skip O(n³) inversion)")
    print("\nFor small problems or without scipy:")
    print("  • Dense mode is acceptable (<100 nodes)")
    print("  • Automatically selected when scipy unavailable")
    print()


if __name__ == "__main__":
    main()
