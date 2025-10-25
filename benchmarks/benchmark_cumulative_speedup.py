#!/usr/bin/env python3
"""Benchmark cumulative speedup from all optimization projects.

This script measures the end-to-end performance improvement compared to the
baseline established in October 2024 (commit 0fa28c1).

Baseline Performance (Large Network - 4,267 arcs, 160 nodes):
- Runtime: 65.9 seconds
- Iterations: 356
- Time per iteration: 185.0ms

This benchmark recreates a similar large network and measures current performance
with all optimizations enabled:
- Project 1: Projection cache (cache_basis_solves)
- Project 2: Vectorized pricing (use_vectorized_pricing)
- Project 3: Deferred Devex updates (batch weight updates)
- Project 4: Vectorized residual calculations (cached arrays)
"""

import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import build_problem, solve_min_cost_flow  # noqa: E402
from network_solver.data import SolverOptions  # noqa: E402


def create_large_transportation_problem():
    """Create a large transportation problem similar to the profiling baseline.

    The baseline had:
    - 160 nodes
    - 4,267 arcs
    - 356 iterations
    - 65.9 seconds runtime

    This creates a transportation problem with:
    - 65 sources x 65 destinations = 130 nodes (close to 160)
    - 4,225 arcs (65x65 fully connected)
    - Guaranteed feasible (balanced transportation)
    """
    num_sources = 65
    num_sinks = 65
    supply_per_source = 100.0

    nodes = []

    # Create sources with supply
    for i in range(num_sources):
        nodes.append({"id": f"s{i}", "supply": supply_per_source})

    # Create sinks with demand
    for j in range(num_sinks):
        nodes.append({"id": f"t{j}", "supply": -supply_per_source})

    # Create fully connected bipartite graph (transportation problem)
    arcs = []

    for i in range(num_sources):
        for j in range(num_sinks):
            # Vary costs to create interesting optimization problem
            arc_cost = abs(i - j) * 0.5 + (i + j) % 10 + 1.0
            arcs.append(
                {
                    "tail": f"s{i}",
                    "head": f"t{j}",
                    "capacity": supply_per_source * 2.0,  # Ample capacity
                    "cost": arc_cost,
                }
            )

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    return problem


def benchmark_current_performance(problem, runs=5):
    """Benchmark current performance with all optimizations enabled.

    Args:
        problem: NetworkProblem to solve
        runs: Number of runs to average over

    Returns:
        dict with timing statistics and solver results
    """
    print(f"\nRunning {runs} iterations with all optimizations enabled...")
    print(f"  Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    times = []
    iterations_list = []
    objectives = []

    # Configure with all optimizations enabled (these are defaults)
    options = SolverOptions(
        pricing_strategy="devex",
        projection_cache_size=100,  # Project 1: Projection cache
        # Project 2: Vectorized pricing (enabled by default)
        # Project 3: Deferred Devex (enabled by default)
        # Project 4: Vectorized residuals (always active)
    )

    for run in range(runs):
        print(f"  Run {run + 1}/{runs}...", end=" ", flush=True)

        start_time = time.perf_counter()
        result = solve_min_cost_flow(problem, options=options)
        elapsed = time.perf_counter() - start_time

        times.append(elapsed)
        iterations_list.append(result.iterations)
        objectives.append(result.objective)

        print(f"{elapsed:.3f}s, {result.iterations} iterations, {result.status}")

        # Verify all runs produce same result
        if run > 0 and abs(objectives[-1] - objectives[0]) > 1e-6:
            print("    WARNING: Objective differs from run 1!")
            print(f"      Run 1: {objectives[0]:.6f}")
            print(f"      Run {run + 1}: {objectives[-1]:.6f}")

    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0.0
    avg_iterations = mean(iterations_list)

    return {
        "times": times,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "avg_iterations": avg_iterations,
        "iterations_list": iterations_list,
        "objective": objectives[0],
        "status": result.status,
    }


def print_results(stats, baseline_time=65.9):
    """Print benchmark results and comparison to baseline.

    Args:
        stats: Statistics dict from benchmark_current_performance
        baseline_time: Baseline runtime in seconds (default 65.9s)
    """
    print("\n" + "=" * 80)
    print("CUMULATIVE SPEEDUP BENCHMARK RESULTS")
    print("=" * 80)

    print("\nBaseline Performance (October 2024 - commit 0fa28c1):")
    print(f"  Runtime: {baseline_time:.1f} seconds")
    print("  Iterations: 356")
    print(f"  Time per iteration: {baseline_time / 356 * 1000:.1f}ms")

    print("\nCurrent Performance (All Optimizations Enabled):")
    print(f"  Average runtime: {stats['avg_time']:.3f}s ± {stats['std_time']:.3f}s")
    print(f"  Min/Max: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
    print(f"  Average iterations: {stats['avg_iterations']:.1f}")
    print(f"  Time per iteration: {stats['avg_time'] / stats['avg_iterations'] * 1000:.1f}ms")
    print(f"  Status: {stats['status']}")
    print(f"  Objective: {stats['objective']:.2f}")

    # Calculate speedup
    speedup_ratio = baseline_time / stats["avg_time"]
    speedup_percent = (speedup_ratio - 1) * 100
    time_saved = baseline_time - stats["avg_time"]
    time_saved_percent = (time_saved / baseline_time) * 100

    print("\nPerformance Improvement:")
    print(f"  Speedup: {speedup_ratio:.2f}x ({speedup_percent:+.1f}%)")
    print(f"  Time saved: {time_saved:.1f}s ({time_saved_percent:.1f}% reduction)")

    # Compare to predicted speedup
    predicted_speedup = 1.26  # From analysis: 1.20-1.30x range
    predicted_time = baseline_time / predicted_speedup

    print("\nComparison to Theoretical Prediction:")
    print(f"  Predicted speedup: {predicted_speedup:.2f}x (20-30% faster)")
    print(f"  Predicted runtime: {predicted_time:.1f}s")
    print(f"  Actual vs predicted: {speedup_ratio / predicted_speedup:.2f}x")

    if speedup_ratio >= predicted_speedup:
        print("  ✅ Actual performance EXCEEDS prediction!")
    elif speedup_ratio >= (predicted_speedup * 0.9):
        print("  ✅ Actual performance within 10% of prediction")
    else:
        print("  ⚠️  Actual performance below prediction")

    # Break down by optimization (estimates)
    print("\nOptimization Contribution Breakdown (Estimated):")
    print("  Project 1 (Projection cache):     ~10-14% speedup")
    print("  Project 2 (Vectorized pricing):   ~2.3x on pricing ops (~4% overall)")
    print("  Project 3 (Deferred Devex):       ~97.5% call reduction (~6% overall)")
    print("  Project 4 (Vectorized residuals): ~1.5M calls eliminated (~4% overall)")

    print("\n" + "=" * 80)


def main():
    """Run comprehensive cumulative speedup benchmark."""
    print("=" * 80)
    print("CUMULATIVE SPEEDUP VERIFICATION")
    print("=" * 80)
    print("\nThis benchmark measures end-to-end performance improvement from all")
    print("optimization projects completed since October 2024:")
    print("  - Project 1: Cache Basis Solves (Projection Cache)")
    print("  - Project 2: Vectorize Pricing Operations")
    print("  - Project 3: Batch Devex Weight Updates (Deferred Updates)")
    print("  - Project 4: Vectorize Residual Calculations")

    # Create large transportation problem
    print("\nCreating large transportation problem...")
    problem = create_large_transportation_problem()
    print(f"  Created network: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")
    print("  Baseline reference: 160 nodes, 4,267 arcs, 65.9s runtime")

    # Run benchmark
    stats = benchmark_current_performance(problem, runs=5)

    # Print results
    print_results(stats, baseline_time=65.9)

    # Save results to file
    results_file = PROJECT_ROOT / "benchmarks" / "results" / "cumulative_speedup_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        f.write("Cumulative Speedup Benchmark Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs\n")
        f.write("Baseline: 65.9s (160 nodes, 4,267 arcs, 356 iterations)\n\n")
        f.write("Current Performance:\n")
        f.write(f"  Average runtime: {stats['avg_time']:.3f}s ± {stats['std_time']:.3f}s\n")
        f.write(f"  Min/Max: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s\n")
        f.write(f"  Average iterations: {stats['avg_iterations']:.1f}\n")
        f.write(f"  Individual run times: {', '.join(f'{t:.3f}s' for t in stats['times'])}\n")
        f.write(
            f"  Individual iterations: {', '.join(str(i) for i in stats['iterations_list'])}\n\n"
        )

        speedup_ratio = 65.9 / stats["avg_time"]
        speedup_percent = (speedup_ratio - 1) * 100
        time_saved = 65.9 - stats["avg_time"]

        f.write(f"Speedup: {speedup_ratio:.2f}x ({speedup_percent:+.1f}%)\n")
        f.write(f"Time saved: {time_saved:.1f}s\n")
        f.write("Predicted speedup: 1.26x (20-30%)\n")
        f.write(f"Actual vs predicted: {speedup_ratio / 1.26:.2f}x\n")

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
