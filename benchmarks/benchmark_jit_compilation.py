#!/usr/bin/env python3
"""Benchmark JIT compilation speedup for Forrest-Tomlin operations.

This script measures the performance impact of Numba JIT compilation on
the Forrest-Tomlin update loop, which is the primary computational bottleneck
(49.7% of runtime in baseline profiling).

Expected impact: 20-40% speedup on Forrest-Tomlin solve operations.
"""

import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import build_problem, solve_min_cost_flow  # noqa: E402
from network_solver.data import SolverOptions  # noqa: E402

# Check if Numba is available
try:
    import numba

    HAS_NUMBA = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    HAS_NUMBA = False
    NUMBA_VERSION = "not installed"


def create_large_transportation_problem():
    """Create a large transportation problem to stress Forrest-Tomlin operations.

    Returns a problem similar to the baseline profiling (4,267 arcs, 160 nodes)
    that will accumulate many Forrest-Tomlin updates.
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
                    "capacity": supply_per_source * 2.0,
                    "cost": arc_cost,
                }
            )

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    return problem


def benchmark_with_jit(problem, use_jit, runs=3):
    """Benchmark solver with or without JIT compilation.

    Args:
        problem: NetworkProblem to solve
        use_jit: Whether to enable JIT compilation
        runs: Number of runs to average over

    Returns:
        dict with timing statistics
    """
    times = []
    iterations_list = []

    options = SolverOptions(
        pricing_strategy="devex",
        use_jit=use_jit,
    )

    for run in range(runs):
        start_time = time.perf_counter()
        result = solve_min_cost_flow(problem, options=options)
        elapsed = time.perf_counter() - start_time

        times.append(elapsed)
        iterations_list.append(result.iterations)

        if run == 0:
            status = result.status
            objective = result.objective

    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0.0

    return {
        "times": times,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "avg_iterations": mean(iterations_list),
        "status": status,
        "objective": objective,
    }


def main():
    """Run JIT compilation benchmark."""
    print("=" * 80)
    print("JIT COMPILATION BENCHMARK - Forrest-Tomlin Operations")
    print("=" * 80)

    print(f"\nNumba status: {NUMBA_VERSION}")

    if not HAS_NUMBA:
        print("\n⚠️  Numba is not installed!")
        print("   Install with: pip install 'network-flow-solver[jit]'")
        print("   or: pip install numba>=0.58.0")
        print("\nRunning benchmark with Python-only implementation...")

    # Create large problem
    print("\nCreating large transportation problem...")
    problem = create_large_transportation_problem()
    print(f"  Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")
    print("  Expected FT updates: ~100-200 per basis rebuild")

    # Benchmark without JIT
    print("\n" + "-" * 80)
    print("Benchmarking WITHOUT JIT compilation (pure Python)...")
    print("-" * 80)
    stats_no_jit = benchmark_with_jit(problem, use_jit=False, runs=3)
    print(f"  Average time: {stats_no_jit['avg_time']:.3f}s ± {stats_no_jit['std_time']:.3f}s")
    print(f"  Min/Max: {stats_no_jit['min_time']:.3f}s / {stats_no_jit['max_time']:.3f}s")
    print(f"  Iterations: {stats_no_jit['avg_iterations']:.0f}")
    print(f"  Status: {stats_no_jit['status']}")

    if HAS_NUMBA:
        # Benchmark with JIT
        print("\n" + "-" * 80)
        print("Benchmarking WITH JIT compilation (Numba)...")
        print("-" * 80)
        print("  Note: First run includes JIT compilation overhead")
        stats_with_jit = benchmark_with_jit(problem, use_jit=True, runs=3)
        print(
            f"  Average time: {stats_with_jit['avg_time']:.3f}s ± {stats_with_jit['std_time']:.3f}s"
        )
        print(f"  Min/Max: {stats_with_jit['min_time']:.3f}s / {stats_with_jit['max_time']:.3f}s")
        print(f"  Iterations: {stats_with_jit['avg_iterations']:.0f}")
        print(f"  Status: {stats_with_jit['status']}")

        # Calculate speedup
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        speedup = stats_no_jit["avg_time"] / stats_with_jit["avg_time"]
        speedup_percent = (speedup - 1) * 100
        time_saved = stats_no_jit["avg_time"] - stats_with_jit["avg_time"]

        print("\nJIT Compilation Impact:")
        print(f"  Speedup: {speedup:.2f}x ({speedup_percent:+.1f}%)")
        print(f"  Time saved: {time_saved:.3f}s per solve")

        # Verify results match
        if abs(stats_no_jit["objective"] - stats_with_jit["objective"]) < 1e-6:
            print(f"  ✓ Results verified (objective: {stats_with_jit['objective']:.2f})")
        else:
            print("  ✗ WARNING: Results differ!")
            print(f"    No JIT: {stats_no_jit['objective']:.6f}")
            print(f"    With JIT: {stats_with_jit['objective']:.6f}")

        # Save results
        results_file = PROJECT_ROOT / "benchmarks" / "results" / "jit_compilation_results.txt"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            f.write("JIT Compilation Benchmark Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Numba version: {NUMBA_VERSION}\n")
            f.write(f"Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs\n\n")
            f.write("Without JIT:\n")
            f.write(
                f"  Average time: {stats_no_jit['avg_time']:.3f}s ± {stats_no_jit['std_time']:.3f}s\n"
            )
            f.write(f"  Iterations: {stats_no_jit['avg_iterations']:.0f}\n\n")
            f.write("With JIT:\n")
            f.write(
                f"  Average time: {stats_with_jit['avg_time']:.3f}s ± {stats_with_jit['std_time']:.3f}s\n"
            )
            f.write(f"  Iterations: {stats_with_jit['avg_iterations']:.0f}\n\n")
            f.write(f"Speedup: {speedup:.2f}x ({speedup_percent:+.1f}%)\n")
            f.write(f"Time saved: {time_saved:.3f}s per solve\n")

        print(f"\nResults saved to: {results_file}")
    else:
        print("\n" + "=" * 80)
        print("Skipping JIT benchmark (Numba not installed)")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
