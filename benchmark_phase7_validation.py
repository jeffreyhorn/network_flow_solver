#!/usr/bin/env python3
"""Benchmark Phase 7 sparse matrix implementation for performance and memory impact.

This script validates the Phase 7 memory optimization by measuring
memory usage and performance with the sparse matrix implementation.

Expected outcomes:
- Memory: 80% reduction in peak memory (1.02 GB → ~200 MB)
- Performance: Neutral to 1.1-1.2x speedup from cache effects
- Correctness: Identical solutions
"""

import sys
import time
import tracemalloc
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402
from network_solver.data import build_problem  # noqa: E402


def format_bytes(bytes_val):
    """Format bytes as human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def benchmark_with_memory_tracking(problem, label, runs=3):
    """Run benchmark with memory tracking."""
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    times = []
    peak_memories = []
    iterations_list = []
    objectives = []

    for run in range(runs):
        print(f"\n  Run {run + 1}/{runs}...")

        tracemalloc.start()

        if run == 0:
            print("    Warming up JIT...")
            _ = solve_min_cost_flow(problem)
            tracemalloc.stop()
            tracemalloc.start()

        start_time = time.time()
        result = solve_min_cost_flow(problem)
        elapsed = time.time() - start_time

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        peak_memories.append(peak_mem)
        iterations_list.append(result.iterations)
        objectives.append(result.objective)

        print(f"    Time: {elapsed:.3f}s")
        print(f"    Peak memory: {format_bytes(peak_mem)}")
        print(f"    Iterations: {result.iterations}")

    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0
    avg_memory = mean(peak_memories)

    print(f"\n  Summary ({runs} runs):")
    print(f"    Time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"    Peak memory: {format_bytes(avg_memory)}")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "avg_memory": avg_memory,
        "objective": objectives[0],
    }


def main():
    """Run Phase 7 validation benchmarks."""
    print("=" * 70)
    print("PHASE 7 VALIDATION: Sparse Matrix Performance & Memory Benchmark")
    print("=" * 70)

    test_problems = [
        "benchmarks/problems/lemon/gridgen/gridgen_8_08a.min",
        "benchmarks/problems/lemon/gridgen/gridgen_8_12a.min",
    ]

    results = {}

    for problem_file in test_problems:
        problem_path = PROJECT_ROOT / problem_file

        if not problem_path.exists():
            print(f"\nSkipping {problem_file} (not found)")
            continue

        print(f"\n{'#' * 70}")
        print(f"Problem: {problem_file}")
        print(f"{'#' * 70}")

        print(f"\nParsing {problem_file}...")
        problem = parse_dimacs_file(str(problem_path))

        print(f"  Nodes: {len(problem.nodes):,}")
        print(f"  Arcs: {len(problem.arcs):,}")

        result = benchmark_with_memory_tracking(
            problem, "SPARSE MATRIX IMPLEMENTATION (Phase 7)", runs=3
        )

        results[problem_file] = result

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for problem_file, result in results.items():
        print(f"\n{problem_file}:")
        print(f"  Time: {result['avg_time']:.3f}s ± {result['std_time']:.3f}s")
        print(f"  Peak memory: {format_bytes(result['avg_memory'])}")
        print(f"  Objective: {result['objective']:.2f}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nTo compare against pre-Phase-7 baseline:")
    print("  1. git checkout 40404bb  # Before Phase 7")
    print("  2. python benchmark_phase7_validation.py > baseline.txt")
    print("  3. git checkout main")
    print("  4. python benchmark_phase7_validation.py > phase7.txt")
    print("  5. Compare results")


if __name__ == "__main__":
    main()
