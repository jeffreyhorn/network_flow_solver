#!/usr/bin/env python3
"""Profile the solver to identify performance bottlenecks.

This script runs the solver on a benchmark problem with Python's cProfile
to identify which functions are consuming the most time.

Usage:
    python profile_solver.py [problem_file]

Example:
    python profile_solver.py benchmarks/problems/lemon/gridgen/gridgen_8_09a.min
"""

import cProfile
import pstats
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmarks.parsers.dimacs import parse_dimacs_file
from network_solver.solver import solve_min_cost_flow


def profile_solve(problem_file: str):
    """Profile solver on a single problem."""
    print(f"Parsing problem: {problem_file}")
    problem = parse_dimacs_file(problem_file)

    print(f"Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")
    print("\nStarting profiled solve...")
    print("=" * 70)

    # Profile the solve
    profiler = cProfile.Profile()
    profiler.enable()

    result = solve_min_cost_flow(problem, max_iterations=10000)

    profiler.disable()

    print("=" * 70)
    print(f"\nSolver finished: {result.status}")
    print(f"Objective: {result.objective}")
    print(f"Iterations: {result.iterations}")
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)

    # Print statistics
    stats = pstats.Stats(profiler)

    print("\n--- Top 30 functions by cumulative time ---")
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("\n--- Top 30 functions by total time (excluding subcalls) ---")
    stats.sort_stats("tottime")
    stats.print_stats(30)

    print("\n--- Functions called most frequently ---")
    stats.sort_stats("ncalls")
    stats.print_stats(30)

    # Save detailed stats to file
    stats_file = "profile_stats.txt"
    with open(stats_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        f.write("=" * 80 + "\n")
        f.write("CUMULATIVE TIME (includes subcalls)\n")
        f.write("=" * 80 + "\n")
        stats.sort_stats("cumulative")
        stats.print_stats()

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOTAL TIME (excludes subcalls - shows pure function time)\n")
        f.write("=" * 80 + "\n")
        stats.sort_stats("tottime")
        stats.print_stats()

        f.write("\n" + "=" * 80 + "\n")
        f.write("CALL COUNT\n")
        f.write("=" * 80 + "\n")
        stats.sort_stats("ncalls")
        stats.print_stats()

    print(f"\nDetailed statistics saved to: {stats_file}")

    return result, stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_solver.py <problem_file>")
        print("\nExample:")
        print("  python profile_solver.py benchmarks/problems/lemon/gridgen/gridgen_8_09a.min")
        sys.exit(1)

    problem_file = sys.argv[1]
    if not Path(problem_file).exists():
        print(f"Error: File not found: {problem_file}")
        sys.exit(1)

    result, stats = profile_solve(problem_file)
