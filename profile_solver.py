#!/usr/bin/env python3
"""Profile the network flow solver on a benchmark problem.

Usage:
    python profile_solver.py benchmarks/problems/lemon/gridgen/gridgen_8_12a.min
"""

import cProfile
import pstats
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from src.network_solver.solver import solve_min_cost_flow  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_solver.py <problem_file>")
        sys.exit(1)

    problem_file = sys.argv[1]
    print(f"Profiling solver on: {problem_file}")

    # Parse problem
    print("Parsing problem...")
    problem = parse_dimacs_file(problem_file)
    print(f"Problem size: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    # Profile the solve
    print("\nRunning profiler...")
    profiler = cProfile.Profile()
    profiler.enable()

    result = solve_min_cost_flow(problem)

    profiler.disable()

    # Print results
    print("\nSolve completed:")
    print(f"  Status: {result.status}")
    print(f"  Objective: {result.objective}")
    print(f"  Iterations: {result.iterations}")

    # Save stats to file
    stats_file = "profile.stats"
    profiler.dump_stats(stats_file)
    print(f"\nProfile data saved to: {stats_file}")

    # Print top functions by cumulative time
    print("\n" + "=" * 80)
    print("TOP FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    # Print top functions by total time
    print("\n" + "=" * 80)
    print("TOP FUNCTIONS BY TOTAL TIME (excluding subcalls)")
    print("=" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
