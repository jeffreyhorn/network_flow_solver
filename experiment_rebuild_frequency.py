#!/usr/bin/env python3
"""Test script to measure rebuild frequency with different parameter configurations.

This script runs the solver on a test problem with various parameter settings
to find the optimal balance between rebuild frequency and performance.
"""

import time

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.data import SolverOptions
from src.network_solver.solver import solve_min_cost_flow


def test_configuration(problem_file: str, options: SolverOptions, config_name: str):
    """Test a single configuration and report metrics."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 70}")

    # Parse problem
    problem = parse_dimacs_file(problem_file)

    # Track rebuild count via progress callback
    rebuild_count = [0]

    def progress_callback(info):
        # Track the latest rebuild count from progress info
        if hasattr(info, "ft_rebuilds"):
            rebuild_count[0] = info.ft_rebuilds

    # Solve with timing
    start_time = time.time()
    result = solve_min_cost_flow(
        problem,
        options=options,
        max_iterations=10000,
        progress_callback=progress_callback,
        progress_interval=100,  # Check periodically
    )
    elapsed = time.time() - start_time

    # Calculate rebuild frequency
    rebuild_freq = (rebuild_count[0] / result.iterations * 100) if result.iterations > 0 else 0

    print(f"Status: {result.status}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Iterations: {result.iterations}")
    print(f"Rebuilds: {rebuild_count[0]} ({rebuild_freq:.1f}%)")

    if result.status == "optimal":
        print(f"Objective: {result.objective:.2f}")

    return {
        "config": config_name,
        "status": result.status,
        "time": elapsed,
        "iterations": result.iterations,
        "rebuilds": rebuild_count[0],
        "rebuild_freq": rebuild_freq,
        "objective": result.objective if result.status == "optimal" else None,
    }


def main():
    """Run experiments with different configurations."""
    problem_file = "benchmarks/problems/lemon/gridgen/gridgen_8_09a.min"

    # Current baseline configuration
    configs = [
        (
            "Baseline (current)",
            SolverOptions(
                ft_update_limit=64,
                condition_number_threshold=1e12,
                condition_check_interval=10,
            ),
        ),
        (
            "Higher FT limit (100)",
            SolverOptions(
                ft_update_limit=100,
                condition_number_threshold=1e12,
                condition_check_interval=10,
            ),
        ),
        (
            "Higher FT limit (150)",
            SolverOptions(
                ft_update_limit=150,
                condition_number_threshold=1e12,
                condition_check_interval=10,
            ),
        ),
        (
            "Higher condition threshold (1e14)",
            SolverOptions(
                ft_update_limit=64,
                condition_number_threshold=1e14,
                condition_check_interval=10,
            ),
        ),
        (
            "Less frequent condition checks (20)",
            SolverOptions(
                ft_update_limit=64,
                condition_number_threshold=1e12,
                condition_check_interval=20,
            ),
        ),
        (
            "Combined: FT=100, threshold=1e14",
            SolverOptions(
                ft_update_limit=100,
                condition_number_threshold=1e14,
                condition_check_interval=10,
            ),
        ),
        (
            "Aggressive: FT=150, threshold=1e14, interval=20",
            SolverOptions(
                ft_update_limit=150,
                condition_number_threshold=1e14,
                condition_check_interval=20,
            ),
        ),
    ]

    print("=" * 70)
    print("REBUILD FREQUENCY OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"Test problem: {problem_file}")
    print()

    results = []
    for config_name, options in configs:
        result = test_configuration(problem_file, options, config_name)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Configuration':<40} {'Time':<8} {'Speedup':<8} {'Iters':<7} {'Rebuilds':<10} {'Freq %'}"
    )
    print("-" * 70)

    baseline_time = results[0]["time"]
    for r in results:
        speedup_str = f"{baseline_time / r['time']:.2f}x" if r["time"] > 0 else "-"
        print(
            f"{r['config']:<40} {r['time']:<8.2f} {speedup_str:<8} {r['iterations']:<7} {r['rebuilds']:<10} {r['rebuild_freq']:.1f}%"
        )

    # Find best configuration
    print("\n" + "=" * 70)
    best = min(results, key=lambda x: x["time"])
    print(f"Best configuration: {best['config']}")
    print(f"Time: {best['time']:.2f}s ({baseline_time / best['time']:.2f}x speedup)")
    print(f"Rebuild frequency: {best['rebuild_freq']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
