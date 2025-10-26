#!/usr/bin/env python3
"""Simple test script to measure performance with different rebuild parameters."""

import time

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.data import SolverOptions
from src.network_solver.solver import solve_min_cost_flow


def test_config(problem_file: str, config_name: str, **options):
    """Test a configuration and return results."""
    print(f"\nTesting: {config_name}")

    problem = parse_dimacs_file(problem_file)
    solver_options = SolverOptions(**options)

    start = time.time()
    result = solve_min_cost_flow(problem, options=solver_options, max_iterations=10000)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.2f}s, Iterations: {result.iterations}, Status: {result.status}")

    return {
        "name": config_name,
        "time": elapsed,
        "iterations": result.iterations,
        "status": result.status,
    }


def main():
    problem = "benchmarks/problems/lemon/gridgen/gridgen_8_09a.min"

    print("=" * 70)
    print("REBUILD FREQUENCY OPTIMIZATION EXPERIMENTS")
    print("=" * 70)

    configs = [
        (
            "Baseline (FT=64, threshold=1e12, interval=10)",
            {
                "ft_update_limit": 64,
                "condition_number_threshold": 1e12,
                "condition_check_interval": 10,
            },
        ),
        (
            "Higher FT limit (100)",
            {
                "ft_update_limit": 100,
                "condition_number_threshold": 1e12,
                "condition_check_interval": 10,
            },
        ),
        (
            "Higher FT limit (150)",
            {
                "ft_update_limit": 150,
                "condition_number_threshold": 1e12,
                "condition_check_interval": 10,
            },
        ),
        (
            "Higher threshold (1e14)",
            {
                "ft_update_limit": 64,
                "condition_number_threshold": 1e14,
                "condition_check_interval": 10,
            },
        ),
        (
            "Less frequent checks (interval=20)",
            {
                "ft_update_limit": 64,
                "condition_number_threshold": 1e12,
                "condition_check_interval": 20,
            },
        ),
        (
            "Less frequent checks (interval=50)",
            {
                "ft_update_limit": 64,
                "condition_number_threshold": 1e12,
                "condition_check_interval": 50,
            },
        ),
        (
            "Combined: FT=100, threshold=1e14",
            {
                "ft_update_limit": 100,
                "condition_number_threshold": 1e14,
                "condition_check_interval": 10,
            },
        ),
        (
            "Aggressive: FT=150, threshold=1e14, interval=20",
            {
                "ft_update_limit": 150,
                "condition_number_threshold": 1e14,
                "condition_check_interval": 20,
            },
        ),
    ]

    results = []
    for name, opts in configs:
        result = test_config(problem, name, **opts)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<50} {'Time':<10} {'Speedup'}")
    print("-" * 70)

    baseline_time = results[0]["time"]
    for r in results:
        speedup = f"{baseline_time / r['time']:.3f}x" if r["time"] > 0 else "-"
        print(f"{r['name']:<50} {r['time']:<10.2f} {speedup}")

    # Best
    print("\n" + "=" * 70)
    best = min(results, key=lambda x: x["time"])
    print(f"Best: {best['name']}")
    print(f"Time: {best['time']:.2f}s (speedup: {baseline_time / best['time']:.3f}x)")
    print("=" * 70)


if __name__ == "__main__":
    main()
