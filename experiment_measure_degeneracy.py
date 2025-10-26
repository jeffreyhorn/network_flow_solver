#!/usr/bin/env python3
"""Measure degeneracy rate across different benchmark problems.

This script analyzes how many pivots are degenerate (theta ≈ 0) to determine
if degeneracy is a significant factor limiting performance.
"""

import time

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow


def measure_degeneracy(problem_file: str, problem_name: str):
    """Measure degeneracy rate for a problem."""
    print(f"\nAnalyzing: {problem_name}")
    print("-" * 60)

    problem = parse_dimacs_file(problem_file)

    start = time.time()
    result = solve_min_cost_flow(problem, max_iterations=10000)
    elapsed = time.time() - start

    # We need to access the simplex instance to get degeneracy stats
    # For now, we'll create a wrapper that exposes this
    # TODO: Add degeneracy stats to FlowResult or use logging

    print(f"Status: {result.status}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Iterations: {result.iterations}")

    # Note: We need to add degeneracy tracking to the result
    # For now, return basic metrics
    return {
        "name": problem_name,
        "status": result.status,
        "time": elapsed,
        "iterations": result.iterations,
    }


def main():
    """Measure degeneracy across key benchmark problems."""

    problems = [
        ("benchmarks/problems/lemon/gridgen/gridgen_8_08a.min", "gridgen_8_08a (256 nodes)"),
        ("benchmarks/problems/lemon/gridgen/gridgen_8_09a.min", "gridgen_8_09a (512 nodes)"),
        ("benchmarks/problems/lemon/goto/goto_8_08a.min", "goto_8_08a (256 nodes)"),
        ("benchmarks/problems/lemon/goto/goto_8_09a.min", "goto_8_09a (512 nodes)"),
        ("benchmarks/problems/lemon/netgen/netgen_8_08a.min", "netgen_8_08a (256 nodes)"),
    ]

    print("=" * 60)
    print("DEGENERACY ANALYSIS")
    print("=" * 60)
    print("\nMeasuring how many pivots are degenerate (theta ≈ 0)")
    print("High degeneracy means we're wasting iterations making no progress")

    results = []
    for problem_file, problem_name in problems:
        result = measure_degeneracy(problem_file, problem_name)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Problem':<40} {'Iterations':<12} {'Time'}")
    print("-" * 60)

    for r in results:
        if r["status"] == "optimal":
            print(f"{r['name']:<40} {r['iterations']:<12} {r['time']:.2f}s")
        else:
            print(f"{r['name']:<40} {r['status']:<12} {r['time']:.2f}s")

    print("\n" + "=" * 60)
    print("NOTE: To see degeneracy rates, we need to expose degenerate_pivots")
    print("in the solver result. This will be added in the next step.")
    print("=" * 60)


if __name__ == "__main__":
    main()
