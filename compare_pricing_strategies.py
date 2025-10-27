#!/usr/bin/env python3
"""Test script to compare different pricing strategies."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow


def test_strategy(problem_file: str, strategy_name: str) -> dict:
    """Test a pricing strategy on a problem."""
    print(f"\n{'=' * 60}")
    print(f"Testing {strategy_name} pricing strategy")
    print(f"{'=' * 60}")

    # Parse problem
    problem = parse_dimacs_file(problem_file)
    print(f"Problem: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    # Configure solver options
    from src.network_solver.data import SolverOptions

    options = SolverOptions()

    # Set pricing strategy based on name
    if strategy_name == "dantzig":
        options.pricing_strategy = "dantzig"
    elif strategy_name == "devex":
        options.pricing_strategy = "devex"
    elif strategy_name == "candidate_list":
        options.pricing_strategy = "candidate_list"
    elif strategy_name == "adaptive":
        options.pricing_strategy = "adaptive"
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Solve with timing
    start_time = time.time()
    result = solve_min_cost_flow(problem, options=options)
    elapsed = time.time() - start_time

    print(f"Status: {result.status}")
    print(f"Objective: {result.objective}")
    print(f"Iterations: {result.iterations}")
    print(f"Time: {elapsed:.2f}s")

    return {
        "strategy": strategy_name,
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "time": elapsed,
    }


def main():
    """Compare all pricing strategies."""
    if len(sys.argv) < 2:
        print("Usage: python test_pricing_strategies.py <problem_file>")
        sys.exit(1)

    problem_file = sys.argv[1]

    print(f"Comparing pricing strategies on: {problem_file}")

    strategies = ["devex", "candidate_list", "adaptive"]
    results = []

    for strategy in strategies:
        try:
            result = test_strategy(problem_file, strategy)
            results.append(result)
        except Exception as e:
            print(f"ERROR with {strategy}: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<20} {'Time (s)':<12} {'Iterations':<12} {'Objective'}")
    print(f"{'-' * 60}")

    for r in results:
        print(f"{r['strategy']:<20} {r['time']:<12.2f} {r['iterations']:<12} {r['objective']}")

    # Find best
    if results:
        best = min(results, key=lambda x: x["time"])
        print(f"\nFastest: {best['strategy']} ({best['time']:.2f}s)")

        # Calculate speedup vs devex (baseline)
        devex_result = next((r for r in results if r["strategy"] == "devex"), None)
        if devex_result and best["strategy"] != "devex":
            speedup = devex_result["time"] / best["time"]
            print(f"Speedup vs Devex: {speedup:.2f}x")


if __name__ == "__main__":
    main()
