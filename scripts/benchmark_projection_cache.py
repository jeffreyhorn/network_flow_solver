#!/usr/bin/env python3
"""Benchmark projection cache performance on network flow problems."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import solve_min_cost_flow
from network_solver.data import NetworkProblem, Node, Arc, SolverOptions


def create_network_flow_problem(num_sources, num_intermediates, num_demands):
    """Create a network flow problem with specified dimensions."""
    supply_per_source = 10.0
    demand_per_sink = (num_sources * supply_per_source) / num_demands

    nodes = {}

    # Sources
    for i in range(num_sources):
        nodes[f"S{i}"] = Node(id=f"S{i}", supply=supply_per_source)

    # Intermediate nodes
    for i in range(num_intermediates):
        nodes[f"I{i}"] = Node(id=f"I{i}", supply=0.0)

    # Demand nodes
    for i in range(num_demands):
        nodes[f"D{i}"] = Node(id=f"D{i}", supply=-demand_per_sink)

    # Create dense arcs
    arcs = []
    cost = 1.0

    # Connect sources to intermediates
    for s in range(num_sources):
        for i in range(num_intermediates):
            arcs.append(
                Arc(
                    tail=f"S{s}",
                    head=f"I{i}",
                    capacity=supply_per_source * 2,
                    cost=cost,
                )
            )
            cost += 0.1

    # Connect intermediates to demands
    for i in range(num_intermediates):
        for d in range(num_demands):
            arcs.append(
                Arc(
                    tail=f"I{i}",
                    head=f"D{d}",
                    capacity=demand_per_sink * 2,
                    cost=cost,
                )
            )
            cost += 0.1

    return NetworkProblem(
        directed=True,
        nodes=nodes,
        arcs=arcs,
        tolerance=1e-6,
    )


def benchmark_cache_settings(problem, cache_size, name):
    """Run benchmark with specific cache settings."""
    options = SolverOptions(
        projection_cache_size=cache_size,
        pricing_strategy="devex",
    )

    start_time = time.perf_counter()
    result = solve_min_cost_flow(problem, options=options)
    elapsed = time.perf_counter() - start_time

    return {
        "name": name,
        "cache_size": cache_size,
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "elapsed_time": elapsed,
    }


def main():
    """Run cache benchmarks on different problem sizes."""
    print("=" * 80)
    print("PROJECTION CACHE BENCHMARK")
    print("=" * 80)
    print()

    test_cases = [
        ("Small Network", 10, 15, 10),
        ("Medium Network", 20, 30, 20),
        ("Large Network", 40, 50, 40),
    ]

    for problem_name, num_sources, num_intermediates, num_demands in test_cases:
        print(
            f"\n{problem_name}: {num_sources} sources, {num_intermediates} intermediates, {num_demands} demands"
        )
        print("-" * 80)

        problem = create_network_flow_problem(num_sources, num_intermediates, num_demands)
        print(f"Problem: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

        # Test configurations
        configs = [
            (0, "Cache Disabled"),
            (50, "Cache Size 50"),
            (100, "Cache Size 100"),
            (200, "Cache Size 200"),
        ]

        results = []
        baseline_time = None

        for cache_size, config_name in configs:
            print(f"\n  {config_name}...")
            result = benchmark_cache_settings(problem, cache_size, config_name)
            results.append(result)

            if cache_size == 0:
                baseline_time = result["elapsed_time"]

            print(f"    Status: {result['status']}")
            print(f"    Iterations: {result['iterations']}")
            print(f"    Time: {result['elapsed_time']:.3f}s")

            if cache_size > 0 and baseline_time:
                speedup = baseline_time / result["elapsed_time"]
                time_saved = baseline_time - result["elapsed_time"]
                print(f"    Speedup: {speedup:.2f}x ({time_saved:.3f}s saved)")

        # Summary
        print(f"\n  Summary for {problem_name}:")
        print(f"  {'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Iterations':<12}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 12}")

        for result in results:
            speedup = baseline_time / result["elapsed_time"] if baseline_time else 1.0
            print(
                f"  {result['name']:<20} {result['elapsed_time']:>10.3f}  {speedup:>8.2f}x  {result['iterations']:>10}"
            )

        # Best configuration
        best = min(results, key=lambda r: r["elapsed_time"])
        if best["cache_size"] > 0:
            speedup = baseline_time / best["elapsed_time"]
            print(f"\n  Best: {best['name']} with {speedup:.2f}x speedup")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
