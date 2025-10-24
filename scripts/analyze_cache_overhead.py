#!/usr/bin/env python3
"""Analyze cache overhead and hit rates to understand performance impact."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import solve_min_cost_flow  # noqa: E402
from network_solver.data import Arc, NetworkProblem, Node, SolverOptions  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


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


def analyze_problem(problem, cache_size, name):
    """Analyze cache performance for a problem."""
    options = SolverOptions(
        projection_cache_size=cache_size,
        pricing_strategy="devex",
    )

    solver = NetworkSimplex(problem, options=options)

    start_time = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start_time

    total_requests = solver.basis.cache_hits + solver.basis.cache_misses
    hit_rate = solver.basis.cache_hits / total_requests if total_requests > 0 else 0.0

    return {
        "name": name,
        "cache_size": cache_size,
        "elapsed": elapsed,
        "iterations": result.iterations,
        "total_requests": total_requests,
        "cache_hits": solver.basis.cache_hits,
        "cache_misses": solver.basis.cache_misses,
        "hit_rate": hit_rate,
        "cache_final_size": len(solver.basis.projection_cache),
    }


def main():
    """Analyze cache overhead."""
    print("=" * 80)
    print("CACHE OVERHEAD ANALYSIS")
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

        # Test with cache disabled
        no_cache = analyze_problem(problem, 0, "No Cache")

        # Test with cache enabled
        with_cache = analyze_problem(problem, 100, "Cache Size 100")

        # Calculate overhead
        overhead_seconds = with_cache["elapsed"] - no_cache["elapsed"]
        overhead_percent = (overhead_seconds / no_cache["elapsed"]) * 100

        print(f"\n  No Cache:")
        print(f"    Time: {no_cache['elapsed']:.3f}s")
        print(f"    Iterations: {no_cache['iterations']}")
        print(f"    Total projection requests: {no_cache['total_requests']:,}")

        print(f"\n  With Cache (size=100):")
        print(f"    Time: {with_cache['elapsed']:.3f}s")
        print(f"    Iterations: {with_cache['iterations']}")
        print(f"    Total projection requests: {with_cache['total_requests']:,}")
        print(f"    Cache hits: {with_cache['cache_hits']:,}")
        print(f"    Cache misses: {with_cache['cache_misses']:,}")
        print(f"    Hit rate: {with_cache['hit_rate']:.1%}")
        print(f"    Final cache size: {with_cache['cache_final_size']}")

        print(f"\n  Cache Impact:")
        if overhead_seconds > 0:
            print(f"    Overhead: +{overhead_seconds:.3f}s (+{overhead_percent:.1f}%)")
            print(f"    Speedup: {no_cache['elapsed'] / with_cache['elapsed']:.2f}x")
        else:
            speedup_actual = no_cache["elapsed"] / with_cache["elapsed"]
            time_saved = -overhead_seconds
            print(f"    Time saved: {time_saved:.3f}s ({-overhead_percent:.1f}%)")
            print(f"    Speedup: {speedup_actual:.2f}x")

        # Calculate overhead per cache operation
        if with_cache["cache_hits"] > 0:
            overhead_per_hit = (
                overhead_seconds / with_cache["cache_hits"]
            ) * 1_000_000  # microseconds
            print(f"    Overhead per cache hit: {overhead_per_hit:.1f} μs")

        # Analyze if cache is beneficial
        if with_cache["hit_rate"] > 0:
            print(f"\n  Analysis:")
            if overhead_seconds > 0:
                print(f"    ❌ Cache adds overhead despite {with_cache['hit_rate']:.1%} hit rate")
                print(f"    Reason: Cache operations cost more than projection recomputation")
            else:
                print(f"    ✅ Cache beneficial with {with_cache['hit_rate']:.1%} hit rate")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
