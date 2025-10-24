#!/usr/bin/env python3
"""Benchmark vectorized pricing performance improvement."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import solve_min_cost_flow  # noqa: E402
from network_solver.data import Arc, NetworkProblem, Node, SolverOptions  # noqa: E402


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


def benchmark_with_vectorization(problem, enabled=True, runs=3):
    """Run benchmark with vectorization enabled or disabled."""
    from network_solver.simplex_pricing import DevexPricing

    # Temporarily modify DevexPricing to control vectorization
    original_method = DevexPricing.select_entering_arc

    if not enabled:
        # Override to skip vectorization check
        def select_entering_arc_no_vec(
            self, arcs, basis, actual_arc_count, allow_zero, tolerance, solver=None
        ):
            # Force solver=None to disable vectorization
            return original_method(
                self, arcs, basis, actual_arc_count, allow_zero, tolerance, solver=None
            )

        DevexPricing.select_entering_arc = select_entering_arc_no_vec

    times = []
    result = None

    try:
        for _ in range(runs):
            options = SolverOptions(pricing_strategy="devex")
            start_time = time.perf_counter()
            result = solve_min_cost_flow(problem, options=options)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
    finally:
        # Restore original method
        if not enabled:
            DevexPricing.select_entering_arc = original_method

    avg_time = sum(times) / len(times)
    return avg_time, result


def main():
    print("=" * 70)
    print("Vectorized Pricing Performance Benchmark")
    print("=" * 70)

    # Test different problem sizes
    test_configs = [
        (10, 15, 10, "Small"),
        (15, 20, 15, "Medium"),
    ]

    for sources, intermediates, demands, size_name in test_configs:
        print(f"\n{size_name} Problem ({sources}x{intermediates}x{demands}):")
        print("-" * 70)

        problem = create_network_flow_problem(sources, intermediates, demands)
        num_nodes = len(problem.nodes)
        num_arcs = len(problem.arcs)
        print(f"  Nodes: {num_nodes}, Arcs: {num_arcs}")

        # Benchmark without vectorization (loop-based)
        print("\n  Testing loop-based pricing (baseline)...")
        time_baseline, result_baseline = benchmark_with_vectorization(
            problem, enabled=False, runs=3
        )
        print(f"    Average time: {time_baseline:.3f}s")
        print(f"    Iterations: {result_baseline.iterations}")
        print(f"    Status: {result_baseline.status}")

        # Benchmark with vectorization
        print("\n  Testing vectorized pricing...")
        time_vectorized, result_vectorized = benchmark_with_vectorization(
            problem, enabled=True, runs=3
        )
        print(f"    Average time: {time_vectorized:.3f}s")
        print(f"    Iterations: {result_vectorized.iterations}")
        print(f"    Status: {result_vectorized.status}")

        # Calculate improvement
        speedup = (time_baseline / time_vectorized - 1) * 100
        time_saved = time_baseline - time_vectorized

        print("\n  Performance Improvement:")
        print(f"    Speedup: {speedup:+.1f}%")
        print(f"    Time saved: {time_saved:.3f}s")

        # Verify results match
        if abs(result_baseline.objective - result_vectorized.objective) < 1e-6:
            print(f"    ✓ Results verified (objective: {result_vectorized.objective:.2f})")
        else:
            print("    ✗ WARNING: Results differ!")
            print(f"      Baseline: {result_baseline.objective:.6f}")
            print(f"      Vectorized: {result_vectorized.objective:.6f}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
