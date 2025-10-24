#!/usr/bin/env python3
"""Test the performance impact of adding .copy() back to cache returns."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import Arc, NetworkProblem, Node, SolverOptions  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


def create_medium_network():
    """Create medium network flow problem."""
    num_sources = 20
    num_intermediates = 30
    num_demands = 20

    supply_per_source = 10.0
    demand_per_sink = (num_sources * supply_per_source) / num_demands

    nodes = {}
    for i in range(num_sources):
        nodes[f"S{i}"] = Node(id=f"S{i}", supply=supply_per_source)
    for i in range(num_intermediates):
        nodes[f"I{i}"] = Node(id=f"I{i}", supply=0.0)
    for i in range(num_demands):
        nodes[f"D{i}"] = Node(id=f"D{i}", supply=-demand_per_sink)

    arcs = []
    cost = 1.0
    for s in range(num_sources):
        for i in range(num_intermediates):
            arcs.append(Arc(tail=f"S{s}", head=f"I{i}", capacity=supply_per_source * 2, cost=cost))
            cost += 0.1

    for i in range(num_intermediates):
        for d in range(num_demands):
            arcs.append(Arc(tail=f"I{i}", head=f"D{d}", capacity=demand_per_sink * 2, cost=cost))
            cost += 0.1

    return NetworkProblem(directed=True, nodes=nodes, arcs=arcs, tolerance=1e-6)


def benchmark(problem, cache_size, name, num_runs=3):
    """Benchmark with multiple runs."""
    times = []

    for _ in range(num_runs):
        options = SolverOptions(projection_cache_size=cache_size, pricing_strategy="devex")
        solver = NetworkSimplex(problem, options=options)

        start = time.perf_counter()
        _result = solver.solve()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return avg_time


print("Testing .copy() performance impact on Medium Network (70 nodes, 1200 arcs)")
print("=" * 80)

problem = create_medium_network()

print("\nBaseline (cache disabled)...")
baseline = benchmark(problem, 0, "No Cache")
print(f"  Average time: {baseline:.3f}s")

print("\nOptimized cache (no .copy() on return)...")
optimized = benchmark(problem, 100, "Optimized")
print(f"  Average time: {optimized:.3f}s")
print(f"  Speedup: {baseline / optimized:.2f}x")

print("\n" + "=" * 80)
print(f"Performance improvement: {((baseline - optimized) / baseline * 100):.1f}%")
print("\nNow manually test with .copy() added back by editing basis.py line 224")
print("and re-running this script to see the impact.")
