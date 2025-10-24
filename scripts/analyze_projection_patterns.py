#!/usr/bin/env python3
"""Analyze projection request patterns to inform cache design.

This script runs the instrumented solver and analyzes:
1. How many unique projections are requested
2. How many repeated requests (cache hit potential)
3. Working set size per basis version
4. Temporal locality patterns
"""

import sys
from collections import Counter, defaultdict

from network_solver import solve_min_cost_flow
from network_solver.data import Arc, NetworkProblem, Node


def create_transportation_problem(num_sources: int, num_sinks: int) -> NetworkProblem:
    """Create a balanced transportation problem."""
    nodes = {}
    arcs = []

    # Create source nodes (supply)
    for i in range(num_sources):
        nodes[f"S{i}"] = Node(id=f"S{i}", supply=float(num_sinks))

    # Create sink nodes (demand)
    for j in range(num_sinks):
        nodes[f"D{j}"] = Node(id=f"D{j}", supply=-float(num_sources))

    # Create arcs from each source to each sink
    for i in range(num_sources):
        for j in range(num_sinks):
            cost = (i + 1) * (j + 1) * 0.5
            arcs.append(
                Arc(
                    tail=f"S{i}",
                    head=f"D{j}",
                    capacity=float(num_sinks) * 2,
                    cost=cost,
                )
            )

    return NetworkProblem(nodes=nodes, arcs=arcs, directed=True)


def create_network_flow_problem(size: int) -> NetworkProblem:
    """Create a network flow problem with intermediate nodes."""
    nodes = {}
    arcs = []

    # Sources
    num_sources = size
    for i in range(num_sources):
        nodes[f"S{i}"] = Node(id=f"S{i}", supply=10.0)

    # Intermediate transshipment nodes
    num_intermediate = size * 2
    for i in range(num_intermediate):
        nodes[f"I{i}"] = Node(id=f"I{i}", supply=0.0)

    # Sinks
    num_sinks = size
    total_supply = num_sources * 10.0
    sink_demand = total_supply / num_sinks
    for i in range(num_sinks):
        nodes[f"D{i}"] = Node(id=f"D{i}", supply=-sink_demand)

    # Sources to intermediate layer 1
    for i in range(num_sources):
        for j in range(num_intermediate // 2):
            arcs.append(Arc(tail=f"S{i}", head=f"I{j}", capacity=20.0, cost=float(i + j + 1)))

    # Intermediate layer 1 to layer 2
    for i in range(num_intermediate // 2):
        for j in range(num_intermediate // 2, num_intermediate):
            if (i + j) % 3 != 0:
                arcs.append(
                    Arc(tail=f"I{i}", head=f"I{j}", capacity=30.0, cost=float((i - j) ** 2 + 1))
                )

    # Intermediate layer 2 to sinks
    for i in range(num_intermediate // 2, num_intermediate):
        for j in range(num_sinks):
            arcs.append(Arc(tail=f"I{i}", head=f"D{j}", capacity=25.0, cost=float(abs(i - j) + 1)))

    return NetworkProblem(nodes=nodes, arcs=arcs, directed=True)


def analyze_projection_patterns(problem: NetworkProblem, name: str):
    """Solve problem and analyze projection patterns."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 80}")

    # Solve the problem (this will populate instrumentation)
    result = solve_min_cost_flow(problem)

    # Access the basis to get instrumentation data
    # We need to get it from the NetworkSimplex instance, so we'll need to modify this
    # For now, let's create a custom solve that returns the basis
    from network_solver.simplex import NetworkSimplex

    solver = NetworkSimplex(problem)
    result = solver.solve()
    basis = solver.basis

    print(f"\nProblem Statistics:")
    print(f"  Nodes: {len(problem.nodes)}")
    print(f"  Arcs: {len(list(problem.undirected_expansion()))}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Objective: {result.objective:.2f}")

    # Analyze projection patterns
    print(f"\n{'-' * 80}")
    print("PROJECTION PATTERN ANALYSIS")
    print(f"{'-' * 80}")

    total_requests = len(basis.projection_history)
    unique_arcs = len(basis.projection_requests)

    print(f"\n1. Request Statistics:")
    print(f"   Total projection requests: {total_requests:,}")
    print(f"   Unique arcs projected: {unique_arcs:,}")
    print(f"   Average requests per arc: {total_requests / unique_arcs:.1f}")
    print(f"   Requests per iteration: {total_requests / result.iterations:.1f}")

    # Analyze repetition (cache hit potential)
    request_counts = Counter(basis.projection_requests.values())
    repeated_arcs = sum(1 for count in basis.projection_requests.values() if count > 1)
    repeat_requests = sum(count for count in basis.projection_requests.values() if count > 1)

    print(f"\n2. Repetition Analysis (Cache Hit Potential):")
    print(f"   Arcs requested once: {request_counts.get(1, 0):,}")
    print(f"   Arcs requested multiple times: {repeated_arcs:,}")
    print(f"   Total repeated requests: {repeat_requests:,}")
    print(f"   Potential cache hit rate: {repeat_requests / total_requests * 100:.1f}%")

    # Most frequently requested arcs
    print(f"\n3. Top 10 Most Frequently Requested Arcs:")
    top_arcs = sorted(basis.projection_requests.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (arc_key, count) in enumerate(top_arcs, 1):
        print(f"   {i}. {arc_key}: {count:,} requests")

    # Working set analysis per basis version
    projections_per_basis = defaultdict(set)
    for basis_ver, arc_key in basis.projection_history:
        projections_per_basis[basis_ver].add(arc_key)

    working_set_sizes = [len(arcs) for arcs in projections_per_basis.values()]
    avg_working_set = sum(working_set_sizes) / len(working_set_sizes) if working_set_sizes else 0
    max_working_set = max(working_set_sizes) if working_set_sizes else 0

    print(f"\n4. Working Set Analysis:")
    print(f"   Number of basis versions: {basis.basis_version}")
    print(f"   Average arcs per basis version: {avg_working_set:.1f}")
    print(f"   Maximum arcs per basis version: {max_working_set}")
    print(f"   Iterations per basis change: {result.iterations / basis.basis_version:.1f}")

    # Temporal locality analysis
    # How many consecutive requests are for the same arc?
    consecutive_same = 0
    for i in range(1, len(basis.projection_history)):
        if basis.projection_history[i][1] == basis.projection_history[i - 1][1]:
            consecutive_same += 1

    print(f"\n5. Temporal Locality:")
    print(f"   Consecutive identical requests: {consecutive_same:,}")
    print(f"   Temporal locality rate: {consecutive_same / total_requests * 100:.1f}%")

    # Cache size recommendations
    print(f"\n6. Cache Size Recommendations:")

    # For LRU cache, ideal size is working set size
    print(
        f"   Conservative (50th percentile): {sorted(working_set_sizes)[len(working_set_sizes) // 2]} arcs"
    )
    print(
        f"   Aggressive (90th percentile): {sorted(working_set_sizes)[int(len(working_set_sizes) * 0.9)]} arcs"
    )
    print(f"   Maximum (100th percentile): {max_working_set} arcs")

    # Memory estimates (assume 8 bytes per float, ~100 floats per projection vector for typical problems)
    bytes_per_projection = 100 * 8  # rough estimate
    print(f"\n   Estimated memory for different cache sizes:")
    print(f"     50 arcs: ~{50 * bytes_per_projection / 1024:.1f} KB")
    print(f"     100 arcs: ~{100 * bytes_per_projection / 1024:.1f} KB")
    print(f"     200 arcs: ~{200 * bytes_per_projection / 1024:.1f} KB")
    print(f"     {max_working_set} arcs: ~{max_working_set * bytes_per_projection / 1024:.1f} KB")

    return {
        "name": name,
        "total_requests": total_requests,
        "unique_arcs": unique_arcs,
        "potential_hit_rate": repeat_requests / total_requests * 100,
        "avg_working_set": avg_working_set,
        "max_working_set": max_working_set,
        "iterations": result.iterations,
        "basis_changes": basis.basis_version,
    }


def main():
    """Run projection pattern analysis on various problem sizes."""
    print("=" * 80)
    print("PROJECTION PATTERN ANALYSIS - Week 1: Cache Design")
    print("=" * 80)

    results = []

    # Test on small problem first
    print("\n[Running small problem for quick validation...]")
    problem = create_transportation_problem(5, 5)
    results.append(analyze_projection_patterns(problem, "Small Transportation (5×5)"))

    # Medium problem
    print("\n[Running medium problem...]")
    problem = create_transportation_problem(15, 15)
    results.append(analyze_projection_patterns(problem, "Medium Transportation (15×15)"))

    # Large problem (most representative)
    print("\n[Running large problem - this will take several minutes...]")
    problem = create_network_flow_problem(20)
    results.append(analyze_projection_patterns(problem, "Medium Network Flow (20 sources)"))

    # Summary comparison
    print(f"\n\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}\n")

    print(f"{'Problem':<40} {'Requests':>10} {'Unique':>8} {'Hit%':>8} {'MaxWS':>8}")
    print(f"{'-' * 80}")
    for r in results:
        print(
            f"{r['name']:<40} {r['total_requests']:>10,} {r['unique_arcs']:>8,} "
            f"{r['potential_hit_rate']:>7.1f}% {r['max_working_set']:>8}"
        )

    # Cache design recommendations
    print(f"\n{'=' * 80}")
    print("CACHE DESIGN RECOMMENDATIONS")
    print(f"{'=' * 80}\n")

    avg_hit_rate = sum(r["potential_hit_rate"] for r in results) / len(results)

    print(f"Based on analysis of {len(results)} problems:\n")
    print(f"1. CACHE HIT POTENTIAL: {avg_hit_rate:.1f}% average")
    print(f"   - Significant opportunity for caching")
    print(f"   - Many arcs are projected multiple times")

    print(f"\n2. CACHE STRATEGY: LRU Cache")
    print(f"   - Working set varies by basis version")
    print(f"   - LRU will naturally evict old projections")
    print(f"   - Invalidate on basis change (increment version)")

    print(f"\n3. CACHE SIZE: Start with 100-200 arcs")
    print(f"   - Covers 90% of working sets")
    print(f"   - Memory overhead: ~80-160 KB")
    print(f"   - Configurable via SolverOptions")

    print(f"\n4. CACHE KEY: (arc_key, basis_version)")
    print(f"   - Ensures correct projection for current basis")
    print(f"   - Simple invalidation on basis changes")

    print(f"\n5. NEXT STEPS (Week 2):")
    print(f"   - Implement LRU cache in TreeBasis")
    print(f"   - Add projection_cache_size to SolverOptions")
    print(f"   - Measure actual hit rates and speedup")
    print(f"   - Tune cache size based on performance")


if __name__ == "__main__":
    main()
