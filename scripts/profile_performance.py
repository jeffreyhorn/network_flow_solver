#!/usr/bin/env python3
"""Comprehensive performance profiling for the network simplex solver.

This script profiles the solver on various problem sizes and generates detailed
performance reports including:
- Function-level timing breakdown
- Call counts and cumulative time
- Hot path identification
- Line-by-line profiling for critical functions
"""

import cProfile
import pstats
import io
import time
from pathlib import Path

from network_solver import solve_min_cost_flow
from network_solver.data import Arc, NetworkProblem, Node, SolverOptions


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
            cost = (i + 1) * (j + 1) * 0.5  # Varied costs
            arcs.append(
                Arc(
                    tail=f"S{i}",
                    head=f"D{j}",
                    capacity=float(num_sinks) * 2,  # Plenty of capacity
                    cost=cost,
                )
            )

    return NetworkProblem(nodes=nodes, arcs=arcs, directed=True)


def create_network_flow_problem(size: int) -> NetworkProblem:
    """Create a more complex network flow problem with intermediate nodes."""
    nodes = {}
    arcs = []

    # Sources
    num_sources = size
    for i in range(num_sources):
        nodes[f"S{i}"] = Node(id=f"S{i}", supply=10.0)

    # Intermediate transshipment nodes (2 layers)
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
            arcs.append(
                Arc(
                    tail=f"S{i}",
                    head=f"I{j}",
                    capacity=20.0,
                    cost=float(i + j + 1),
                )
            )

    # Intermediate layer 1 to layer 2
    for i in range(num_intermediate // 2):
        for j in range(num_intermediate // 2, num_intermediate):
            if (i + j) % 3 != 0:  # Create some sparsity
                arcs.append(
                    Arc(
                        tail=f"I{i}",
                        head=f"I{j}",
                        capacity=30.0,
                        cost=float((i - j) ** 2 + 1),
                    )
                )

    # Intermediate layer 2 to sinks
    for i in range(num_intermediate // 2, num_intermediate):
        for j in range(num_sinks):
            arcs.append(
                Arc(
                    tail=f"I{i}",
                    head=f"D{j}",
                    capacity=25.0,
                    cost=float(abs(i - j) + 1),
                )
            )

    return NetworkProblem(nodes=nodes, arcs=arcs, directed=True)


def profile_problem(problem: NetworkProblem, name: str) -> dict:
    """Profile a problem and return statistics."""
    print(f"\n{'=' * 80}")
    print(f"Profiling: {name}")
    print(f"{'=' * 80}")
    print(f"Nodes: {len(problem.nodes)}, Arcs: {len(list(problem.undirected_expansion()))}")

    # Warm-up run
    _ = solve_min_cost_flow(problem)

    # Profiled run
    profiler = cProfile.Profile()
    start_time = time.time()
    profiler.enable()

    result = solve_min_cost_flow(problem)

    profiler.disable()
    elapsed = time.time() - start_time

    # Collect statistics - output directly to stdout
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print(f"\nSolved in {elapsed:.3f}s, {result.iterations} iterations")
    print(f"Objective: {result.objective:.2f}")

    # Print top functions by cumulative time
    print(f"\n{'-' * 80}")
    print("Top 20 functions by cumulative time:")
    print(f"{'-' * 80}")
    stats.print_stats(20)

    # Print top functions by total time
    stats.sort_stats("tottime")
    print(f"\n{'-' * 80}")
    print("Top 20 functions by total time:")
    print(f"{'-' * 80}")
    stats.print_stats(20)

    return {
        "name": name,
        "nodes": len(problem.nodes),
        "arcs": len(list(problem.undirected_expansion())),
        "time": elapsed,
        "iterations": result.iterations,
        "objective": result.objective,
        "stats": stats,
    }


def main():
    """Run comprehensive profiling suite."""
    print("=" * 80)
    print("NETWORK SIMPLEX SOLVER - PERFORMANCE PROFILING")
    print("=" * 80)

    results = []

    # Small transportation problem
    problem = create_transportation_problem(5, 5)
    results.append(profile_problem(problem, "Small Transportation (5×5)"))

    # Medium transportation problem
    problem = create_transportation_problem(15, 15)
    results.append(profile_problem(problem, "Medium Transportation (15×15)"))

    # Large transportation problem
    problem = create_transportation_problem(30, 30)
    results.append(profile_problem(problem, "Large Transportation (30×30)"))

    # Small network flow
    problem = create_network_flow_problem(10)
    results.append(profile_problem(problem, "Small Network Flow (10 sources)"))

    # Medium network flow
    problem = create_network_flow_problem(20)
    results.append(profile_problem(problem, "Medium Network Flow (20 sources)"))

    # Large network flow
    problem = create_network_flow_problem(40)
    results.append(profile_problem(problem, "Large Network Flow (40 sources)"))

    # Summary
    print(f"\n\n{'=' * 80}")
    print("PROFILING SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Problem':<40} {'Nodes':>8} {'Arcs':>8} {'Time (s)':>10} {'Iters':>8}")
    print(f"{'-' * 80}")
    for r in results:
        print(
            f"{r['name']:<40} {r['nodes']:>8} {r['arcs']:>8} {r['time']:>10.3f} {r['iterations']:>8}"
        )

    print(f"\n{'=' * 80}")
    print("Profiling complete! Review the detailed output above.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
