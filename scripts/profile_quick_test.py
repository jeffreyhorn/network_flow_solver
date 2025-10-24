#!/usr/bin/env python3
"""Quick test to verify profiling output works."""

import cProfile
import pstats
from network_solver import solve_min_cost_flow
from network_solver.data import Arc, NetworkProblem, Node


def create_small_problem():
    """Create a tiny test problem."""
    nodes = {
        "S1": Node(id="S1", supply=10.0),
        "S2": Node(id="S2", supply=10.0),
        "D1": Node(id="D1", supply=-10.0),
        "D2": Node(id="D2", supply=-10.0),
    }

    arcs = [
        Arc(tail="S1", head="D1", capacity=20.0, cost=1.0),
        Arc(tail="S1", head="D2", capacity=20.0, cost=2.0),
        Arc(tail="S2", head="D1", capacity=20.0, cost=3.0),
        Arc(tail="S2", head="D2", capacity=20.0, cost=1.0),
    ]

    return NetworkProblem(nodes=nodes, arcs=arcs, directed=True)


def main():
    print("Testing profiling output...")
    problem = create_small_problem()

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    result = solve_min_cost_flow(problem)
    profiler.disable()

    # Print stats
    print(f"\nSolved! Objective: {result.objective}, Iterations: {result.iterations}")
    print("\n" + "=" * 80)
    print("Top 30 functions by cumulative time:")
    print("=" * 80)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("\n" + "=" * 80)
    print("Top 30 functions by total time:")
    print("=" * 80)

    stats.sort_stats("tottime")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
