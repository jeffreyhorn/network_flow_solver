"""Profile the network simplex solver to identify performance bottlenecks.

This script creates representative problems and profiles them to find hot paths.
"""

import cProfile
import pstats
from io import StringIO

from network_solver import build_problem, solve_min_cost_flow


def create_small_transportation_problem():
    """Create a small transportation problem (5x5)."""
    num_sources = 5
    num_sinks = 5
    supply_per_source = 20.0

    nodes = []
    for i in range(num_sources):
        nodes.append({"id": f"s{i}", "supply": supply_per_source})
    for j in range(num_sinks):
        nodes.append({"id": f"t{j}", "supply": -supply_per_source})

    arcs = []
    for i in range(num_sources):
        for j in range(num_sinks):
            cost = abs(i - j) + 1.0
            arcs.append(
                {
                    "tail": f"s{i}",
                    "head": f"t{j}",
                    "capacity": 50.0,
                    "cost": cost,
                }
            )

    return build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def create_medium_transportation_problem():
    """Create a medium transportation problem (20x20)."""
    num_sources = 20
    num_sinks = 20
    supply_per_source = 50.0

    nodes = []
    for i in range(num_sources):
        nodes.append({"id": f"s{i}", "supply": supply_per_source})
    for j in range(num_sinks):
        nodes.append({"id": f"t{j}", "supply": -supply_per_source})

    arcs = []
    for i in range(num_sources):
        for j in range(num_sinks):
            cost = (i + j) * 0.5 + 1.0
            arcs.append(
                {
                    "tail": f"s{i}",
                    "head": f"t{j}",
                    "capacity": 100.0,
                    "cost": cost,
                }
            )

    return build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def create_large_transportation_problem():
    """Create a large transportation problem (50x50)."""
    num_sources = 50
    num_sinks = 50
    supply_per_source = 100.0

    nodes = []
    for i in range(num_sources):
        nodes.append({"id": f"s{i}", "supply": supply_per_source})
    for j in range(num_sinks):
        nodes.append({"id": f"t{j}", "supply": -supply_per_source})

    arcs = []
    for i in range(num_sources):
        for j in range(num_sinks):
            cost = (i * j) % 100 + 1.0
            arcs.append(
                {
                    "tail": f"s{i}",
                    "head": f"t{j}",
                    "capacity": 200.0,
                    "cost": cost,
                }
            )

    return build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def create_network_flow_problem():
    """Create a general network flow problem with intermediate nodes."""
    # Create a network with sources, intermediates, and sinks
    num_sources = 10
    num_intermediates = 30
    num_sinks = 10

    nodes = []
    supply_per_source = 100.0

    for i in range(num_sources):
        nodes.append({"id": f"src{i}", "supply": supply_per_source})

    for i in range(num_intermediates):
        nodes.append({"id": f"mid{i}", "supply": 0.0})

    for i in range(num_sinks):
        nodes.append({"id": f"sink{i}", "supply": -supply_per_source})

    arcs = []

    # Connect sources to intermediates
    for i in range(num_sources):
        for j in range(num_intermediates):
            if (i * 3) <= j < (i * 3 + 5):  # Each source connects to ~5 intermediates
                arcs.append(
                    {
                        "tail": f"src{i}",
                        "head": f"mid{j}",
                        "capacity": 150.0,
                        "cost": float(i + j + 1),
                    }
                )

    # Connect intermediates to intermediates (create network structure)
    for i in range(num_intermediates - 1):
        for j in range(i + 1, min(i + 4, num_intermediates)):
            arcs.append(
                {
                    "tail": f"mid{i}",
                    "head": f"mid{j}",
                    "capacity": 100.0,
                    "cost": float(abs(i - j)),
                }
            )

    # Connect intermediates to sinks
    for i in range(num_intermediates):
        for j in range(num_sinks):
            if (i % num_sinks) == j or ((i + 1) % num_sinks) == j:
                arcs.append(
                    {
                        "tail": f"mid{i}",
                        "head": f"sink{j}",
                        "capacity": 150.0,
                        "cost": float(i + j + 1),
                    }
                )

    return build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)


def profile_problem(problem, description):
    """Profile solving a specific problem."""
    print(f"\n{'=' * 80}")
    print(f"Profiling: {description}")
    print(f"{'=' * 80}")

    # Create profiler
    profiler = cProfile.Profile()

    # Profile the solve
    profiler.enable()
    result = solve_min_cost_flow(problem)
    profiler.disable()

    # Print results
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")
    print(f"Objective: {result.objective:.2f}")

    # Create stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    # Sort by cumulative time and print top functions
    print("\nTop 30 functions by cumulative time:")
    print("-" * 80)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())

    # Sort by total time
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    print("\nTop 30 functions by total time:")
    print("-" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(30)
    print(stream.getvalue())

    return stats


def main():
    """Run profiling on various problem sizes."""
    print("Network Simplex Solver - Performance Profiling")
    print("=" * 80)

    # Profile small problem
    small_problem = create_small_transportation_problem()
    profile_problem(small_problem, "Small Transportation Problem (5x5, 25 arcs)")

    # Profile medium problem
    medium_problem = create_medium_transportation_problem()
    profile_problem(medium_problem, "Medium Transportation Problem (20x20, 400 arcs)")

    # Profile large problem
    large_problem = create_large_transportation_problem()
    profile_problem(large_problem, "Large Transportation Problem (50x50, 2500 arcs)")

    # Profile network flow problem
    network_problem = create_network_flow_problem()
    profile_problem(
        network_problem, "Network Flow Problem (10 sources, 30 intermediates, 10 sinks)"
    )

    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
