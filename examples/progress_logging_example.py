"""Example demonstrating progress logging for long-running solves."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import ProgressInfo, build_problem, solve_min_cost_flow  # noqa: E402


def main() -> None:
    """Demonstrate progress logging for visibility into solver execution."""

    print("=" * 70)
    print("PROGRESS LOGGING DEMONSTRATION")
    print("=" * 70)

    # Create a larger problem to demonstrate progress updates
    print("\nBuilding transportation problem...")
    print("  10 suppliers, 10 customers, 200 arcs")

    nodes = []
    # 10 suppliers with 100 units each
    for i in range(10):
        nodes.append({"id": f"supplier_{i}", "supply": 100.0})

    # 10 customers with 100 units demand each
    for i in range(10):
        nodes.append({"id": f"customer_{i}", "supply": -100.0})

    arcs = []
    # Each supplier can ship to each customer (100 arcs)
    for i in range(10):
        for j in range(10):
            cost = abs(i - j) + 1.0  # Cost increases with distance
            arcs.append(
                {
                    "tail": f"supplier_{i}",
                    "head": f"customer_{j}",
                    "capacity": 50.0,
                    "cost": cost,
                }
            )

    # Add some intermediate warehouse nodes to make it more interesting
    for i in range(10):
        nodes.append({"id": f"warehouse_{i}", "supply": 0.0})

    # Connect suppliers to warehouses (100 more arcs)
    for i in range(10):
        for j in range(10):
            arcs.append(
                {
                    "tail": f"supplier_{i}",
                    "head": f"warehouse_{j}",
                    "capacity": 30.0,
                    "cost": 1.0,
                }
            )

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    print(f"  Nodes: {len(nodes)}")
    print(f"  Arcs: {len(arcs)}")

    # Progress callback with formatted output
    last_percent = -1

    def progress_callback(info: ProgressInfo) -> None:
        nonlocal last_percent
        percent = int(100 * info.iteration / info.max_iterations)

        # Only print when percentage changes to avoid too much output
        if percent != last_percent:
            last_percent = percent
            phase_name = "Phase 1 (Feasibility)" if info.phase == 1 else "Phase 2 (Optimality)"
            print(
                f"\r{phase_name}: {percent:3d}% | "
                f"Iter: {info.iteration:5d}/{info.max_iterations} | "
                f"Objective: ${info.objective_estimate:12,.2f} | "
                f"Time: {info.elapsed_time:6.2f}s",
                end="",
                flush=True,
            )

    print("\nSolving with progress logging...")
    print("-" * 70)

    result = solve_min_cost_flow(
        problem,
        progress_callback=progress_callback,
        progress_interval=10,  # Update every 10 iterations
    )

    print()  # New line after progress bar
    print("-" * 70)

    print("\nSolution found:")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:,.2f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Non-zero flows: {len(result.flows)}")

    print("\n" + "=" * 70)
    print("PROGRESS LOGGING FEATURES")
    print("=" * 70)
    print("\nThe progress callback provides:")
    print("  • Real-time iteration count and progress percentage")
    print("  • Current phase (Phase 1: feasibility, Phase 2: optimality)")
    print("  • Objective value estimate during solve")
    print("  • Elapsed time tracking")
    print("  • Customizable update frequency via progress_interval")

    print("\nUse cases:")
    print("  • Monitor long-running optimizations")
    print("  • Implement custom progress bars or GUIs")
    print("  • Log progress to files or monitoring systems")
    print("  • Detect slow convergence issues")
    print("  • Cancel solver if taking too long (via exception in callback)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
