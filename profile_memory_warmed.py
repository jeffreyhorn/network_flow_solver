#!/usr/bin/env python3
"""Memory profiling script with JIT warm-up.

This script runs a small problem first to trigger Numba JIT compilation,
then profiles the target problem without JIT compilation overhead.

Usage:
    python profile_memory_warmed.py <problem_file>
"""

import sys
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))  # noqa: E402

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from network_solver.data import build_problem  # noqa: E402
from network_solver.solver import solve_min_cost_flow  # noqa: E402


def format_bytes(bytes_val):
    """Format bytes as human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def warmup_jit():
    """Run a tiny problem to trigger Numba JIT compilation."""
    print("Warming up JIT compilation...")
    tiny_problem = build_problem(
        nodes=[
            {"id": "s", "supply": 10.0},
            {"id": "t", "supply": -10.0},
        ],
        arcs=[
            {"tail": "s", "head": "t", "cost": 1.0, "capacity": 20.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solve_min_cost_flow(tiny_problem)
    print("   JIT compilation complete\n")


def profile_memory_usage(problem_file: str):
    """Profile memory usage while solving a problem."""
    print(f"Memory profiling (with JIT warm-up): {problem_file}")
    print("=" * 70)

    # Parse problem
    print("\n1. Parsing problem...")
    problem = parse_dimacs_file(problem_file)
    print(f"   Problem: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    # Warm up JIT compilation first (not tracked)
    print("\n2. JIT warm-up (not tracked)...")
    warmup_jit()

    # Start memory tracking AFTER JIT compilation
    print("3. Starting memory tracking (post-JIT)...")
    tracemalloc.start()

    # Take snapshot before solving
    snapshot_before = tracemalloc.take_snapshot()
    mem_before = tracemalloc.get_traced_memory()[0]
    print(f"   Memory before solve: {format_bytes(mem_before)}")

    # Solve the problem
    print("\n4. Solving problem...")
    result = solve_min_cost_flow(problem)

    # Take snapshot after solving
    snapshot_after = tracemalloc.take_snapshot()
    mem_after, mem_peak = tracemalloc.get_traced_memory()

    print("\n5. Results:")
    print(f"   Status: {result.status}")
    print(f"   Objective: {result.objective}")
    print(f"   Iterations: {result.iterations}")

    print("\n6. Memory Usage (excluding JIT compilation):")
    print(f"   Memory after solve: {format_bytes(mem_after)}")
    print(f"   Peak memory: {format_bytes(mem_peak)}")
    print(f"   Memory increase: {format_bytes(mem_after - mem_before)}")

    # Compare snapshots to see what allocated memory
    print("\n7. Top 20 Memory Allocations:")
    print("   " + "-" * 66)

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

    for i, stat in enumerate(top_stats[:20], 1):
        print(f"   #{i:2d} {stat}")

    # Show statistics by file
    print("\n8. Memory by File:")
    print("   " + "-" * 66)

    file_stats = snapshot_after.compare_to(snapshot_before, "filename")
    for i, stat in enumerate(file_stats[:15], 1):
        print(f"   #{i:2d} {stat}")

    # Show current memory breakdown
    print("\n9. Current Memory Snapshot (Top 20):")
    print("   " + "-" * 66)

    current_stats = snapshot_after.statistics("lineno")
    for i, stat in enumerate(current_stats[:20], 1):
        print(f"   #{i:2d} {stat}")

    tracemalloc.stop()

    print("\n" + "=" * 70)
    print("Memory profiling complete!")
    print("\nNote: Peak memory measured AFTER JIT compilation warm-up")
    print("This excludes Numba's one-time JIT compilation overhead")

    return {
        "mem_before": mem_before,
        "mem_after": mem_after,
        "mem_peak": mem_peak,
        "mem_increase": mem_after - mem_before,
        "result": result,
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python profile_memory_warmed.py <problem_file>")
        print("\nExample:")
        print(
            "  python profile_memory_warmed.py benchmarks/problems/lemon/gridgen/gridgen_8_12a.min"
        )
        sys.exit(1)

    problem_file = sys.argv[1]

    if not Path(problem_file).exists():
        print(f"Error: Problem file not found: {problem_file}")
        sys.exit(1)

    profile_memory_usage(problem_file)


if __name__ == "__main__":
    main()
