#!/usr/bin/env python3
"""Memory profiling script using psutil for system-level memory tracking.

This script uses psutil to track actual process memory (RSS), which includes
all memory allocated by the process including C extensions like NumPy and scipy.

Usage:
    python profile_memory_psutil.py <problem_file>
"""

import os
import sys
import time
from pathlib import Path

import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))  # noqa: E402

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from src.network_solver.solver import solve_min_cost_flow  # noqa: E402


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


def get_memory_usage():
    """Get current process memory usage in bytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size (actual physical memory)


def profile_with_psutil(problem_file: str):
    """Profile memory usage using psutil."""
    print(f"Memory profiling (psutil/RSS): {problem_file}")
    print("=" * 70)

    # Parse problem
    print("\n1. Parsing problem...")
    problem = parse_dimacs_file(problem_file)
    print(f"   Problem: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    # Get baseline memory
    print("\n2. Baseline memory...")
    mem_before_parse = get_memory_usage()
    print(f"   Memory after parsing: {format_bytes(mem_before_parse)}")

    # Solve and track peak memory
    print("\n3. Solving problem...")
    mem_before_solve = get_memory_usage()

    # Start solve
    start_time = time.time()
    result = solve_min_cost_flow(problem)
    elapsed = time.time() - start_time

    # Get final memory
    mem_after_solve = get_memory_usage()

    # Note: We can't perfectly track peak during solve without threading,
    # but we can get before/after and a rough estimate
    print("\n4. Results:")
    print(f"   Status: {result.status}")
    print(f"   Objective: {result.objective}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Time: {elapsed:.2f}s")

    print("\n5. Memory Usage (Process RSS):")
    print(f"   Memory before solve: {format_bytes(mem_before_solve)}")
    print(f"   Memory after solve: {format_bytes(mem_after_solve)}")
    print(f"   Memory increase: {format_bytes(mem_after_solve - mem_before_solve)}")
    print("   ")
    print("   Note: This shows residual memory. Peak memory during solve")
    print("   may have been higher (garbage collected after solve).")

    print("\n" + "=" * 70)
    print("Memory profiling complete!")
    print("\nRSS = Resident Set Size (actual physical memory used by process)")
    print("This includes all memory: Python objects, NumPy arrays, scipy C code, etc.")

    return {
        "mem_before": mem_before_solve,
        "mem_after": mem_after_solve,
        "mem_increase": mem_after_solve - mem_before_solve,
        "result": result,
        "elapsed": elapsed,
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python profile_memory_psutil.py <problem_file>")
        print("\nExample:")
        print(
            "  python profile_memory_psutil.py benchmarks/problems/lemon/gridgen/gridgen_8_12a.min"
        )
        sys.exit(1)

    problem_file = sys.argv[1]

    if not Path(problem_file).exists():
        print(f"Error: Problem file not found: {problem_file}")
        sys.exit(1)

    profile_with_psutil(problem_file)


if __name__ == "__main__":
    main()
