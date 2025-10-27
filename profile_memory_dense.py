#!/usr/bin/env python3
"""Memory profiling script that forces DENSE LU (no sparse) for baseline comparison.

This temporarily disables scipy imports to force dense-only LU factorization,
allowing us to measure the baseline memory usage without sparse optimization.

Usage:
    python profile_memory_dense.py <problem_file>
"""

# DISABLE scipy by removing it from sys.modules before import
# This forces basis_lu.py to use dense LU only
import os
import sys
import threading
import time
from pathlib import Path

import psutil

if "scipy" in sys.modules:
    del sys.modules["scipy"]
if "scipy.sparse" in sys.modules:
    del sys.modules["scipy.sparse"]

# Monkey-patch to prevent scipy import
_original_import = __builtins__.__import__


def _no_scipy_import(name, *args, **kwargs):
    if name.startswith("scipy"):
        raise ModuleNotFoundError("scipy disabled for baseline test")
    return _original_import(name, *args, **kwargs)


__builtins__.__import__ = _no_scipy_import

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))  # noqa: E402

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
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


class MemoryMonitor:
    """Background memory monitor that tracks peak usage."""

    def __init__(self, interval=0.01):
        """Initialize monitor.

        Args:
            interval: Polling interval in seconds (default 10ms)
        """
        self.interval = interval
        self.peak_memory = 0
        self.current_memory = 0
        self.running = False
        self.thread = None
        self.process = psutil.Process(os.getpid())
        self.samples = []

    def _monitor(self):
        """Background monitoring loop."""
        while self.running:
            mem = self.process.memory_info().rss
            self.current_memory = mem
            if mem > self.peak_memory:
                self.peak_memory = mem
            self.samples.append(mem)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.peak_memory = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return peak memory."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_memory

    def get_stats(self):
        """Get memory statistics."""
        if not self.samples:
            return None

        import statistics

        return {
            "min": min(self.samples),
            "max": max(self.samples),
            "mean": statistics.mean(self.samples),
            "median": statistics.median(self.samples),
            "samples": len(self.samples),
        }


def profile_dense_baseline(problem_file: str):
    """Profile memory usage with DENSE LU only (baseline)."""
    print(f"Memory profiling BASELINE (dense LU only): {problem_file}")
    print("=" * 70)
    print("\n⚠️  scipy disabled - using DENSE LU factorization only")
    print("   This is the BASELINE for comparison\n")

    # Parse problem
    print("1. Parsing problem...")
    problem = parse_dimacs_file(problem_file)
    print(f"   Problem: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")

    # Get baseline memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    print(f"\n2. Memory before solve: {format_bytes(mem_before)}")

    # Start memory monitoring
    print("\n3. Starting memory monitor (polling every 10ms)...")
    monitor = MemoryMonitor(interval=0.01)
    monitor.start()

    # Solve
    print("\n4. Solving problem with DENSE LU...")
    start_time = time.time()
    result = solve_min_cost_flow(problem)
    elapsed = time.time() - start_time

    # Stop monitoring
    peak_memory = monitor.stop()
    mem_after = process.memory_info().rss
    stats = monitor.get_stats()

    print("\n5. Results:")
    print(f"   Status: {result.status}")
    print(f"   Objective: {result.objective}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Time: {elapsed:.2f}s")

    print("\n6. Memory Usage (Process RSS) - DENSE BASELINE:")
    print(f"   Memory before solve: {format_bytes(mem_before)}")
    print(f"   Memory after solve:  {format_bytes(mem_after)}")
    print(f"   Peak memory:         {format_bytes(peak_memory)}")
    print(f"   Memory increase:     {format_bytes(mem_after - mem_before)}")

    if stats:
        print("\n7. Memory Statistics:")
        print(f"   Min memory:    {format_bytes(stats['min'])}")
        print(f"   Mean memory:   {format_bytes(stats['mean'])}")
        print(f"   Median memory: {format_bytes(stats['median'])}")
        print(f"   Max memory:    {format_bytes(stats['max'])}")
        print(f"   Samples taken: {stats['samples']}")

    print("\n" + "=" * 70)
    print("BASELINE profiling complete!")
    print("\nThis is the memory usage WITHOUT sparse LU optimization.")
    print("Compare this to profile_memory_peak.py to see the improvement.")

    return {
        "mem_before": mem_before,
        "mem_after": mem_after,
        "mem_peak": peak_memory,
        "mem_increase": mem_after - mem_before,
        "stats": stats,
        "result": result,
        "elapsed": elapsed,
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python profile_memory_dense.py <problem_file>")
        print("\nExample:")
        print(
            "  python profile_memory_dense.py benchmarks/problems/lemon/gridgen/gridgen_8_12a.min"
        )
        sys.exit(1)

    problem_file = sys.argv[1]

    if not Path(problem_file).exists():
        print(f"Error: Problem file not found: {problem_file}")
        sys.exit(1)

    profile_dense_baseline(problem_file)


if __name__ == "__main__":
    main()
