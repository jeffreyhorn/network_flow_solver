#!/usr/bin/env python3
"""Benchmark runner for min-cost flow solver.

This script runs the solver on benchmark instances and collects performance metrics
including solve time, iterations, objective value, and solution status.

Usage:
    python benchmarks/scripts/run_benchmark.py --small           # Run on small instances
    python benchmarks/scripts/run_benchmark.py --medium          # Run on medium instances
    python benchmarks/scripts/run_benchmark.py --file instance.min  # Run on specific file
    python benchmarks/scripts/run_benchmark.py --dir benchmarks/problems/lemon/netgen/  # Run on directory
    python benchmarks/scripts/run_benchmark.py --small --output results.json  # Save results to JSON

Features:
    - Automatic instance discovery from benchmark directories
    - Performance metrics collection (time, iterations, objective)
    - Multiple output formats (JSON, CSV, markdown table)
    - Progress reporting with live updates
    - Error handling and timeout support

Example:
    # Run benchmarks on small instances and save results
    python benchmarks/scripts/run_benchmark.py --small --output results.json

    # Generate markdown report
    python benchmarks/scripts/run_benchmark.py --small --output results.md --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from src.network_solver.solver import solve_min_cost_flow  # noqa: E402


@dataclass
class BenchmarkResult:
    """Result from running solver on a single instance."""

    instance_name: str
    instance_path: str
    nodes: int
    arcs: int
    status: str
    objective: float | None
    iterations: int
    solve_time_ms: float
    parse_time_ms: float
    total_time_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def run_single_benchmark(
    instance_path: Path,
    timeout_seconds: float = 300.0,
    max_iterations: int | None = None,
) -> BenchmarkResult:
    """Run solver on a single benchmark instance with timeout enforcement.

    Args:
        instance_path: Path to DIMACS instance file.
        timeout_seconds: Maximum time to allow for solving (default 300s = 5 min).
        max_iterations: Maximum simplex iterations (None uses solver default: 20 * arcs).

    Returns:
        BenchmarkResult with performance metrics.
    """
    instance_name = instance_path.stem
    error = None
    status = "unknown"
    objective = None
    iterations = 0
    nodes = 0
    arcs = 0
    parse_time_ms = 0.0
    solve_time_ms = 0.0

    try:
        # Parse instance
        parse_start = time.perf_counter()
        problem = parse_dimacs_file(str(instance_path))
        parse_end = time.perf_counter()
        parse_time_ms = (parse_end - parse_start) * 1000

        nodes = len(problem.nodes)
        arcs = len(problem.arcs)

        # Solve instance with timeout enforcement using threading
        result_container = {}
        exception_container = {}

        def solve_with_timeout():
            """Run solver in a separate thread."""
            try:
                solve_result = solve_min_cost_flow(problem, max_iterations=max_iterations)
                result_container["result"] = solve_result
            except Exception as e:
                exception_container["exception"] = e

        solve_start = time.perf_counter()
        solver_thread = threading.Thread(target=solve_with_timeout, daemon=True)
        solver_thread.start()
        solver_thread.join(timeout=timeout_seconds)
        solve_end = time.perf_counter()
        solve_time_ms = (solve_end - solve_start) * 1000

        # Check if thread completed or timed out
        if solver_thread.is_alive():
            # Timeout occurred - thread is still running
            error = f"Timeout after {timeout_seconds:.1f}s"
            status = "timeout"
            # Note: We cannot force-kill the thread, but we stop waiting for it
        elif "exception" in exception_container:
            # Solver raised an exception
            error = str(exception_container["exception"])
            status = "error"
        elif "result" in result_container:
            # Solver completed successfully
            result = result_container["result"]
            status = result.status
            objective = result.objective if result.status == "optimal" else None
            iterations = result.iterations
        else:
            # Unexpected state
            error = "Solver completed without result or exception"
            status = "error"

    except Exception as e:
        error = str(e)
        status = "error"

    total_time_ms = parse_time_ms + solve_time_ms

    return BenchmarkResult(
        instance_name=instance_name,
        instance_path=str(instance_path),
        nodes=nodes,
        arcs=arcs,
        status=status,
        objective=objective,
        iterations=iterations,
        solve_time_ms=solve_time_ms,
        parse_time_ms=parse_time_ms,
        total_time_ms=total_time_ms,
        error=error,
    )


def discover_instances(
    size_category: str | None = None,
    directory: Path | None = None,
) -> list[Path]:
    """Discover benchmark instances to run.

    Args:
        size_category: Size category (small, medium, large) or None for all.
        directory: Specific directory to search, or None for default locations.

    Returns:
        List of Path objects for discovered instances.
    """
    instances: list[Path] = []

    if directory:
        # Search specific directory
        instances.extend(directory.glob("*.min"))
    else:
        # Search by size category
        base_dir = Path("benchmarks/problems/lemon")
        if not base_dir.exists():
            return instances

        for family_dir in base_dir.iterdir():
            if family_dir.is_dir():
                instances.extend(family_dir.glob("*.min"))

    # Filter by size if requested (based on file naming convention)
    if size_category:
        filtered = []
        for inst in instances:
            # Small: *_08*.min, *_09*.min, *_10*.min, *_11*.min
            # Medium: *_12*.min, *_13*.min, *_14*.min
            # Large: *_15*.min and above
            if (
                size_category == "small"
                and any(f"_{i:02d}" in inst.stem for i in range(8, 12))
                or size_category == "medium"
                and any(f"_{i:02d}" in inst.stem for i in range(12, 15))
                or size_category == "large"
                and any(f"_{i:02d}" in inst.stem for i in range(15, 23))
            ):
                filtered.append(inst)
        instances = filtered

    return sorted(instances)


def save_results_json(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: List of benchmark results.
        output_path: Path to output JSON file.
    """
    data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_instances": len(results),
            "successful": sum(1 for r in results if r.status == "optimal"),
            "failed": sum(1 for r in results if r.status == "error"),
            "timeout": sum(1 for r in results if r.status == "timeout"),
        },
        "results": [r.to_dict() for r in results],
    }

    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to CSV file.

    Args:
        results: List of benchmark results.
        output_path: Path to output CSV file.
    """
    import csv

    with output_path.open("w", newline="") as f:
        if not results:
            return

        fieldnames = list(results[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"Results saved to {output_path}")


def save_results_markdown(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to markdown table.

    Args:
        results: List of benchmark results.
        output_path: Path to output markdown file.
    """
    lines = ["# Benchmark Results\n"]
    lines.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Total Instances**: {len(results)}\n")
    lines.append(f"**Successful**: {sum(1 for r in results if r.status == 'optimal')}\n")
    lines.append(f"**Failed**: {sum(1 for r in results if r.status == 'error')}\n")
    lines.append(f"**Timeout**: {sum(1 for r in results if r.status == 'timeout')}\n")
    lines.append("\n## Results Table\n")
    lines.append(
        "| Instance | Nodes | Arcs | Status | Objective | Iterations | Solve Time (ms) |\n"
    )
    lines.append("|----------|-------|------|--------|-----------|------------|----------------|\n")

    for r in results:
        obj_str = f"{r.objective:.2f}" if r.objective is not None else "N/A"
        lines.append(
            f"| {r.instance_name} | {r.nodes} | {r.arcs} | {r.status} | {obj_str} | {r.iterations} | {r.solve_time_ms:.2f} |\n"
        )

    output_path.write_text("".join(lines))
    print(f"Results saved to {output_path}")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print summary statistics.

    Args:
        results: List of benchmark results.
    """
    if not results:
        print("No results to summarize")
        return

    successful = [r for r in results if r.status == "optimal"]
    failed = [r for r in results if r.status == "error"]
    timeout = [r for r in results if r.status == "timeout"]

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total instances: {len(results)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"  ⏱ Timeout: {len(timeout)}")

    if successful:
        avg_solve_time = sum(r.solve_time_ms for r in successful) / len(successful)
        avg_iterations = sum(r.iterations for r in successful) / len(successful)
        print(f"\nAverage solve time: {avg_solve_time:.2f} ms")
        print(f"Average iterations: {avg_iterations:.0f}")

    print("=" * 70)


def main() -> int:
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run min-cost flow solver on benchmark instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Instance selection
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run on small instances (256-512 nodes)",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run on medium instances (4K-16K nodes)",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Run on large instances (>16K nodes)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Run on specific instance file",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Run on all instances in directory",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (extension determines format: .json, .csv, .md)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        help="Output format (overrides file extension)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds per instance (default: 300)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        metavar="N",
        help="Maximum simplex iterations (default: solver uses 20 * arcs)",
    )

    args = parser.parse_args()

    # Determine instances to run
    instances: list[Path] = []

    if args.file:
        instances = [args.file]
    elif args.dir:
        instances = discover_instances(directory=args.dir)
    else:
        size_category = None
        if args.small:
            size_category = "small"
        elif args.medium:
            size_category = "medium"
        elif args.large:
            size_category = "large"

        if size_category:
            instances = discover_instances(size_category=size_category)
        else:
            parser.print_help()
            return 1

    if not instances:
        print("No instances found to run")
        return 1

    print(f"Found {len(instances)} instances to run")
    print("=" * 70)

    # Run benchmarks
    results: list[BenchmarkResult] = []

    for i, instance_path in enumerate(instances, 1):
        print(f"\n[{i}/{len(instances)}] Running {instance_path.name}...", end=" ")
        sys.stdout.flush()

        result = run_single_benchmark(
            instance_path,
            timeout_seconds=args.timeout,
            max_iterations=args.max_iterations,
        )
        results.append(result)

        if result.status == "optimal":
            print(f"✓ {result.solve_time_ms:.1f}ms, {result.iterations} iterations")
        elif result.status == "error":
            print(f"✗ Error: {result.error}")
        elif result.status == "timeout":
            print("⏱ Timeout")
        else:
            print(f"? Status: {result.status}")

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        # Determine format
        if args.format:
            output_format = args.format
        else:
            # Infer from extension
            ext = args.output.suffix.lower()
            if ext == ".json":
                output_format = "json"
            elif ext == ".csv":
                output_format = "csv"
            elif ext == ".md":
                output_format = "markdown"
            else:
                print(f"Unknown output format for {ext}, defaulting to JSON")
                output_format = "json"

        # Save
        if output_format == "json":
            save_results_json(results, args.output)
        elif output_format == "csv":
            save_results_csv(results, args.output)
        elif output_format == "markdown":
            save_results_markdown(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
