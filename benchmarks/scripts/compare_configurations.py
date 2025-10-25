#!/usr/bin/env python3
"""Compare solver performance across different configurations.

This tool runs benchmarks with different solver options and compares results,
useful for evaluating performance improvements and regressions.

Usage:
    # Compare default vs explicit Dantzig on GOTO instances
    python benchmarks/scripts/compare_configurations.py \
        --instances benchmarks/problems/lemon/goto/*.min \
        --configs default dantzig \
        --output comparison_results.json

    # Compare with custom timeout
    python benchmarks/scripts/compare_configurations.py \
        --instances benchmarks/problems/lemon/goto/*.min \
        --configs default dantzig \
        --timeout 60 \
        --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from src.network_solver.data import SolverOptions  # noqa: E402
from src.network_solver.simplex import NetworkSimplex  # noqa: E402


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    instance: str
    config: str
    status: str
    iterations: int
    solve_time: float
    objective: float | None
    timeout: bool


# Predefined configurations
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "default": {
        "description": "Default solver settings (auto-detection enabled)",
        "options": {},
    },
    "dantzig": {
        "description": "Force Dantzig pricing (most-negative reduced cost)",
        "options": {"pricing_strategy": "dantzig"},
    },
    "devex": {
        "description": "Force Devex pricing (normalized pricing)",
        "options": {"pricing_strategy": "devex"},
    },
    "no_scaling": {
        "description": "Disable automatic problem scaling",
        "options": {"enable_scaling": False},
    },
}


def run_benchmark(
    instance_path: Path,
    config_name: str,
    config_options: dict[str, Any],
    timeout: float,
    max_iterations: int,
) -> BenchmarkResult:
    """Run a single benchmark with specified configuration."""
    # Parse instance
    problem = parse_dimacs_file(str(instance_path))

    # Create solver options with iteration-based timeout
    # Estimate: ~100 iterations/second on average for timeout approximation
    estimated_iters_per_sec = 100
    timeout_iterations = min(max_iterations, int(timeout * estimated_iters_per_sec))

    options_dict = {
        "max_iterations": timeout_iterations,
        **config_options,
    }
    # Mark explicit pricing strategy to override auto-detection if pricing_strategy is specified
    if "pricing_strategy" in config_options:
        options_dict["explicit_pricing_strategy"] = True

    options = SolverOptions(**options_dict)

    # Run solver
    solver = NetworkSimplex(problem, options=options)

    start_time = time.perf_counter()
    try:
        result = solver.solve()
        solve_time = time.perf_counter() - start_time

        # Check if we hit iteration limit (likely timeout)
        timed_out = result.status != "optimal" and result.iterations >= timeout_iterations

        return BenchmarkResult(
            instance=instance_path.name,
            config=config_name,
            status=result.status,
            iterations=result.iterations,
            solve_time=solve_time,
            objective=result.objective if result.status == "optimal" else None,
            timeout=timed_out,
        )

    except Exception as e:
        solve_time = time.perf_counter() - start_time
        return BenchmarkResult(
            instance=instance_path.name,
            config=config_name,
            status=f"error: {type(e).__name__}",
            iterations=0,
            solve_time=solve_time,
            objective=None,
            timeout=False,
        )


def compare_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate comparison statistics from results."""
    # Group by instance
    by_instance: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        if result.instance not in by_instance:
            by_instance[result.instance] = []
        by_instance[result.instance].append(result)

    # Calculate statistics
    comparison = {
        "instances": {},
        "summary": {},
    }

    for instance, instance_results in by_instance.items():
        instance_comparison = {}
        for result in instance_results:
            instance_comparison[result.config] = {
                "status": result.status,
                "iterations": result.iterations,
                "time": round(result.solve_time, 3),
                "objective": result.objective,
                "timeout": result.timeout,
            }
        comparison["instances"][instance] = instance_comparison

    # Summary statistics by config
    by_config: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        if result.config not in by_config:
            by_config[result.config] = []
        by_config[result.config].append(result)

    for config, config_results in by_config.items():
        optimal_count = sum(1 for r in config_results if r.status == "optimal")
        timeout_count = sum(1 for r in config_results if r.timeout)
        total_time = sum(r.solve_time for r in config_results)
        avg_time = total_time / len(config_results) if config_results else 0

        comparison["summary"][config] = {
            "total_instances": len(config_results),
            "optimal": optimal_count,
            "timeouts": timeout_count,
            "total_time": round(total_time, 3),
            "avg_time": round(avg_time, 3),
        }

    return comparison


def print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 100)

    # Print instance-by-instance comparison
    print("\nPer-Instance Results:")
    print("-" * 100)
    print(f"{'Instance':<30} {'Config':<15} {'Status':<12} {'Iterations':>10} {'Time (s)':>10}")
    print("-" * 100)

    for instance, configs in sorted(comparison["instances"].items()):
        for i, (config, result) in enumerate(sorted(configs.items())):
            instance_name = instance if i == 0 else ""
            status = result["status"]
            if result["timeout"]:
                status += " (TIMEOUT)"

            print(
                f"{instance_name:<30} {config:<15} {status:<12} "
                f"{result['iterations']:>10} {result['time']:>10.2f}"
            )
        print()

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY BY CONFIGURATION")
    print("=" * 100)
    print(
        f"{'Config':<15} {'Instances':>10} {'Optimal':>10} {'Timeouts':>10} {'Total Time':>12} {'Avg Time':>10}"
    )
    print("-" * 100)

    for config, summary in sorted(comparison["summary"].items()):
        print(
            f"{config:<15} {summary['total_instances']:>10} {summary['optimal']:>10} "
            f"{summary['timeouts']:>10} {summary['total_time']:>12.2f} {summary['avg_time']:>10.2f}"
        )

    print("=" * 100)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare solver performance across configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available configurations:
{chr(10).join(f"  {name}: {cfg['description']}" for name, cfg in CONFIGURATIONS.items())}

Examples:
  # Compare default vs Dantzig on GOTO instances
  python benchmarks/scripts/compare_configurations.py \\
      --instances benchmarks/problems/lemon/goto/*.min \\
      --configs default dantzig

  # Compare all configs on specific instances
  python benchmarks/scripts/compare_configurations.py \\
      --instances goto_8_08a.min goto_8_08b.min \\
      --configs default dantzig devex
        """,
    )

    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="Instance files to benchmark (supports glob patterns)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGURATIONS.keys()),
        required=True,
        help="Configurations to compare",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per instance in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100000,
        help="Maximum iterations per solve (default: 100000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Expand instance paths (handle globs)
    instance_paths: list[Path] = []
    for pattern in args.instances:
        path = Path(pattern)
        if path.is_file():
            instance_paths.append(path)
        else:
            # Try as glob pattern
            matches = list(Path(".").glob(pattern))
            instance_paths.extend(m for m in matches if m.is_file())

    if not instance_paths:
        print(f"Error: No instances found matching patterns: {args.instances}")
        return 1

    print(
        f"\nRunning comparison on {len(instance_paths)} instances with {len(args.configs)} configurations..."
    )
    print(f"Timeout: {args.timeout}s, Max iterations: {args.max_iterations}\n")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    total_runs = len(instance_paths) * len(args.configs)
    current_run = 0

    for instance_path in sorted(instance_paths):
        for config_name in args.configs:
            current_run += 1
            print(
                f"[{current_run}/{total_runs}] Running {instance_path.name} with {config_name}...",
                end=" ",
                flush=True,
            )

            config = CONFIGURATIONS[config_name]
            result = run_benchmark(
                instance_path,
                config_name,
                config["options"],
                args.timeout,
                args.max_iterations,
            )
            results.append(result)

            status_str = "✓" if result.status == "optimal" else "✗"
            print(f"{status_str} {result.status} ({result.solve_time:.2f}s)")

    # Generate comparison
    comparison = compare_results(results)

    # Print results
    print_comparison_table(comparison)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
