#!/usr/bin/env python3
"""Diagnostic script to investigate GOTO instance convergence issues.

This script runs a GOTO instance with detailed logging and analysis to understand
why these instances don't converge in reasonable time.

Usage:
    python benchmarks/scripts/diagnose_goto.py benchmarks/problems/lemon/goto/goto_8_08a.min
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.parsers.dimacs import parse_dimacs_file  # noqa: E402
from src.network_solver.data import SolverOptions  # noqa: E402
from src.network_solver.simplex import NetworkSimplex  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to show solver internals."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def analyze_problem_structure(problem) -> dict:
    """Analyze structural properties of the problem."""
    # Extract supply from nodes dict (nodes is dict[node_id, Node])
    supplies = [node.supply for node in problem.nodes.values()]

    info = {
        "nodes": len(problem.nodes),
        "arcs": len(problem.arcs),
        "total_supply": sum(abs(s) for s in supplies),
        "supply_nodes": sum(1 for s in supplies if s > 0),
        "demand_nodes": sum(1 for s in supplies if s < 0),
        "transshipment_nodes": sum(1 for s in supplies if s == 0),
    }

    # Analyze arc costs and capacities
    if problem.arcs:
        costs = [arc.cost for arc in problem.arcs]
        capacities = [arc.capacity for arc in problem.arcs]
        info["min_cost"] = min(costs)
        info["max_cost"] = max(costs)
        info["avg_cost"] = sum(costs) / len(costs)
        info["min_capacity"] = min(capacities)
        info["max_capacity"] = max(capacities)
        info["avg_capacity"] = sum(capacities) / len(capacities)

    # Analyze node degrees (using node IDs from dict keys)
    node_ids = list(problem.nodes.keys())
    in_degree = {node_id: 0 for node_id in node_ids}
    out_degree = {node_id: 0 for node_id in node_ids}
    for arc in problem.arcs:
        out_degree[arc.tail] += 1
        in_degree[arc.head] += 1

    degrees = list(in_degree.values()) + list(out_degree.values())
    if degrees:
        info["min_degree"] = min(degrees)
        info["max_degree"] = max(degrees)
        info["avg_degree"] = sum(degrees) / len(degrees)
    else:
        info["min_degree"] = 0
        info["max_degree"] = 0
        info["avg_degree"] = 0

    return info


def diagnose_instance(
    instance_path: Path,
    max_iterations: int = 10000,
    timeout: int = 60,
    verbose: bool = False,
) -> None:
    """Run diagnostic analysis on a GOTO instance."""
    print("=" * 70)
    print(f"GOTO Instance Diagnostic: {instance_path.name}")
    print("=" * 70)

    # Parse instance
    print("\n1. Parsing instance...")
    parse_start = time.perf_counter()
    problem = parse_dimacs_file(str(instance_path))
    parse_time = time.perf_counter() - parse_start
    print(f"   Parsed in {parse_time * 1000:.1f}ms")

    # Analyze structure
    print("\n2. Analyzing problem structure...")
    info = analyze_problem_structure(problem)
    print(f"   Nodes: {info['nodes']}")
    print(f"   Arcs: {info['arcs']}")
    print(f"   Total supply: {info['total_supply']}")
    print(f"   Supply nodes: {info['supply_nodes']}")
    print(f"   Demand nodes: {info['demand_nodes']}")
    print(f"   Transshipment nodes: {info['transshipment_nodes']}")
    print(
        f"   Arc costs: min={info['min_cost']}, max={info['max_cost']}, avg={info['avg_cost']:.1f}"
    )
    print(
        f"   Arc capacities: min={info['min_capacity']}, max={info['max_capacity']}, avg={info['avg_capacity']:.1f}"
    )
    print(
        f"   Node degrees: min={info['min_degree']}, max={info['max_degree']}, avg={info['avg_degree']:.1f}"
    )

    # Run solver with limited iterations
    print(f"\n3. Running solver (max {max_iterations} iterations, {timeout}s timeout)...")

    options = SolverOptions(
        pricing_strategy="dantzig",  # Use Dantzig's rule (most-negative reduced cost)
        max_iterations=max_iterations,
    )

    solver = NetworkSimplex(problem, options=options)

    solve_start = time.perf_counter()
    try:
        result = solver.solve()
        solve_time = time.perf_counter() - solve_start

        print(f"\n4. Results:")
        print(f"   Status: {result.status}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Solve time: {solve_time * 1000:.1f}ms")
        if result.status == "optimal":
            print(f"   Objective: {result.objective}")

        # Analyze convergence
        if result.iterations >= max_iterations:
            print(f"\n   ⚠ Hit iteration limit ({max_iterations})")
            print(f"   This instance may need special handling or algorithmic improvements")

    except Exception as e:
        solve_time = time.perf_counter() - solve_start
        print(f"\n4. Error after {solve_time:.1f}s:")
        print(f"   {type(e).__name__}: {e}")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose GOTO instance convergence issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "instance",
        type=Path,
        help="Path to GOTO instance file (.min)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum iterations to attempt (default: 10000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if not args.instance.exists():
        print(f"Error: Instance file not found: {args.instance}")
        return 1

    setup_logging(args.verbose)

    diagnose_instance(
        args.instance,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
