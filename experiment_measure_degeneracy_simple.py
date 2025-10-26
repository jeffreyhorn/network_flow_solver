#!/usr/bin/env python3
"""Simple script to measure degeneracy rates with logging enabled."""

import logging
import sys

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow

# Enable INFO logging to see degeneracy stats
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def test_problem(problem_file: str, name: str):
    """Test a problem and show degeneracy stats."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print("=" * 70)

    problem = parse_dimacs_file(problem_file)
    result = solve_min_cost_flow(problem, max_iterations=10000)

    print(f"\nResult: {result.status}, Iterations: {result.iterations}")
    print("(Degeneracy stats shown in log above)")


def main():
    """Measure degeneracy on key problems."""

    print("=" * 70)
    print("DEGENERACY MEASUREMENT")
    print("=" * 70)
    print("\nLook for 'degenerate_pivots' and 'degeneracy_rate' in the logs below.")
    print("High degeneracy (>20%) suggests we're wasting iterations.")

    problems = [
        ("benchmarks/problems/lemon/gridgen/gridgen_8_08a.min", "gridgen_8_08a (256 nodes)"),
        ("benchmarks/problems/lemon/gridgen/gridgen_8_09a.min", "gridgen_8_09a (512 nodes)"),
        ("benchmarks/problems/lemon/goto/goto_8_08a.min", "goto_8_08a (256 nodes)"),
        ("benchmarks/problems/lemon/goto/goto_8_09a.min", "goto_8_09a (512 nodes)"),
        ("benchmarks/problems/lemon/netgen/netgen_8_08a.min", "netgen_8_08a (256 nodes)"),
    ]

    for problem_file, name in problems:
        test_problem(problem_file, name)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nReview the degeneracy_rate values above to determine if degeneracy")
    print("is a significant factor limiting performance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
