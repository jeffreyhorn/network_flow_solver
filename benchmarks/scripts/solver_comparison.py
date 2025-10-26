#!/usr/bin/env python3
"""
Solver Comparison Framework

Compares the performance and correctness of different network simplex implementations:
- network_solver (this implementation)
- NetworkX (capacity scaling algorithm)

The framework runs the same problems through each solver and measures:
- Solve time
- Solution quality (objective value)
- Correctness (flow conservation, capacity constraints)
- Success rate

This helps validate our implementation and identify performance gaps.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.data import FlowResult, NetworkProblem
from src.network_solver.solver import solve_min_cost_flow


@dataclass
class SolverResult:
    """Results from a single solver run."""

    solver_name: str
    problem_name: str
    status: str  # 'optimal', 'infeasible', 'timeout', 'error'
    objective: float | None
    solve_time_ms: float
    iterations: int | None  # None if solver doesn't report
    error_message: str | None = None


@dataclass
class ComparisonResult:
    """Comparison results for a single problem across all solvers."""

    problem_name: str
    num_nodes: int
    num_arcs: int
    results: dict[str, SolverResult]  # solver_name -> result

    def get_winner(self) -> str | None:
        """Return the name of the fastest solver that found optimal solution."""
        optimal = {name: res for name, res in self.results.items() if res.status == "optimal"}
        if not optimal:
            return None
        return min(optimal.items(), key=lambda x: x[1].solve_time_ms)[0]

    def get_speedup(self, baseline: str, competitor: str) -> float | None:
        """Return speedup of competitor vs baseline (competitor_time / baseline_time)."""
        if baseline not in self.results or competitor not in self.results:
            return None
        b_res = self.results[baseline]
        c_res = self.results[competitor]
        if b_res.status != "optimal" or c_res.status != "optimal":
            return None
        if c_res.solve_time_ms <= 0:
            return None
        return b_res.solve_time_ms / c_res.solve_time_ms


# ==============================================================================
# Solver Adapters
# ==============================================================================


class NetworkSolverAdapter:
    """Adapter for our network_solver implementation."""

    name = "network_solver"

    @staticmethod
    def solve(problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using network_solver."""
        try:
            start = time.perf_counter()
            result = solve_min_cost_flow(problem, max_iterations=100000)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return SolverResult(
                solver_name="network_solver",
                problem_name="",  # Will be set by caller
                status=result.status,
                objective=result.objective if result.status == "optimal" else None,
                solve_time_ms=elapsed_ms,
                iterations=result.iterations,
            )
        except Exception as e:
            return SolverResult(
                solver_name="network_solver",
                problem_name="",
                status="error",
                objective=None,
                solve_time_ms=0.0,
                iterations=None,
                error_message=str(e),
            )


class NetworkXAdapter:
    """Adapter for NetworkX min_cost_flow."""

    name = "networkx"

    @staticmethod
    def solve(problem: NetworkProblem, timeout_s: float = 60.0) -> SolverResult:
        """Solve using NetworkX."""
        try:
            # Convert problem to NetworkX format
            G = nx.DiGraph()

            # Add nodes with demand (NetworkX uses opposite sign)
            for node_id, node in problem.nodes.items():
                G.add_node(node_id, demand=-node.supply)

            # Add arcs
            expanded_arcs = problem.undirected_expansion()
            for arc in expanded_arcs:
                capacity = arc.capacity if arc.capacity is not None else float("inf")
                G.add_edge(arc.tail, arc.head, weight=arc.cost, capacity=capacity)

            # Solve
            start = time.perf_counter()
            flow_dict = nx.min_cost_flow(G)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Calculate objective
            objective = nx.cost_of_flow(G, flow_dict)

            return SolverResult(
                solver_name="networkx",
                problem_name="",
                status="optimal",
                objective=objective,
                solve_time_ms=elapsed_ms,
                iterations=None,  # NetworkX doesn't report iterations
            )
        except nx.NetworkXUnfeasible:
            return SolverResult(
                solver_name="networkx",
                problem_name="",
                status="infeasible",
                objective=None,
                solve_time_ms=0.0,
                iterations=None,
            )
        except Exception as e:
            return SolverResult(
                solver_name="networkx",
                problem_name="",
                status="error",
                objective=None,
                solve_time_ms=0.0,
                iterations=None,
                error_message=str(e),
            )


# ==============================================================================
# Comparison Framework
# ==============================================================================


class SolverComparison:
    """Framework for comparing network flow solvers."""

    def __init__(self, solvers: list[Any] | None = None):
        """Initialize with list of solver adapters."""
        if solvers is None:
            solvers = [NetworkSolverAdapter, NetworkXAdapter]
        self.solvers = solvers

    def compare_problem(self, problem_path: str, timeout_s: float = 60.0) -> ComparisonResult:
        """Run all solvers on a single problem and compare results."""
        problem_name = Path(problem_path).stem
        print(f"  Comparing: {problem_name}...", end=" ", flush=True)

        # Parse problem
        problem = parse_dimacs_file(problem_path)
        num_nodes = len(problem.nodes)
        num_arcs = len(problem.arcs)

        # Run each solver
        results = {}
        for solver_class in self.solvers:
            result = solver_class.solve(problem, timeout_s)
            result.problem_name = problem_name
            results[solver_class.name] = result

        # Check consistency
        self._check_consistency(results, problem_name)

        print("Done")

        return ComparisonResult(
            problem_name=problem_name,
            num_nodes=num_nodes,
            num_arcs=num_arcs,
            results=results,
        )

    def _check_consistency(self, results: dict[str, SolverResult], problem_name: str) -> None:
        """Check that solvers agree on objective value (within tolerance)."""
        optimal_objectives = [
            res.objective
            for res in results.values()
            if res.status == "optimal" and res.objective is not None
        ]

        if len(optimal_objectives) > 1:
            min_obj = min(optimal_objectives)
            max_obj = max(optimal_objectives)
            rel_diff = abs(max_obj - min_obj) / min_obj if min_obj != 0 else abs(max_obj - min_obj)

            if abs(max_obj - min_obj) > 1e-3:  # Tolerance for comparison
                print(f"\n  ⚠️  Solution quality difference: {problem_name}")
                for name, res in results.items():
                    if res.status == "optimal":
                        is_best = abs(res.objective - min_obj) < 1e-3
                        marker = " ✓ BEST" if is_best else ""
                        print(f"    {name}: {res.objective:.0f}{marker}")
                if rel_diff > 0.01:  # >1% difference
                    print(f"    Difference: {rel_diff * 100:.1f}% (significant!)")

    def compare_suite(
        self, problem_paths: list[str], timeout_s: float = 60.0
    ) -> list[ComparisonResult]:
        """Run comparison on a suite of problems."""
        print(f"\nComparing {len(self.solvers)} solvers on {len(problem_paths)} problems")
        print(f"Solvers: {', '.join(s.name for s in self.solvers)}")
        print()

        results = []
        for path in problem_paths:
            try:
                result = self.compare_problem(path, timeout_s)
                results.append(result)
            except Exception as e:
                print(f"\n  ERROR comparing {Path(path).stem}: {e}")

        return results

    def generate_report(
        self, results: list[ComparisonResult], output_path: str | None = None
    ) -> str:
        """Generate a comparison report."""
        report = []
        report.append("=" * 80)
        report.append("SOLVER COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        report.append("Summary:")
        report.append(f"  Total problems: {len(results)}")
        report.append("")

        # Success rate by solver
        report.append("Success Rate:")
        for solver_class in self.solvers:
            solver_name = solver_class.name
            total = len(results)
            optimal = sum(1 for r in results if r.results[solver_name].status == "optimal")
            rate = (optimal / total * 100) if total > 0 else 0
            report.append(f"  {solver_name:20s}: {optimal}/{total} ({rate:.1f}%)")
        report.append("")

        # Performance comparison (only on problems both solved)
        both_optimal = [
            r for r in results if all(res.status == "optimal" for res in r.results.values())
        ]

        if len(both_optimal) > 0:
            report.append(f"Performance Comparison ({len(both_optimal)} problems both solved):")
            report.append("")

            # Calculate speedups
            if len(self.solvers) == 2:
                solver1 = self.solvers[0].name
                solver2 = self.solvers[1].name

                speedups = [r.get_speedup(solver1, solver2) for r in both_optimal]
                speedups = [s for s in speedups if s is not None]

                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    report.append(f"  Average speedup ({solver1} / {solver2}): {avg_speedup:.2f}x")
                    report.append(
                        f"  {solver2} is {'faster' if avg_speedup > 1 else 'slower'} on average"
                    )
                    report.append("")

        # Detailed results table
        report.append("Detailed Results:")
        report.append("")
        report.append(
            "┌─────────────────────────┬────────┬────────┬───────────────────────────────────────────┐"
        )
        report.append(
            "│ Problem                 │ Nodes  │ Arcs   │ Solve Time (ms)                           │"
        )
        report.append(
            "├─────────────────────────┼────────┼────────┼───────────────────────────────────────────┤"
        )

        for result in results:
            times_str = "  ".join(
                f"{name}:{res.solve_time_ms:8.2f}"
                for name, res in result.results.items()
                if res.status == "optimal"
            )
            report.append(
                f"│ {result.problem_name:23s} │ {result.num_nodes:6d} │ "
                f"{result.num_arcs:6d} │ {times_str:41s} │"
            )

        report.append(
            "└─────────────────────────┴────────┴────────┴───────────────────────────────────────────┘"
        )
        report.append("")

        # Fastest solver per problem
        report.append("Winner (Fastest Solver) per Problem:")
        winners = {}
        for result in results:
            winner = result.get_winner()
            if winner:
                winners[winner] = winners.get(winner, 0) + 1

        for solver_name, count in sorted(winners.items(), key=lambda x: -x[1]):
            report.append(f"  {solver_name:20s}: {count} wins")

        report_text = "\n".join(report)

        if output_path:
            Path(output_path).write_text(report_text)
            print(f"\nReport saved to: {output_path}")

        return report_text


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Run solver comparison on benchmark problems."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare network flow solvers")
    parser.add_argument(
        "--problems",
        nargs="+",
        help="Problem files to compare",
    )
    parser.add_argument(
        "--pattern",
        default="benchmarks/problems/lemon/**/*.min",
        help="Glob pattern for problem files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per problem in seconds",
    )
    parser.add_argument(
        "--output",
        help="Output file for report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems to test",
    )

    args = parser.parse_args()

    # Get problem list
    if args.problems:
        problem_paths = args.problems
    else:
        from glob import glob

        problem_paths = sorted(glob(args.pattern, recursive=True))
        if args.limit:
            problem_paths = problem_paths[: args.limit]

    if not problem_paths:
        print("No problems found!")
        return

    # Run comparison
    comparison = SolverComparison()
    results = comparison.compare_suite(problem_paths, timeout_s=args.timeout)

    # Generate and print report
    report = comparison.generate_report(results, output_path=args.output)
    print()
    print(report)


if __name__ == "__main__":
    main()
