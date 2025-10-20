#!/usr/bin/env python3
"""Example demonstrating numeric validation and diagnostics features.

This example shows how to:
1. Analyze numeric properties of problems
2. Detect and handle ill-conditioned problems
3. Monitor convergence during solving
4. Use diagnostics to identify issues
"""

from network_solver import (
    ConvergenceMonitor,
    analyze_numeric_properties,
    build_problem,
    solve_min_cost_flow,
    validate_numeric_properties,
)


def example_well_conditioned_problem():
    """Example 1: Well-conditioned problem passes validation."""
    print("\n" + "=" * 70)
    print("Example 1: Well-Conditioned Problem")
    print("=" * 70)

    problem = build_problem(
        nodes=[
            {"id": "factory", "supply": 100.0},
            {"id": "warehouse", "supply": -100.0},
        ],
        arcs=[
            {"tail": "factory", "head": "warehouse", "capacity": 150.0, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Analyze numeric properties
    analysis = analyze_numeric_properties(problem)

    print("\nNumeric Analysis:")
    print(f"  Well-conditioned: {analysis.is_well_conditioned}")
    print(f"  Cost range: {analysis.cost_range:.2e}")
    print(f"  Capacity range: {analysis.capacity_range:.2e}")
    print(f"  Supply range: {analysis.supply_range:.2e}")
    print(f"  Extreme values: {analysis.has_extreme_values}")
    print(f"  Warnings: {len(analysis.warnings)}")
    print(f"  Recommended tolerance: {analysis.recommended_tolerance:.2e}")

    # Solve
    result = solve_min_cost_flow(problem)
    print("\nSolution:")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")


def example_ill_conditioned_problem():
    """Example 2: Ill-conditioned problem triggers warnings."""
    print("\n" + "=" * 70)
    print("Example 2: Ill-Conditioned Problem (Wide Coefficient Range)")
    print("=" * 70)

    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1e-6},  # Very small cost
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 1e6},  # Very large cost
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Analyze numeric properties
    analysis = analyze_numeric_properties(problem)

    print("\nNumeric Analysis:")
    print(f"  Well-conditioned: {analysis.is_well_conditioned}")
    print(f"  Cost range: {analysis.cost_range:.2e}")
    print(f"  Recommended tolerance: {analysis.recommended_tolerance:.2e}")

    print(f"\nWarnings ({len(analysis.warnings)}):")
    for warning in analysis.warnings:
        print(f"  [{warning.severity.upper()}] {warning.category}")
        print(f"    Message: {warning.message}")
        print(f"    Recommendation: {warning.recommendation}")

    # Still solve (with recommended tolerance)
    from network_solver import SolverOptions

    options = SolverOptions(tolerance=analysis.recommended_tolerance)
    result = solve_min_cost_flow(problem, options=options)

    print("\nSolution (with adjusted tolerance):")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")
    print(f"  Iterations: {result.iterations}")


def example_extreme_values():
    """Example 3: Extreme values detected."""
    print("\n" + "=" * 70)
    print("Example 3: Extreme Values Detection")
    print("=" * 70)

    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 1e11},  # Very large supply
            {"id": "B", "supply": -1e11},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 2e11, "cost": 1e11},  # Extreme values
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Validate with warnings
    print("\nValidating problem (warnings enabled)...")
    validate_numeric_properties(problem, strict=False, warn=True)

    # Analyze
    analysis = analyze_numeric_properties(problem)
    print(f"\nExtreme values detected: {analysis.has_extreme_values}")
    print(f"Number of warnings: {len(analysis.warnings)}")

    # Show recommendations
    medium_warnings = [w for w in analysis.warnings if w.severity == "medium"]
    if medium_warnings:
        print("\nRecommendations:")
        for warning in medium_warnings[:3]:  # Show first 3
            print(f"  • {warning.recommendation}")


def example_convergence_monitoring():
    """Example 4: Monitor convergence during solving."""
    print("\n" + "=" * 70)
    print("Example 4: Convergence Monitoring")
    print("=" * 70)

    # Create a moderately complex problem
    nodes = [
        {"id": "factory1", "supply": 100.0},
        {"id": "factory2", "supply": 150.0},
        {"id": "warehouse1", "supply": -120.0},
        {"id": "warehouse2", "supply": -130.0},
    ]
    arcs = [
        {"tail": "factory1", "head": "warehouse1", "capacity": 200.0, "cost": 5.0},
        {"tail": "factory1", "head": "warehouse2", "capacity": 200.0, "cost": 3.0},
        {"tail": "factory2", "head": "warehouse1", "capacity": 200.0, "cost": 2.0},
        {"tail": "factory2", "head": "warehouse2", "capacity": 200.0, "cost": 4.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    # Set up convergence monitor
    monitor = ConvergenceMonitor(window_size=50, stall_threshold=1e-8)

    def track_convergence(info):
        """Progress callback to track convergence."""
        monitor.record_iteration(
            objective=info.objective_estimate,
            is_degenerate=False,  # Simplified for example
            iteration=info.iteration,
        )

        # Report progress
        phase_name = "Phase 1" if info.phase == 1 else "Phase 2"
        print(
            f"  {phase_name} | Iter {info.iteration:4d} | "
            f"Obj: ${info.objective_estimate:10.2f} | "
            f"Time: {info.elapsed_time:.2f}s"
        )

        # Check for issues
        if monitor.is_stalled():
            print("    ⚠️  Stalling detected")

        if monitor.is_highly_degenerate():
            print(f"    ⚠️  High degeneracy: {monitor.get_degeneracy_ratio():.1%}")

    print("\nSolving with progress monitoring:")
    result = solve_min_cost_flow(
        problem,
        progress_callback=track_convergence,
        progress_interval=5,  # Report every 5 iterations
    )

    # Show final diagnostics
    diagnostics = monitor.get_diagnostic_summary()
    print("\nFinal Diagnostics:")
    print(f"  Total pivots: {diagnostics['total_pivots']}")
    print(f"  Degenerate pivots: {diagnostics['degenerate_pivots']}")
    print(f"  Degeneracy ratio: {diagnostics['degeneracy_ratio']:.2%}")
    print(f"  Final stalled: {diagnostics['is_stalled']}")
    print(f"  Recent improvement: {diagnostics['recent_improvement']:.2e}")

    print("\nSolution:")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:.2f}")
    print(f"  Total iterations: {result.iterations}")


def example_strict_validation():
    """Example 5: Strict validation prevents solving bad problems."""
    print("\n" + "=" * 70)
    print("Example 5: Strict Validation Mode")
    print("=" * 70)

    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 1e-8, "cost": 1.0},  # Tiny capacity
            {"tail": "B", "head": "C", "capacity": 1e10, "cost": 1.0},  # Huge capacity
        ],
        directed=True,
        tolerance=1e-6,
    )

    print("\nAttempting strict validation...")
    try:
        validate_numeric_properties(problem, strict=True, warn=False)
        print("  ✓ Problem passed strict validation")
    except ValueError as e:
        print("  ✗ Validation failed (as expected)")
        print(f"\nError message:\n{e}")
        print("\n→ This prevents accidentally solving an ill-conditioned problem.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NUMERIC VALIDATION AND DIAGNOSTICS EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the numeric validation and convergence")
    print("monitoring features added to improve solver reliability.\n")

    example_well_conditioned_problem()
    example_ill_conditioned_problem()
    example_extreme_values()
    example_convergence_monitoring()
    example_strict_validation()

    print("\n" + "=" * 70)
    print("For more information, see docs/troubleshooting.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
