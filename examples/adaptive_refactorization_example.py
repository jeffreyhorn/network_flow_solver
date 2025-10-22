"""
Demonstrates adaptive basis refactorization for improved numerical stability.

Adaptive refactorization monitors the condition number of the basis matrix
and triggers rebuilds when numerical issues are detected. This helps maintain
accuracy and stability, especially for ill-conditioned problems.

This example shows:
- How to enable/disable adaptive refactorization
- Configuring condition number thresholds
- Adjusting adaptive ft_update_limit bounds
- Monitoring condition number history
- Comparing adaptive vs fixed refactorization strategies
"""

import argparse
import logging

from network_solver import SolverOptions, build_problem, solve_min_cost_flow


def setup_logging(verbose: int = 0):
    """Configure logging based on verbosity level."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(message)s",
    )


def create_ill_conditioned_problem() -> tuple:
    """Create a problem with wide value ranges that may cause conditioning issues.

    Returns a transportation problem with micro-costs and macro-capacities to
    demonstrate adaptive refactorization benefits.
    """
    # Transportation problem with extreme value ranges
    # Small costs (0.001 - 0.005) with large capacities (millions)
    nodes = [
        {"id": "factory_1", "supply": 1_000_000.0},
        {"id": "factory_2", "supply": 2_000_000.0},
        {"id": "factory_3", "supply": 1_500_000.0},
        {"id": "warehouse_1", "supply": -1_800_000.0},
        {"id": "warehouse_2", "supply": -1_200_000.0},
        {"id": "warehouse_3", "supply": -1_500_000.0},
    ]

    arcs = [
        # From factory 1
        {"tail": "factory_1", "head": "warehouse_1", "capacity": 2_000_000.0, "cost": 0.001},
        {"tail": "factory_1", "head": "warehouse_2", "capacity": 2_000_000.0, "cost": 0.003},
        {"tail": "factory_1", "head": "warehouse_3", "capacity": 2_000_000.0, "cost": 0.002},
        # From factory 2
        {"tail": "factory_2", "head": "warehouse_1", "capacity": 3_000_000.0, "cost": 0.002},
        {"tail": "factory_2", "head": "warehouse_2", "capacity": 3_000_000.0, "cost": 0.001},
        {"tail": "factory_2", "head": "warehouse_3", "capacity": 3_000_000.0, "cost": 0.004},
        # From factory 3
        {"tail": "factory_3", "head": "warehouse_1", "capacity": 2_500_000.0, "cost": 0.004},
        {"tail": "factory_3", "head": "warehouse_2", "capacity": 2_500_000.0, "cost": 0.002},
        {"tail": "factory_3", "head": "warehouse_3", "capacity": 2_500_000.0, "cost": 0.001},
    ]

    return nodes, arcs


def create_standard_problem() -> tuple:
    """Create a well-conditioned standard transportation problem."""
    nodes = [
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 150.0},
        {"id": "warehouse_1", "supply": -80.0},
        {"id": "warehouse_2", "supply": -120.0},
        {"id": "warehouse_3", "supply": -50.0},
    ]

    arcs = [
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 100.0, "cost": 2.5},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "factory_a", "head": "warehouse_3", "capacity": 100.0, "cost": 1.5},
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 150.0, "cost": 1.8},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 150.0, "cost": 2.2},
        {"tail": "factory_b", "head": "warehouse_3", "capacity": 150.0, "cost": 2.8},
    ]

    return nodes, arcs


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def print_subsection(title: str):
    """Print a subsection header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def main():
    """Demonstrate adaptive refactorization features."""
    parser = argparse.ArgumentParser(description="Adaptive refactorization example")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    print_section("ADAPTIVE REFACTORIZATION DEMONSTRATION")

    # =========================================================================
    # Example 1: Default Adaptive Settings
    # =========================================================================
    print_subsection("Example 1: Default Adaptive Settings")
    print("Using default adaptive refactorization settings:")
    print("  - adaptive_refactorization: True (enabled)")
    print("  - condition_number_threshold: 1e12")
    print("  - adaptive_ft_min: 20, adaptive_ft_max: 200")
    print()

    nodes, arcs = create_standard_problem()
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # Default settings enable adaptive refactorization
    options = SolverOptions()
    result = solve_min_cost_flow(problem, options=options)

    print(f"Status: {result.status}")
    print(f"Objective: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()
    print("✓ Adaptive refactorization is enabled by default for numerical stability")

    # =========================================================================
    # Example 2: Disable Adaptive Refactorization
    # =========================================================================
    print_subsection("Example 2: Disable Adaptive Refactorization")
    print("Disabling adaptive refactorization (use fixed ft_update_limit only):")
    print()

    options = SolverOptions(adaptive_refactorization=False, ft_update_limit=64)
    result = solve_min_cost_flow(problem, options=options)

    print(f"Status: {result.status}")
    print(f"Objective: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()
    print("ℹ Solver uses fixed ft_update_limit (64) without condition monitoring")

    # =========================================================================
    # Example 3: Custom Condition Number Threshold
    # =========================================================================
    print_subsection("Example 3: Custom Condition Number Threshold")
    print("Setting a lower condition number threshold for more aggressive rebuilds:")
    print("  - condition_number_threshold: 1e10 (more conservative)")
    print()

    options = SolverOptions(
        adaptive_refactorization=True,
        condition_number_threshold=1e10,  # More aggressive (lower threshold)
    )
    result = solve_min_cost_flow(problem, options=options)

    print(f"Status: {result.status}")
    print(f"Objective: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()
    print("✓ Lower threshold triggers more frequent rebuilds for better stability")

    # =========================================================================
    # Example 4: Custom Adaptive FT Limits
    # =========================================================================
    print_subsection("Example 4: Custom Adaptive FT Bounds")
    print("Customizing the adaptive ft_update_limit range:")
    print("  - adaptive_ft_min: 10 (minimum limit)")
    print("  - adaptive_ft_max: 100 (maximum limit)")
    print()

    options = SolverOptions(
        adaptive_refactorization=True,
        adaptive_ft_min=10,
        adaptive_ft_max=100,
    )
    result = solve_min_cost_flow(problem, options=options)

    print(f"Status: {result.status}")
    print(f"Objective: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()
    print("✓ Adaptive limit constrained to [10, 100] range")

    # =========================================================================
    # Example 5: Ill-Conditioned Problem
    # =========================================================================
    print_subsection("Example 5: Ill-Conditioned Problem")
    print("Testing with a problem that has extreme value ranges:")
    print("  - Costs: 0.001 to 0.005 (micro-scale)")
    print("  - Capacities: millions of units (macro-scale)")
    print("  - Range: ~9 orders of magnitude difference")
    print()

    nodes, arcs = create_ill_conditioned_problem()
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    # With adaptive refactorization (default)
    print("Solving WITH adaptive refactorization:")
    options_adaptive = SolverOptions(adaptive_refactorization=True)
    result_adaptive = solve_min_cost_flow(problem, options=options_adaptive)

    print(f"  Status: {result_adaptive.status}")
    print(f"  Objective: ${result_adaptive.objective:,.2f}")
    print(f"  Iterations: {result_adaptive.iterations}")

    # Without adaptive refactorization
    print()
    print("Solving WITHOUT adaptive refactorization:")
    options_fixed = SolverOptions(adaptive_refactorization=False, ft_update_limit=64)
    result_fixed = solve_min_cost_flow(problem, options=options_fixed)

    print(f"  Status: {result_fixed.status}")
    print(f"  Objective: ${result_fixed.objective:,.2f}")
    print(f"  Iterations: {result_fixed.iterations}")

    print()
    if result_adaptive.objective == result_fixed.objective:
        print("✓ Both approaches found the same optimal solution")
    print("ℹ Adaptive refactorization helps maintain stability for ill-conditioned problems")
    print("  Note: Automatic scaling is also enabled by default, which helps with this problem")

    # =========================================================================
    # Example 6: Very Tight FT Bounds
    # =========================================================================
    print_subsection("Example 6: Very Tight Adaptive Bounds")
    print("Using very tight bounds to force frequent adaptation:")
    print("  - adaptive_ft_min: 20")
    print("  - adaptive_ft_max: 30")
    print()

    nodes, arcs = create_standard_problem()
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    options = SolverOptions(
        adaptive_refactorization=True,
        adaptive_ft_min=20,
        adaptive_ft_max=30,
    )
    result = solve_min_cost_flow(problem, options=options)

    print(f"Status: {result.status}")
    print(f"Objective: ${result.objective:,.2f}")
    print(f"Iterations: {result.iterations}")
    print()
    print("✓ Narrow adaptive range allows only small adjustments")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY: WHEN TO USE ADAPTIVE REFACTORIZATION")

    print("✓ ENABLE adaptive refactorization when:")
    print("  • Working with ill-conditioned problems (wide value ranges)")
    print("  • Numerical stability is critical")
    print("  • You need automatic tuning for different problem types")
    print("  • Default behavior (recommended for most users)")
    print()
    print("✗ DISABLE adaptive refactorization when:")
    print("  • You have well-conditioned problems with narrow value ranges")
    print("  • You want predictable, fixed refactorization behavior")
    print("  • Testing or debugging specific numerical behaviors")
    print("  • You've manually tuned ft_update_limit for your workload")
    print()
    print("CONFIGURATION GUIDELINES:")
    print("  • condition_number_threshold:")
    print("    - Lower (1e10): More conservative, more rebuilds, better stability")
    print("    - Higher (1e14): More aggressive, fewer rebuilds, faster but less stable")
    print("    - Default (1e12): Good balance for most problems")
    print()
    print("  • adaptive_ft_min/max:")
    print("    - Narrow range [20, 40]: Tight control, minimal adaptation")
    print("    - Wide range [10, 200]: More flexibility, better adaptation")
    print("    - Default [20, 200]: Works well for diverse problem types")
    print()
    print("NOTE: Adaptive refactorization works in combination with:")
    print("  • Automatic problem scaling (for extreme value ranges)")
    print("  • Forrest-Tomlin basis updates (for efficient refactorization)")
    print("  • Both features are enabled by default for best results")
    print()


if __name__ == "__main__":
    main()
