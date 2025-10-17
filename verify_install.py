#!/usr/bin/env python3
"""Quick verification script to test package installation."""

import sys


def main():
    """Verify the network_solver package is properly installed."""
    print("=" * 60)
    print("Network Flow Solver - Installation Verification")
    print("=" * 60)

    # Test 1: Import package
    print("\n[1/5] Testing package import...")
    try:
        import network_solver

        print(f"    ✓ Package imported successfully")
        print(f"    ✓ Version: {network_solver.__version__}")
    except ImportError as e:
        print(f"    ✗ Failed to import package: {e}")
        return 1

    # Test 2: Check API exports
    print("\n[2/5] Testing API exports...")
    try:
        from network_solver import load_problem, solve_min_cost_flow, save_result

        print(f"    ✓ All public APIs available: {network_solver.__all__}")
    except ImportError as e:
        print(f"    ✗ Failed to import APIs: {e}")
        return 1

    # Test 3: Check dependencies
    print("\n[3/5] Testing dependencies...")
    try:
        import numpy as np
        import scipy

        print(f"    ✓ NumPy {np.__version__}")
        print(f"    ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"    ✗ Missing dependency: {e}")
        return 1

    # Test 4: Check optional UMFPACK
    print("\n[4/5] Testing optional dependencies...")
    try:
        import scikits.umfpack

        print(f"    ✓ scikit-umfpack available (enhanced performance)")
    except ImportError:
        print(f"    ⚠ scikit-umfpack not available (optional, Linux/macOS only)")

    # Test 5: Run a simple solve
    print("\n[5/5] Testing solver with simple problem...")
    try:
        from network_solver.data import build_problem

        nodes = [
            {"id": "s", "supply": 10.0},
            {"id": "t", "supply": -10.0},
        ]
        arcs = [
            {"tail": "s", "head": "t", "capacity": 10.0, "cost": 1.0},
        ]

        problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)

        if result.status == "optimal" and result.objective == 10.0:
            print(f"    ✓ Solver works correctly")
            print(f"    ✓ Status: {result.status}, Objective: {result.objective}")
        else:
            print(f"    ✗ Unexpected result: {result.status}, {result.objective}")
            return 1
    except Exception as e:
        print(f"    ✗ Solver test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Success
    print("\n" + "=" * 60)
    print("✓ All tests passed! Package is ready to use.")
    print("=" * 60)
    print("\nTry running an example:")
    print("    python examples/solve_example.py")
    print("\nOr run the test suite:")
    print("    pytest")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
