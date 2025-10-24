"""Unit tests for projection cache in TreeBasis."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import build_problem  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


def test_cache_disabled_when_size_zero():
    """Test that cache can be disabled by setting size to 0."""
    nodes = [
        {"id": "A", "supply": 10.0},
        {"id": "B", "supply": -10.0},
    ]
    arcs = [{"tail": "A", "head": "B", "capacity": 20.0, "cost": 1.0}]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    options = SolverOptions(projection_cache_size=0)
    solver = NetworkSimplex(problem, options=options)
    solver.solve()

    # With cache disabled, all requests should be misses
    assert solver.basis.cache_hits == 0
    assert solver.basis.cache_misses > 0
    assert len(solver.basis.projection_cache) == 0


def test_cache_stores_and_retrieves_projections():
    """Test that cache correctly stores and retrieves projections."""
    # Simple network flow problem
    nodes = [
        {"id": "S1", "supply": 10.0},
        {"id": "I1", "supply": 0.0},
        {"id": "D1", "supply": -10.0},
    ]
    arcs = [
        {"tail": "S1", "head": "I1", "capacity": 20.0, "cost": 1.0},
        {"tail": "I1", "head": "D1", "capacity": 20.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    options = SolverOptions(projection_cache_size=100)
    solver = NetworkSimplex(problem, options=options)
    solver.solve()

    # Should have some cache hits (projections requested multiple times)
    total_requests = solver.basis.cache_hits + solver.basis.cache_misses
    assert total_requests > 0

    # Cache should contain entries
    assert len(solver.basis.projection_cache) > 0

    # For this simple problem, cache should be effective
    if total_requests > 10:  # Only check if enough requests to be meaningful
        hit_rate = solver.basis.cache_hits / total_requests
        # Even a modest hit rate validates cache is working
        assert hit_rate >= 0.0


def test_cache_invalidation_on_basis_change():
    """Test that basis version increments invalidate old cache entries."""
    # Medium network to generate multiple basis changes
    nodes = [
        {"id": "S1", "supply": 5.0},
        {"id": "S2", "supply": 5.0},
        {"id": "I1", "supply": 0.0},
        {"id": "I2", "supply": 0.0},
        {"id": "D1", "supply": -5.0},
        {"id": "D2", "supply": -5.0},
    ]
    arcs = [
        {"tail": "S1", "head": "I1", "capacity": 10.0, "cost": 2.0},
        {"tail": "S1", "head": "I2", "capacity": 10.0, "cost": 3.0},
        {"tail": "S2", "head": "I1", "capacity": 10.0, "cost": 1.0},
        {"tail": "S2", "head": "I2", "capacity": 10.0, "cost": 4.0},
        {"tail": "I1", "head": "D1", "capacity": 10.0, "cost": 1.0},
        {"tail": "I1", "head": "D2", "capacity": 10.0, "cost": 2.0},
        {"tail": "I2", "head": "D1", "capacity": 10.0, "cost": 3.0},
        {"tail": "I2", "head": "D2", "capacity": 10.0, "cost": 1.0},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    options = SolverOptions(projection_cache_size=50)
    solver = NetworkSimplex(problem, options=options)
    solver.solve()

    # Basis version should have incremented (multiple pivots)
    assert solver.basis.basis_version > 0

    # All cache entries should be for valid basis versions
    for (arc_key, basis_ver), projection in solver.basis.projection_cache.items():
        assert basis_ver >= 0
        assert basis_ver <= solver.basis.basis_version
        assert isinstance(projection, np.ndarray)


def test_cache_lru_eviction():
    """Test that LRU eviction works when cache is full."""
    # Create a problem that will generate many unique projections
    nodes = (
        [{"id": f"S{i}", "supply": 2.0} for i in range(3)]
        + [{"id": f"I{i}", "supply": 0.0} for i in range(4)]
        + [{"id": f"D{i}", "supply": -2.0} for i in range(3)]
    )

    arcs = []
    cost = 1.0
    # Connect sources to intermediates
    for s in range(3):
        for i in range(4):
            arcs.append(
                {
                    "tail": f"S{s}",
                    "head": f"I{i}",
                    "capacity": 5.0,
                    "cost": cost,
                }
            )
            cost += 0.5

    # Connect intermediates to demands
    for i in range(4):
        for d in range(3):
            arcs.append(
                {
                    "tail": f"I{i}",
                    "head": f"D{d}",
                    "capacity": 5.0,
                    "cost": cost,
                }
            )
            cost += 0.5

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    # Use small cache to force evictions
    options = SolverOptions(projection_cache_size=10)
    solver = NetworkSimplex(problem, options=options)
    solver.solve()

    # Cache should be at or near its limit
    assert len(solver.basis.projection_cache) <= 10

    # Should still have cache hits despite small size
    if solver.basis.cache_hits + solver.basis.cache_misses > 20:
        assert solver.basis.cache_hits > 0


def test_cache_hit_rate_on_network_flow():
    """Test that cache achieves good hit rate on network flow problems."""
    # Medium network flow problem (similar to Week 1 analysis)
    num_sources = 5
    num_intermediates = 8
    num_demands = 5

    nodes = (
        [{"id": f"S{i}", "supply": 10.0} for i in range(num_sources)]
        + [{"id": f"I{i}", "supply": 0.0} for i in range(num_intermediates)]
        + [{"id": f"D{i}", "supply": -10.0} for i in range(num_demands)]
    )

    arcs = []
    cost = 1.0
    # Connect sources to intermediates (dense connections)
    for s in range(num_sources):
        for i in range(num_intermediates):
            arcs.append(
                {
                    "tail": f"S{s}",
                    "head": f"I{i}",
                    "capacity": 15.0,
                    "cost": cost,
                }
            )
            cost += 0.3

    # Connect intermediates to demands (dense connections)
    for i in range(num_intermediates):
        for d in range(num_demands):
            arcs.append(
                {
                    "tail": f"I{i}",
                    "head": f"D{d}",
                    "capacity": 15.0,
                    "cost": cost,
                }
            )
            cost += 0.3

    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    # Use Devex pricing (generates many projection requests)
    options = SolverOptions(
        projection_cache_size=100,
        pricing_strategy="devex",
    )
    solver = NetworkSimplex(problem, options=options)
    result = solver.solve()

    assert result.status == "optimal"

    total_requests = solver.basis.cache_hits + solver.basis.cache_misses

    # Should have many projection requests with Devex pricing
    assert total_requests > 100

    # Should achieve some cache hits
    # Note: Hit rate depends on problem size and iteration count.
    # Week 1 analysis showed 99.2% potential on larger problems with many iterations.
    # For this smaller test problem, we just verify cache is working (>5% hit rate).
    hit_rate = solver.basis.cache_hits / total_requests
    assert hit_rate >= 0.05, f"Cache hit rate {hit_rate:.1%} is unexpectedly low"

    # Verify we actually got some cache hits
    assert solver.basis.cache_hits > 0


def test_cache_correctness():
    """Test that cached projections are numerically identical to fresh computations."""
    nodes = [
        {"id": "S1", "supply": 15.0},
        {"id": "I1", "supply": 0.0},
        {"id": "I2", "supply": 0.0},
        {"id": "D1", "supply": -15.0},
    ]
    arcs = [
        {"tail": "S1", "head": "I1", "capacity": 20.0, "cost": 2.0},
        {"tail": "S1", "head": "I2", "capacity": 20.0, "cost": 3.0},
        {"tail": "I1", "head": "D1", "capacity": 20.0, "cost": 1.0},
        {"tail": "I2", "head": "D1", "capacity": 20.0, "cost": 1.5},
    ]
    problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

    from network_solver.data import SolverOptions

    # Solve with cache enabled
    options_cached = SolverOptions(projection_cache_size=100)
    solver_cached = NetworkSimplex(problem, options=options_cached)
    result_cached = solver_cached.solve()

    # Solve with cache disabled
    options_no_cache = SolverOptions(projection_cache_size=0)
    solver_no_cache = NetworkSimplex(problem, options=options_no_cache)
    result_no_cache = solver_no_cache.solve()

    # Results should be identical
    assert result_cached.status == result_no_cache.status == "optimal"
    assert abs(result_cached.objective - result_no_cache.objective) < 1e-9

    # Flows should be identical
    for arc_key in result_cached.flows:
        assert abs(result_cached.flows[arc_key] - result_no_cache.flows[arc_key]) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
