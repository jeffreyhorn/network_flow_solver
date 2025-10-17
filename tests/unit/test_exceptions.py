"""Tests for custom exception hierarchy."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import (  # noqa: E402
    InfeasibleProblemError,
    InvalidProblemError,
    IterationLimitError,
    NetworkSolverError,
    NumericalInstabilityError,
    SolverConfigurationError,
    UnboundedProblemError,
)
from network_solver.data import build_problem  # noqa: E402
from network_solver.io import load_problem  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


def test_all_exceptions_inherit_from_base():
    """Test that all custom exceptions inherit from NetworkSolverError."""
    assert issubclass(InvalidProblemError, NetworkSolverError)
    assert issubclass(InfeasibleProblemError, NetworkSolverError)
    assert issubclass(UnboundedProblemError, NetworkSolverError)
    assert issubclass(NumericalInstabilityError, NetworkSolverError)
    assert issubclass(IterationLimitError, NetworkSolverError)
    assert issubclass(SolverConfigurationError, NetworkSolverError)


def test_base_exception_is_exception():
    """Test that NetworkSolverError inherits from Exception."""
    assert issubclass(NetworkSolverError, Exception)


def test_invalid_problem_unbalanced_supply():
    """Test InvalidProblemError raised for unbalanced supply."""
    nodes = [
        {"id": "s", "supply": 5.0},
        {"id": "t", "supply": -4.0},
    ]
    arcs = []

    with pytest.raises(InvalidProblemError) as exc_info:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    assert "unbalanced" in str(exc_info.value).lower()
    assert "5.0" in str(exc_info.value) or "1.0" in str(exc_info.value)


def test_invalid_problem_missing_node():
    """Test InvalidProblemError raised for missing node reference."""
    nodes = [
        {"id": "s", "supply": 1.0},
        {"id": "t", "supply": -1.0},
    ]
    arcs = [
        {"tail": "s", "head": "missing", "capacity": 5.0, "cost": 1.0},
    ]

    with pytest.raises(InvalidProblemError) as exc_info:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    assert "missing" in str(exc_info.value).lower()


def test_invalid_problem_self_loop():
    """Test InvalidProblemError raised for self-loop."""
    from network_solver.data import Arc

    with pytest.raises(InvalidProblemError) as exc_info:
        Arc(tail="a", head="a", capacity=5.0, cost=1.0)

    assert "self-loop" in str(exc_info.value).lower()


def test_invalid_problem_capacity_less_than_lower():
    """Test InvalidProblemError raised for capacity < lower bound."""
    from network_solver.data import Arc

    with pytest.raises(InvalidProblemError) as exc_info:
        Arc(tail="a", head="b", capacity=5.0, cost=1.0, lower=10.0)

    assert "capacity" in str(exc_info.value).lower()
    assert "lower" in str(exc_info.value).lower()


def test_invalid_problem_duplicate_nodes():
    """Test InvalidProblemError raised for duplicate node IDs."""
    nodes = [
        {"id": "dup", "supply": 0.0},
        {"id": "dup", "supply": 0.0},
    ]
    arcs = []

    with pytest.raises(InvalidProblemError) as exc_info:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

    assert "duplicate" in str(exc_info.value).lower()
    assert "dup" in str(exc_info.value)


def test_invalid_problem_undirected_infinite_capacity():
    """Test InvalidProblemError for undirected arc with infinite capacity."""
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": None, "cost": 1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)

    with pytest.raises(InvalidProblemError) as exc_info:
        problem.undirected_expansion()

    assert "undirected" in str(exc_info.value).lower()
    assert "finite" in str(exc_info.value).lower() or "capacity" in str(exc_info.value).lower()


def test_invalid_problem_malformed_json(tmp_path):
    """Test InvalidProblemError raised for malformed JSON."""
    problem_file = tmp_path / "bad.json"
    problem_file.write_text('{"nodes": "not a list"}')

    with pytest.raises(InvalidProblemError) as exc_info:
        load_problem(problem_file)

    assert "nodes" in str(exc_info.value).lower() or "edges" in str(exc_info.value).lower()


def test_unbounded_problem_with_diagnostics():
    """Test UnboundedProblemError includes diagnostic information."""
    # Create problem that will be unbounded (negative cost cycle with infinite capacity)
    nodes = [
        {"id": "a", "supply": 0.0},
        {"id": "b", "supply": 0.0},
    ]
    arcs = [
        {
            "tail": "a",
            "head": "b",
            "capacity": None,
            "cost": -1.0,
        },  # Infinite capacity, negative cost
        {"tail": "b", "head": "a", "capacity": None, "cost": -1.0},
    ]

    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    with pytest.raises(UnboundedProblemError) as exc_info:
        solver.solve(max_iterations=100)

    error = exc_info.value
    assert "unbounded" in str(error).lower()
    # Check that diagnostic attributes exist
    assert hasattr(error, "entering_arc")
    assert hasattr(error, "reduced_cost")


def test_infeasible_problem_error_tracks_iterations():
    """Test InfeasibleProblemError includes iteration count."""
    error = InfeasibleProblemError("Test infeasible", iterations=42)
    assert error.iterations == 42
    assert "infeasible" in str(error).lower()


def test_numerical_instability_error_tracks_condition():
    """Test NumericalInstabilityError includes condition number."""
    error = NumericalInstabilityError("Singular matrix", condition_number=1e15)
    assert error.condition_number == 1e15
    assert "singular" in str(error).lower()


def test_iteration_limit_error_tracks_state():
    """Test IterationLimitError includes solution state."""
    error = IterationLimitError(
        "Iteration limit", iterations=1000, objective=123.45, status="feasible"
    )
    assert error.iterations == 1000
    assert error.objective == 123.45
    assert error.status == "feasible"


def test_solver_configuration_error():
    """Test SolverConfigurationError for invalid configuration."""
    error = SolverConfigurationError("Invalid phase 3")
    assert "invalid" in str(error).lower() or "phase" in str(error).lower()


def test_catch_all_solver_errors():
    """Test that catching NetworkSolverError catches all solver exceptions."""
    errors = [
        InvalidProblemError("test"),
        InfeasibleProblemError("test"),
        UnboundedProblemError("test"),
        NumericalInstabilityError("test"),
        IterationLimitError("test"),
        SolverConfigurationError("test"),
    ]

    for error in errors:
        with pytest.raises(NetworkSolverError):
            raise error


def test_invalid_problem_missing_edge_fields(tmp_path):
    """Test InvalidProblemError for edge missing tail or head."""
    problem_file = tmp_path / "bad_edge.json"
    problem_file.write_text("""{
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"tail": "a"}]
    }""")

    with pytest.raises(InvalidProblemError) as exc_info:
        load_problem(problem_file)

    assert "tail" in str(exc_info.value).lower() or "head" in str(exc_info.value).lower()


def test_solver_configuration_error_invalid_phase():
    """Test SolverConfigurationError raised for invalid phase in solver."""
    nodes = [{"id": "a", "supply": 0.0}]
    arcs = []
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    solver = NetworkSimplex(problem)

    with pytest.raises(SolverConfigurationError) as exc_info:
        solver._apply_phase_costs(phase=99)

    assert "phase" in str(exc_info.value).lower()
    assert "99" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()


def test_exception_messages_are_helpful():
    """Test that exception messages provide actionable information."""
    # InvalidProblemError
    try:
        build_problem(nodes=[{"id": "a", "supply": 1.0}], arcs=[], directed=True, tolerance=1e-6)
    except InvalidProblemError as e:
        assert "unbalanced" in str(e).lower()
        assert "zero" in str(e).lower() or "balance" in str(e).lower()


def test_exceptions_work_with_try_except():
    """Test that exceptions work correctly in try-except blocks."""
    nodes = [
        {"id": "s", "supply": 10.0},
        {"id": "t", "supply": -5.0},
    ]
    arcs = []

    try:
        build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
        pytest.fail("Expected InvalidProblemError")
    except InvalidProblemError as e:
        assert "unbalanced" in str(e).lower()
    except Exception:
        pytest.fail("Wrong exception type raised")
