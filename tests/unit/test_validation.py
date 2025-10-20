"""Tests for numeric validation and analysis."""

import warnings

import pytest

from network_solver import build_problem
from network_solver.validation import (
    analyze_numeric_properties,
    validate_numeric_properties,
)


def test_well_conditioned_problem():
    """Test that well-conditioned problems pass validation."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert analysis.is_well_conditioned
    assert len(analysis.warnings) == 0
    assert analysis.cost_range == 1.0
    assert analysis.capacity_range == 1.0
    assert analysis.supply_range == 1.0
    assert not analysis.has_extreme_values


def test_extreme_cost_detection():
    """Test detection of extremely large costs."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1e12},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert not analysis.is_well_conditioned
    assert analysis.has_extreme_values
    assert any(w.category == 'range' and 'large cost' in w.message for w in analysis.warnings)


def test_extreme_capacity_detection():
    """Test detection of extremely large capacities."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 1e15, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert not analysis.is_well_conditioned
    assert analysis.has_extreme_values
    assert any(w.category == 'range' and 'large capacity' in w.message for w in analysis.warnings)


def test_extreme_supply_detection():
    """Test detection of extremely large supplies."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 1e13},
            {"id": "B", "supply": -1e13},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 2e13, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert not analysis.is_well_conditioned
    assert analysis.has_extreme_values
    assert any(w.category == 'range' and 'large supply' in w.message for w in analysis.warnings)


def test_wide_cost_range_detection():
    """Test detection of wide cost ranges."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1e-6},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 1e6},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert not analysis.is_well_conditioned
    assert analysis.cost_range > 1e10
    assert any(w.category == 'conditioning' and 'Cost range' in w.message for w in analysis.warnings)


def test_wide_capacity_range_detection():
    """Test detection of wide capacity ranges."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 1e-5, "cost": 5.0},
            {"tail": "B", "head": "C", "capacity": 1e8, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)
    assert not analysis.is_well_conditioned
    assert analysis.capacity_range > 1e10
    assert any(w.category == 'conditioning' and 'Capacity range' in w.message for w in analysis.warnings)


def test_recommended_tolerance_adjustment():
    """Test that recommended tolerance adjusts based on problem conditioning."""
    # Well-conditioned problem
    good_problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    good_analysis = analyze_numeric_properties(good_problem)
    assert good_analysis.recommended_tolerance == 1e-6

    # Ill-conditioned problem
    bad_problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 1e-3, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 1e6, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    bad_analysis = analyze_numeric_properties(bad_problem)
    assert bad_analysis.recommended_tolerance > good_analysis.recommended_tolerance


def test_validate_numeric_properties_with_warnings():
    """Test that validation emits warnings for problematic problems."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 1e11},
            {"id": "B", "supply": -1e11},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 2e11, "cost": 1e11},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Should emit warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_numeric_properties(problem, strict=False, warn=True)
        assert len(w) > 0
        assert any("numeric issues" in str(warning.message).lower() for warning in w)


def test_validate_numeric_properties_strict_mode():
    """Test that strict mode raises exception for high-severity issues."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 1e-8, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 1e10, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Should raise exception in strict mode
    with pytest.raises(ValueError) as exc_info:
        validate_numeric_properties(problem, strict=True, warn=False)
    assert "high-severity" in str(exc_info.value).lower()


def test_validate_numeric_properties_no_warnings_suppressed():
    """Test that validation can suppress warnings."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 1e11},
            {"id": "B", "supply": -1e11},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 2e11, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Should not emit warnings when warn=False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_numeric_properties(problem, strict=False, warn=False)
        # No warnings should be emitted
        assert len(w) == 0


def test_zero_cost_handling():
    """Test handling of zero costs in range calculation."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 0.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 0.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Should not crash on all-zero costs
    analysis = analyze_numeric_properties(problem)
    assert analysis.cost_range == 1.0  # Default when all costs are zero


def test_infinite_capacity_handling():
    """Test handling of infinite capacities in range calculation."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ],
        arcs=[
            {"tail": "A", "head": "B", "capacity": None, "cost": 5.0},
        ],
        directed=True,
        tolerance=1e-6,
    )

    # Should not crash on infinite capacities
    analysis = analyze_numeric_properties(problem)
    assert analysis.capacity_range == 1.0  # Default when no finite capacities


def test_multiple_warning_severities():
    """Test that warnings are properly categorized by severity."""
    problem = build_problem(
        nodes=[
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": 0.0},
            {"id": "D", "supply": -100.0},
        ],
        arcs=[
            # Wide range (high severity)
            {"tail": "A", "head": "B", "capacity": 1e-8, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 1e10, "cost": 1.0},
            # Extreme value (medium severity)
            {"tail": "C", "head": "D", "capacity": 200.0, "cost": 5e11},
        ],
        directed=True,
        tolerance=1e-6,
    )

    analysis = analyze_numeric_properties(problem)

    # Should have multiple warnings with different severities
    severities = [w.severity for w in analysis.warnings]
    assert 'high' in severities
    assert 'medium' in severities
