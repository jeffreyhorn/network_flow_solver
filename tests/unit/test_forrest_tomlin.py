import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.core.forrest_tomlin import ForrestTomlin  # noqa: E402


def test_forrest_tomlin_solve_matches_inverse_after_update():
    # Baseline check: FT-based solves should mirror NumPy's direct inverse after updates.
    base = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    engine = ForrestTomlin(base, tolerance=1e-9, max_updates=5)

    rhs = np.array([1.0, -2.0, 3.0], dtype=float)
    baseline = np.linalg.solve(base, rhs)
    solved = engine.solve(rhs)
    assert solved is not None
    assert np.allclose(solved, baseline)

    updated_column = np.array([2.0, -1.0, 0.5], dtype=float)
    assert engine.update(1, updated_column)
    updated_matrix = base.copy()
    updated_matrix[:, 1] = updated_column

    rhs_next = np.array([-1.0, 4.0, 0.5], dtype=float)
    expected = np.linalg.solve(updated_matrix, rhs_next)
    solved_next = engine.solve(rhs_next)
    assert solved_next is not None
    assert np.allclose(solved_next, expected)

    reconstructed = engine.reconstruct()
    assert np.allclose(reconstructed, updated_matrix)


def test_forrest_tomlin_respects_update_budget():
    # Hitting the max update budget should freeze the state and reject further changes.
    base = np.eye(3)
    engine = ForrestTomlin(base, tolerance=1e-9, max_updates=1)

    first_column = np.array([2.0, 0.0, 0.0], dtype=float)
    assert engine.update(0, first_column)
    snapshot = engine.reconstruct()

    second_column = np.array([3.0, 0.0, 0.0], dtype=float)
    assert engine.update(0, second_column) is False
    assert np.allclose(engine.reconstruct(), snapshot)


def test_forrest_tomlin_validates_square_matrices():
    matrix = np.ones((2, 3))
    with pytest.raises(ValueError, match="square basis matrix"):
        ForrestTomlin(matrix)


def test_reset_requires_matching_shape():
    base = np.eye(2)
    engine = ForrestTomlin(base)
    with pytest.raises(ValueError, match="match the original basis dimensions"):
        engine.reset(np.eye(3))


def test_solve_falls_back_to_numpy_when_lu_missing(monkeypatch):
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    engine = ForrestTomlin(base)
    monkeypatch.setattr(
        "network_solver.core.forrest_tomlin.solve_lu", lambda _f, _rhs: None
    )
    rhs = np.array([1.0, 0.0])
    expected = np.linalg.solve(base, rhs)
    assert np.allclose(engine.solve(rhs), expected)


def test_solve_returns_none_for_singular_matrix(monkeypatch):
    base = np.array([[1.0, 2.0], [2.0, 4.0]])
    engine = ForrestTomlin(base)
    monkeypatch.setattr(
        "network_solver.core.forrest_tomlin.solve_lu", lambda _f, _rhs: None
    )
    assert engine.solve(np.array([1.0, 0.0])) is None


def test_solve_rejects_dimension_mismatch():
    engine = ForrestTomlin(np.eye(2))
    with pytest.raises(ValueError, match="dimension does not match"):
        engine.solve(np.ones(3))


def test_update_no_op_respects_tolerance(monkeypatch):
    engine = ForrestTomlin(np.eye(3), tolerance=1e-9)
    assert engine.update(0, np.array([1.0, 0.0, 0.0])) is True
    assert engine._updates == []


def test_update_aborts_when_solve_fails(monkeypatch):
    engine = ForrestTomlin(np.eye(2))
    monkeypatch.setattr(engine, "solve", lambda _delta: None)
    assert engine.update(0, np.array([2.0, 0.0])) is False


def test_update_rejects_small_theta(monkeypatch):
    engine = ForrestTomlin(np.eye(2), tolerance=1e-6)

    def fake_solve(_delta):
        return np.array([-1.0 + 5e-7, 0.0])

    monkeypatch.setattr(engine, "solve", fake_solve)
    assert engine.update(0, np.array([-1.0, 0.0])) is False


def test_update_rejects_large_norm(monkeypatch):
    engine = ForrestTomlin(np.eye(2), norm_growth_limit=10.0)
    monkeypatch.setattr(engine, "solve", lambda _delta: np.array([0.0, 20.0]))
    assert engine.update(0, np.array([0.0, 1.0])) is False


def test_update_respects_max_updates():
    engine = ForrestTomlin(np.eye(2), max_updates=1)
    assert engine.update(0, np.array([2.0, 0.0])) is True
    assert engine.update(0, np.array([1.0, 0.0])) is False


def test_update_validates_indices_and_dimensions():
    engine = ForrestTomlin(np.eye(2))
    with pytest.raises(ValueError, match="outside of basis range"):
        engine.update(-1, np.zeros(2))
    with pytest.raises(ValueError, match="outside of basis range"):
        engine.update(5, np.zeros(2))
    with pytest.raises(ValueError, match="dimension does not match"):
        engine.update(0, np.zeros(3))


def test_update_records_state_and_reconstruct_reflects_changes():
    base = np.eye(2)
    engine = ForrestTomlin(base)
    new_column = np.array([2.0, 1.0])
    assert engine.update(0, new_column) is True
    assert len(engine._updates) == 1
    assert np.allclose(engine._current_matrix[:, 0], new_column)
    reconstructed = engine.reconstruct()
    assert np.allclose(reconstructed[:, 0], new_column)


def test_reset_successfully_resets_engine():
    """Test that reset() successfully resets the engine with a new matrix."""
    base = np.eye(2)
    engine = ForrestTomlin(base)
    
    # Apply some updates
    engine.update(0, np.array([2.0, 0.0]))
    assert len(engine._updates) > 0
    
    # Reset with a new matrix of the same size
    new_matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    engine.reset(new_matrix)
    
    # Verify updates are cleared
    assert len(engine._updates) == 0
    
    # Verify the matrix was updated
    assert np.allclose(engine._initial_matrix, new_matrix)
    assert np.allclose(engine._current_matrix, new_matrix)
    
    # Verify solving works with the new matrix
    rhs = np.array([1.0, 1.0])
    result = engine.solve(rhs)
    assert result is not None
    expected = np.linalg.solve(new_matrix, rhs)
    assert np.allclose(result, expected)
