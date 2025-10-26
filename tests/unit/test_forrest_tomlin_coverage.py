"""Additional tests to improve forrest_tomlin.py coverage to >90%."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.core.forrest_tomlin import ForrestTomlin


class TestForrestTomlinValidation:
    """Test ForrestTomlin validation edge cases."""

    def test_validate_square_with_non_square_matrix(self):
        """Test _validate_square raises error for non-square matrix (lines 60-61)."""
        # Non-square matrix should raise ValueError
        non_square = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="square basis matrix"):
            ForrestTomlin(non_square)

    def test_validate_square_with_1d_array(self):
        """Test _validate_square raises error for 1D array (lines 60-61)."""
        # 1D array should raise ValueError
        one_d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="square basis matrix"):
            ForrestTomlin(one_d)

    def test_validate_square_with_3d_array(self):
        """Test _validate_square raises error for 3D array (lines 60-61)."""
        # 3D array should raise ValueError
        three_d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        with pytest.raises(ValueError, match="square basis matrix"):
            ForrestTomlin(three_d)


class TestForrestTomlinSolveFallback:
    """Test ForrestTomlin solve fallback paths."""

    def test_solve_uses_numpy_fallback_when_lu_fails(self):
        """Test solve falls back to numpy when LU solve fails (lines 83-86)."""
        # Create a well-conditioned matrix
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        ft = ForrestTomlin(matrix)

        # Mock solve_lu to return None (simulating failure)
        with patch("network_solver.core.forrest_tomlin.solve_lu", return_value=None):
            rhs = np.array([3.0, 3.0])
            result = ft.solve(rhs)

            # Should still get a result via numpy fallback
            assert result is not None
            # Verify solution is correct: [1, 1] solves the system
            expected = np.array([1.0, 1.0])
            assert np.allclose(result, expected)

    def test_solve_returns_none_when_all_methods_fail(self):
        """Test solve returns None when both LU and numpy fail (line 86)."""
        # Create a singular matrix
        singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        ft = ForrestTomlin(singular)

        rhs = np.array([1.0, 1.0])
        result = ft.solve(rhs)

        # Should return None for singular matrix
        assert result is None


class TestForrestTomlinUpdateValidation:
    """Test ForrestTomlin update validation."""

    def test_update_with_invalid_pivot_negative(self):
        """Test update raises ValueError for negative pivot (line 148)."""
        matrix = np.eye(3)
        ft = ForrestTomlin(matrix)

        new_column = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Pivot index outside"):
            ft.update(pivot=-1, new_column=new_column)

    def test_update_with_invalid_pivot_too_large(self):
        """Test update raises ValueError for pivot >= size (line 148)."""
        matrix = np.eye(3)
        ft = ForrestTomlin(matrix)

        new_column = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Pivot index outside"):
            ft.update(pivot=3, new_column=new_column)

    def test_update_with_wrong_column_size(self):
        """Test update raises ValueError for wrong column size (line 158)."""
        matrix = np.eye(3)
        ft = ForrestTomlin(matrix)

        # Column with wrong size
        wrong_size_column = np.array([1.0, 2.0])  # Too short

        with pytest.raises(ValueError, match="Incoming column dimension does not match"):
            ft.update(pivot=0, new_column=wrong_size_column)


class TestForrestTomlinNumbaFallback:
    """Test behavior when Numba is not available."""

    def test_initialization_without_numba(self, monkeypatch):
        """Test ForrestTomlin works without Numba (lines 16-26)."""
        # Mock _HAS_NUMBA to False
        import network_solver.core.forrest_tomlin as ft_module

        original_has_numba = ft_module._HAS_NUMBA
        try:
            monkeypatch.setattr(ft_module, "_HAS_NUMBA", False)

            # Create ForrestTomlin with JIT requested but Numba unavailable
            matrix = np.eye(3)
            ft = ForrestTomlin(matrix, use_jit=True)

            # Should disable JIT when Numba not available
            assert ft.use_jit is False

            # Should still work correctly
            rhs = np.array([1.0, 2.0, 3.0])
            result = ft.solve(rhs)
            assert result is not None
            assert np.allclose(result, rhs)

        finally:
            monkeypatch.setattr(ft_module, "_HAS_NUMBA", original_has_numba)

    def test_solve_with_python_fallback_path(self):
        """Test solve uses Python path when JIT disabled (lines 145-146)."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        # Explicitly disable JIT
        ft = ForrestTomlin(matrix, use_jit=False)

        # Add an update to trigger the update application path
        new_column = np.array([1.0, 0.5])
        success = ft.update(pivot=0, new_column=new_column)
        assert success is True

        # Solve should use Python path (_apply_ft_updates_python)
        rhs = np.array([1.0, 1.0])
        result = ft.solve(rhs)

        # Should still get a valid result
        assert result is not None
        assert len(result) == 2
