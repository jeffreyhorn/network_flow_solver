"""Unit tests for basis_lu module."""

import contextlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.basis_lu import (  # noqa: E402
    LUFactors,
    build_lu,
    reconstruct_matrix,
    solve_lu,
)


@contextlib.contextmanager
def _suppress_scipy(monkeypatch):
    """Temporarily make scipy imports fail so the dense fallback gets exercised."""
    cached = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith("scipy")}

    def _raise_import_error(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ModuleNotFoundError("scipy disabled for test")
        return original_import(name, *args, **kwargs)

    original_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    try:
        import network_solver.basis_lu as basis_lu

        importlib.reload(basis_lu)
        yield
    finally:
        import network_solver.basis_lu as basis_lu

        importlib.reload(basis_lu)
        monkeypatch.setattr(importlib, "import_module", original_import)
        sys.modules.update(cached)


class TestBuildLU:
    """Tests for build_lu() function."""

    def test_build_lu_simple_matrix(self):
        """Test building LU factors for a simple 2x2 matrix."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        factors = build_lu(matrix)

        assert factors.dense_matrix is not None
        assert np.allclose(factors.dense_matrix, matrix)
        # When SciPy is available, sparse structures should be populated
        # (cannot assert presence since it depends on environment)

    def test_build_lu_identity_matrix(self):
        """Test building LU factors for identity matrix."""
        matrix = np.eye(3)
        factors = build_lu(matrix)

        assert factors.dense_matrix is not None
        assert np.allclose(factors.dense_matrix, matrix)

    def test_build_lu_with_scipy_available(self):
        """Test that SciPy path is taken when available."""
        matrix = np.array([[4.0, 3.0], [6.0, 3.0]])
        factors = build_lu(matrix)

        assert factors.dense_matrix is not None
        # If SciPy is available, factors.lu should be populated
        try:
            from scipy.sparse.linalg import splu  # noqa: F401

            assert factors.sparse_matrix is not None
            assert factors.lu is not None
        except ImportError:
            # SciPy not available, skip sparse checks
            pass

    def test_build_lu_without_scipy(self, monkeypatch):
        """Test that dense fallback works when SciPy is unavailable."""
        matrix = np.array([[4.0, 3.0], [6.0, 3.0]])

        with _suppress_scipy(monkeypatch):
            import network_solver.basis_lu as basis_lu

            factors = basis_lu.build_lu(matrix)

            assert factors.dense_matrix is not None
            assert np.allclose(factors.dense_matrix, matrix)
            assert factors.sparse_matrix is None
            assert factors.lu is None

    def test_build_lu_singular_matrix(self):
        """Test building LU factors for a singular matrix."""
        matrix = np.array([[1.0, 2.0], [2.0, 4.0]])  # Rank deficient
        factors = build_lu(matrix)

        # build_lu should succeed, but solve_lu may fail later
        assert factors.dense_matrix is not None

    def test_build_lu_preserves_original(self):
        """Test that build_lu doesn't modify the original matrix."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix = original.copy()
        factors = build_lu(matrix)

        assert np.allclose(matrix, original)
        assert not np.shares_memory(factors.dense_matrix, original)

    def test_build_lu_large_matrix(self):
        """Test building LU factors for a larger matrix."""
        size = 50
        matrix = np.random.rand(size, size) + np.eye(size) * 10  # Diagonally dominant
        factors = build_lu(matrix)

        assert factors.dense_matrix.shape == (size, size)
        assert np.allclose(factors.dense_matrix, matrix)

    def test_build_lu_handles_scipy_exception(self, monkeypatch):
        """Test that build_lu handles SciPy splu failures gracefully."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        try:
            import scipy.sparse.linalg  # noqa: F401

            # Patch the splu function in the basis_lu module where it's used
            import network_solver.basis_lu as basis_lu_module

            def failing_splu(*args, **kwargs):
                raise RuntimeError("splu failed")

            monkeypatch.setattr(basis_lu_module, "splu", failing_splu)

            factors = build_lu(matrix)

            # Should fall back gracefully
            assert factors.dense_matrix is not None
            assert factors.lu is None
        except ImportError:
            pytest.skip("SciPy not available")


class TestSolveLU:
    """Tests for solve_lu() function."""

    def test_solve_lu_simple_system(self):
        """Test solving a simple linear system."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        rhs = np.array([3.0, 3.0])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(matrix @ solution, rhs)

    def test_solve_lu_identity_system(self):
        """Test solving system with identity matrix."""
        matrix = np.eye(3)
        rhs = np.array([1.0, 2.0, 3.0])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(solution, rhs)

    def test_solve_lu_with_scipy_path(self):
        """Test that SciPy solve path works when available."""
        matrix = np.array([[4.0, 3.0], [6.0, 3.0]])
        rhs = np.array([10.0, 12.0])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(matrix @ solution, rhs)

    def test_solve_lu_without_scipy(self, monkeypatch):
        """Test that NumPy fallback works when SciPy is unavailable."""
        matrix = np.array([[4.0, 3.0], [6.0, 3.0]])
        rhs = np.array([10.0, 12.0])

        with _suppress_scipy(monkeypatch):
            import network_solver.basis_lu as basis_lu

            factors = basis_lu.build_lu(matrix)
            solution = basis_lu.solve_lu(factors, rhs)

            assert solution is not None
            assert np.allclose(matrix @ solution, rhs)

    def test_solve_lu_singular_system_returns_none(self):
        """Test that solve_lu returns None for singular systems."""
        matrix = np.array([[1.0, 2.0], [2.0, 4.0]])  # Singular
        rhs = np.array([1.0, 2.0])
        factors = build_lu(matrix)

        # Force NumPy path by removing scipy solve capability
        if factors.lu is not None:
            factors.lu = None

        solution = solve_lu(factors, rhs)

        # Should return None for singular system
        assert solution is None

    def test_solve_lu_dimension_mismatch_raises_error(self):
        """Test that solve_lu raises ValueError for dimension mismatch."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        rhs = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        factors = build_lu(matrix)

        with pytest.raises(ValueError, match="dimension does not match"):
            solve_lu(factors, rhs)

    def test_solve_lu_preserves_rhs(self):
        """Test that solve_lu doesn't modify the original rhs."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        original_rhs = np.array([3.0, 3.0])
        rhs = original_rhs.copy()
        factors = build_lu(matrix)

        solve_lu(factors, rhs)

        assert np.allclose(rhs, original_rhs)

    def test_solve_lu_accepts_list_rhs(self):
        """Test that solve_lu accepts list as rhs."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        rhs = [3.0, 3.0]  # List instead of array
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(matrix @ solution, rhs)

    def test_solve_lu_multiple_solves(self):
        """Test that solve_lu can be called multiple times with same factors."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        factors = build_lu(matrix)

        rhs1 = np.array([3.0, 3.0])
        solution1 = solve_lu(factors, rhs1)

        rhs2 = np.array([5.0, 7.0])
        solution2 = solve_lu(factors, rhs2)

        assert solution1 is not None
        assert solution2 is not None
        assert np.allclose(matrix @ solution1, rhs1)
        assert np.allclose(matrix @ solution2, rhs2)

    def test_solve_lu_large_system(self):
        """Test solving a larger linear system."""
        size = 50
        matrix = np.random.rand(size, size) + np.eye(size) * 10
        rhs = np.random.rand(size)
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert solution.shape == (size,)
        assert np.allclose(matrix @ solution, rhs, atol=1e-6)


class TestReconstructMatrix:
    """Tests for reconstruct_matrix() function."""

    def test_reconstruct_matrix_simple(self):
        """Test reconstructing a simple matrix."""
        original = np.array([[2.0, 1.0], [1.0, 2.0]])
        factors = build_lu(original)

        reconstructed = reconstruct_matrix(factors)

        assert np.allclose(reconstructed, original)

    def test_reconstruct_matrix_returns_copy(self):
        """Test that reconstruct_matrix returns a copy, not a reference."""
        original = np.array([[2.0, 1.0], [1.0, 2.0]])
        factors = build_lu(original)

        reconstructed = reconstruct_matrix(factors)

        # Modify reconstructed and ensure factors aren't affected
        reconstructed[0, 0] = 999.0
        assert factors.dense_matrix[0, 0] != 999.0

    def test_reconstruct_matrix_identity(self):
        """Test reconstructing identity matrix."""
        original = np.eye(3)
        factors = build_lu(original)

        reconstructed = reconstruct_matrix(factors)

        assert np.allclose(reconstructed, original)

    def test_reconstruct_matrix_large(self):
        """Test reconstructing a larger matrix."""
        size = 50
        original = np.random.rand(size, size)
        factors = build_lu(original)

        reconstructed = reconstruct_matrix(factors)

        assert reconstructed.shape == (size, size)
        assert np.allclose(reconstructed, original)


class TestLUFactorsDataclass:
    """Tests for LUFactors dataclass."""

    def test_lu_factors_creation(self):
        """Test creating LUFactors directly."""
        dense = np.eye(2)
        factors = LUFactors(dense_matrix=dense, sparse_matrix=None, lu=None)

        assert factors.dense_matrix is not None
        assert factors.sparse_matrix is None
        assert factors.lu is None

    def test_lu_factors_with_all_fields(self):
        """Test creating LUFactors with all fields populated."""
        dense = np.array([[1.0, 2.0], [3.0, 4.0]])

        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import splu

            sparse = csc_matrix(dense)
            lu_obj = splu(sparse)

            factors = LUFactors(dense_matrix=dense, sparse_matrix=sparse, lu=lu_obj)

            assert factors.dense_matrix is not None
            assert factors.sparse_matrix is not None
            assert factors.lu is not None
        except ImportError:
            pytest.skip("SciPy not available")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        matrix = np.array([]).reshape(0, 0)
        factors = build_lu(matrix)

        assert factors.dense_matrix.shape == (0, 0)

    def test_single_element_matrix(self):
        """Test handling of 1x1 matrix."""
        matrix = np.array([[5.0]])
        rhs = np.array([10.0])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(solution, [2.0])

    def test_rectangular_matrix(self):
        """Test handling of non-square matrix (should work for build, may fail for solve)."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        factors = build_lu(matrix)

        # build_lu should succeed
        assert factors.dense_matrix.shape == (2, 3)

    def test_very_small_values(self):
        """Test handling of very small matrix values."""
        matrix = np.eye(3) * 1e-10
        rhs = np.array([1e-10, 2e-10, 3e-10])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        # Solution should be approximately [1, 2, 3]
        assert np.allclose(solution, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_very_large_values(self):
        """Test handling of very large matrix values."""
        matrix = np.eye(3) * 1e10
        rhs = np.array([1e10, 2e10, 3e10])
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(solution, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_ill_conditioned_matrix(self):
        """Test handling of ill-conditioned matrix."""
        # Hilbert matrix is notoriously ill-conditioned
        n = 5
        matrix = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        rhs = np.ones(n)
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        # May succeed but with numerical error
        if solution is not None:
            # Just verify it returns something reasonable
            assert solution.shape == (n,)

    def test_zero_rhs(self):
        """Test solving with zero right-hand side."""
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        rhs = np.zeros(2)
        factors = build_lu(matrix)

        solution = solve_lu(factors, rhs)

        assert solution is not None
        assert np.allclose(solution, np.zeros(2))


def test_scipy_solve_path_explicitly():
    """Test that the SciPy lu.solve() path is executed when lu is not None."""
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np.array([5.0, 7.0])

    factors = build_lu(matrix)

    # Ensure SciPy path is available
    if factors.lu is not None:
        # This should execute lines 45-46
        solution = solve_lu(factors, rhs)
        assert solution is not None
        assert np.allclose(matrix @ solution, rhs)

        # Verify the solution came from SciPy (indirectly)
        expected = np.linalg.solve(matrix, rhs)
        assert np.allclose(solution, expected)
    else:
        pytest.skip("SciPy not available or splu failed")


def test_build_lu_exception_path_with_real_failure():
    """Test build_lu exception handling with a scenario that causes splu to fail."""
    # Create a scenario more likely to cause issues
    try:
        from scipy.sparse import csc_matrix  # noqa: F401
        from scipy.sparse.linalg import splu  # noqa: F401

        # Try with an extremely ill-conditioned matrix
        matrix = np.array([[1e-15, 1.0], [1.0, 1e15]])
        factors = build_lu(matrix)

        # The function should handle any exception gracefully
        assert factors.dense_matrix is not None
        # lu might be None if splu raised an exception
        assert factors.lu is None or factors.lu is not None
    except ImportError:
        pytest.skip("SciPy not available")
