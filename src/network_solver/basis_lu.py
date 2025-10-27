"""LU helper for the tree basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # SciPy provides high-quality sparse factorizations when available.
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
except ModuleNotFoundError:  # pragma: no cover - exercised when SciPy missing.
    csc_matrix = None
    splu = None


@dataclass
class LUFactors:
    dense_matrix: np.ndarray
    sparse_matrix: csc_matrix | None
    lu: splu | None


def build_lu(matrix: np.ndarray) -> LUFactors:
    """Construct sparse LU factors for the given reduced incidence matrix.

    Args:
        matrix: Dense numpy array representing the basis matrix

    Returns:
        LUFactors object containing both dense and sparse representations.
        The sparse representation is used internally for memory-efficient factorization.
    """
    # Always make a copy of the dense matrix
    dense_matrix = np.array(matrix, dtype=float, copy=True)

    # Convert to sparse for memory-efficient LU factorization
    sparse_mat = None
    lu = None
    if csc_matrix is not None and splu is not None:
        # SciPy path: convert to sparse and factorize
        # This saves memory during factorization (sparse LU uses less memory than dense)
        sparse_mat = csc_matrix(dense_matrix)
        try:
            lu = splu(sparse_mat)
        except Exception:
            lu = None

    return LUFactors(dense_matrix=dense_matrix, sparse_matrix=sparse_mat, lu=lu)


def has_sparse_lu() -> bool:
    """Check if sparse LU factorization is available (requires scipy).

    Returns:
        True if scipy.sparse.linalg.splu is available, False otherwise.
    """
    return splu is not None


def solve_lu(factors: LUFactors, rhs: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    if vec.shape[0] != factors.dense_matrix.shape[0]:
        raise ValueError("Right-hand side dimension does not match factor dimensions.")
    if factors.lu is not None:
        result: np.ndarray = factors.lu.solve(vec)
        return result
    try:
        return np.linalg.solve(factors.dense_matrix, vec)
    except np.linalg.LinAlgError:
        return None


def reconstruct_matrix(factors: LUFactors) -> np.ndarray:
    return np.array(factors.dense_matrix, copy=True)
