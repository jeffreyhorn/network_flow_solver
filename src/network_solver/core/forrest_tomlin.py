"""Forrest–Tomlin update engine for maintaining basis factorizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..basis_lu import LUFactors, build_lu, solve_lu

# Optional Numba JIT compilation for performance-critical loops
try:
    from numba import njit  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    # Provide a no-op decorator when Numba is not available
    from collections.abc import Callable
    from typing import Any, TypeVar

    _F = TypeVar("_F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """No-op decorator fallback when Numba is not available."""

        def decorator(func: _F) -> _F:
            return func

        return decorator


@dataclass
class FTUpdate:
    """Single Forrest–Tomlin column replacement descriptor."""

    pivot: int
    y: np.ndarray
    theta: float


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _apply_ft_updates_jit(
    result: np.ndarray,
    pivots: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
) -> None:
    """Apply Forrest-Tomlin updates to the solution vector (JIT-compiled).

    This function applies the accumulated FT updates to the base solution.
    Each update modifies the result vector based on the pivot element.

    Args:
        result: Solution vector to modify in-place (will be updated)
        pivots: Array of pivot indices for each update
        ys: 2D array where ys[i] is the y vector for update i
        thetas: Array of theta values for each update

    Note:
        This is the performance-critical inner loop that gets called 131k+ times
        per solve. JIT compilation can provide significant speedup here.
    """
    n_updates = len(pivots)
    for i in range(n_updates):
        pivot_idx = pivots[i]
        pivot_value = result[pivot_idx]
        correction = pivot_value / thetas[i]
        # Vectorized subtraction: result -= ys[i] * correction
        for j in range(len(result)):
            result[j] -= ys[i, j] * correction


def _apply_ft_updates_python(
    result: np.ndarray,
    updates: list[FTUpdate],
) -> None:
    """Apply Forrest-Tomlin updates to the solution vector (pure Python).

    This is the original implementation without JIT compilation.
    Used as fallback when Numba is not available or JIT is disabled.

    Args:
        result: Solution vector to modify in-place
        updates: List of FTUpdate objects to apply
    """
    for update in updates:
        pivot_value = result[update.pivot]
        correction = pivot_value / update.theta
        result -= update.y * correction


class ForrestTomlin:
    """Maintain solves for a basis matrix via Forrest–Tomlin updates."""

    def __init__(
        self,
        matrix: np.ndarray,
        tolerance: float = 1e-9,
        max_updates: int | None = 128,
        norm_growth_limit: float | None = None,
        use_jit: bool = True,
    ) -> None:
        self._validate_square(matrix)
        self.size = matrix.shape[0]
        self.tolerance = tolerance
        self.max_updates = max_updates
        self.norm_growth_limit = norm_growth_limit
        self.use_jit = use_jit and _HAS_NUMBA  # Only enable if Numba is available
        self._initial_matrix = np.array(matrix, dtype=float, copy=True)
        self._current_matrix = self._initial_matrix.copy()
        self._base_factors: LUFactors = build_lu(self._initial_matrix)
        self._updates: list[FTUpdate] = []

    @staticmethod
    def _validate_square(matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("ForrestTomlin requires a square basis matrix.")

    def solve(self, rhs: np.ndarray) -> np.ndarray | None:
        """Solve B^{-1} rhs accounting for accumulated updates."""
        vec = np.asarray(rhs, dtype=float).reshape(-1)
        if vec.shape[0] != self.size:
            raise ValueError("Right-hand side dimension does not match basis size.")
        base = solve_lu(self._base_factors, vec)
        if base is None:
            try:
                base = np.linalg.solve(self._initial_matrix, vec)
            except np.linalg.LinAlgError:
                return None
        result = np.array(base, dtype=float, copy=True)

        # Apply accumulated FT updates using JIT path if available and enabled
        if not self._updates:
            return result

        if self.use_jit:
            # JIT path: convert updates to arrays for efficient processing
            n_updates = len(self._updates)
            pivots = np.empty(n_updates, dtype=np.int64)
            thetas = np.empty(n_updates, dtype=np.float64)
            ys = np.empty((n_updates, self.size), dtype=np.float64)

            for i, update in enumerate(self._updates):
                pivots[i] = update.pivot
                thetas[i] = update.theta
                ys[i, :] = update.y

            _apply_ft_updates_jit(result, pivots, ys, thetas)
        else:
            # Python path: original implementation
            _apply_ft_updates_python(result, self._updates)

        return result

    def update(self, pivot: int, new_column: np.ndarray) -> bool:
        """Apply a column replacement and extend the update sequence."""
        if pivot < 0 or pivot >= self.size:
            raise ValueError("Pivot index outside of basis range.")
        if self.max_updates is not None and len(self._updates) >= self.max_updates:
            return False
        column = np.asarray(new_column, dtype=float).reshape(-1)
        if column.shape[0] != self.size:
            raise ValueError("Incoming column dimension does not match basis size.")
        current_column = self._current_matrix[:, pivot]
        delta = column - current_column
        if np.linalg.norm(delta, ord=np.inf) <= self.tolerance:
            self._current_matrix[:, pivot] = column
            return True
        # Compute y = B^{-1} * delta without materialising a fresh factorisation.
        y = self.solve(delta)
        if y is None:
            return False
        theta = 1.0 + y[pivot]
        if abs(theta) <= self.tolerance:
            return False
        if self.norm_growth_limit is not None:
            inf_norm = float(np.linalg.norm(y, ord=np.inf))
            if inf_norm > self.norm_growth_limit:
                return False
        self._updates.append(FTUpdate(pivot=pivot, y=y, theta=theta))
        self._current_matrix[:, pivot] = column
        return True

    def reconstruct(self) -> np.ndarray:
        """Return the current basis matrix implied by the updates."""
        return np.array(self._current_matrix, copy=True)

    def reset(self, matrix: np.ndarray) -> None:
        """Reset the engine with a fresh basis matrix."""
        self._validate_square(matrix)
        if matrix.shape != (self.size, self.size):
            raise ValueError("Reset matrix must match the original basis dimensions.")
        self._initial_matrix = np.array(matrix, dtype=float, copy=True)
        self._current_matrix = self._initial_matrix.copy()
        self._base_factors = build_lu(self._initial_matrix)
        self._updates.clear()
