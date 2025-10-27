"""Tree basis utilities for network simplex."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .basis_lu import LUFactors, build_lu, reconstruct_matrix, solve_lu
from .core.forrest_tomlin import ForrestTomlin
from .jit_tree_ops import build_tree_adj_csr, get_collect_cycle_function

if TYPE_CHECKING:
    from .simplex import ArcState


class TreeBasis:
    """Encapsulates parent/potential bookkeeping for the spanning-tree basis."""

    def __init__(
        self,
        node_count: int,
        root: int,
        tolerance: float,
        use_dense_inverse: bool | None = None,
        projection_cache_size: int = 100,
        use_jit: bool = True,
    ) -> None:
        # Auto-detect if not specified: use sparse if available, else dense
        if use_dense_inverse is None:
            from .basis_lu import has_sparse_lu

            use_dense_inverse = not has_sparse_lu()
        self.node_count = node_count
        self.root = root
        self.tolerance = tolerance
        self.use_dense_inverse = use_dense_inverse
        self.use_jit = use_jit
        self.parent: list[int | None] = [None] * node_count
        self.parent_arc: list[int | None] = [None] * node_count
        self.parent_dir: list[int] = [0] * node_count
        self.potential: list[float] = [0.0] * node_count
        self.depth: list[int] = [0] * node_count
        self.tree_arc_indices: list[int] = []
        self.basis_matrix: np.ndarray | None = None
        self.basis_inverse: np.ndarray | None = None

        # Projection cache (Optimized: simple dict, no LRU overhead)
        self.projection_cache_size = projection_cache_size
        self.projection_cache: dict[tuple[str, str], np.ndarray] = {}  # arc_key -> projection
        self.cache_basis_version = -1  # Track which basis version cache is valid for
        self.cache_hits = 0
        self.cache_misses = 0

        # Instrumentation for projection pattern analysis (Week 1)
        self.projection_requests: dict[tuple[str, str], int] = {}  # arc_key -> request_count
        self.basis_version = 0  # Incremented on basis changes for cache key
        self.projection_history: list[tuple[int, tuple[str, str]]] = []  # (basis_version, arc_key)
        self.lu_factors: LUFactors | None = None
        self.ft_engine: ForrestTomlin | None = None
        self.ft_update_limit = 64
        self.ft_norm_limit: float | None = 1e8
        self.row_nodes: list[int] = []
        self.node_to_row: dict[int, int] = {}
        self.arc_to_pos: dict[int, int] = {}

        # JIT optimization: cache array-based representations
        self._jit_collect_cycle = get_collect_cycle_function() if use_jit else None
        self._arc_tails_cache: np.ndarray | None = None
        self._arc_heads_cache: np.ndarray | None = None
        self._in_tree_cache: np.ndarray | None = None
        self._tree_adj_indices_cache: np.ndarray | None = None
        self._tree_adj_offsets_cache: np.ndarray | None = None
        self._cache_valid = False

    def rebuild(
        self,
        tree_adj: Sequence[Sequence[int]],
        arcs: Sequence[ArcState],
        build_numeric: bool = True,
    ) -> None:
        """Recompute parent pointers and node potentials from the current tree."""
        for idx in range(self.node_count):
            self.parent[idx] = None
            self.parent_arc[idx] = None
            self.parent_dir[idx] = 0
        self.parent[self.root] = self.root
        self.potential[self.root] = 0.0

        queue = deque([self.root])
        visited = {self.root}

        while queue:
            node = queue.popleft()
            for arc_idx in tree_adj[node]:
                arc = arcs[arc_idx]
                if not arc.in_tree:
                    continue
                neighbor = arc.head if arc.tail == node else arc.tail
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                self.parent[neighbor] = node
                self.parent_arc[neighbor] = arc_idx
                if arc.tail == node and arc.head == neighbor:
                    direction = 1
                    self.potential[neighbor] = self.potential[node] + arc.cost
                else:
                    direction = -1
                    self.potential[neighbor] = self.potential[node] - arc.cost
                self.parent_dir[neighbor] = direction
                self.depth[neighbor] = self.depth[node] + 1
                queue.append(neighbor)

        if len(visited) != self.node_count:
            raise RuntimeError("Tree reconstruction failed: disconnected spanning tree.")

        if build_numeric:
            self._build_numeric_basis(arcs)

    def _update_jit_arrays(
        self, tree_adj: Sequence[Sequence[int]], arcs: Sequence[ArcState]
    ) -> None:
        """Update NumPy array representations for JIT functions.

        Optimized to minimize overhead by converting tree_adj directly to CSR
        instead of calling build_tree_adj_csr which rebuilds from scratch.
        """
        num_arcs = len(arcs)

        # Build arc arrays once (immutable)
        if self._arc_tails_cache is None or len(self._arc_tails_cache) != num_arcs:
            self._arc_tails_cache = np.array([arc.tail for arc in arcs], dtype=np.int32)
            self._arc_heads_cache = np.array([arc.head for arc in arcs], dtype=np.int32)

        # Allocate in_tree array if needed
        if self._in_tree_cache is None or len(self._in_tree_cache) != num_arcs:
            self._in_tree_cache = np.zeros(num_arcs, dtype=np.bool_)
        else:
            self._in_tree_cache.fill(False)

        # Mark tree arcs as True using tree_adj (already filtered)
        for node_arcs in tree_adj:
            for arc_idx in node_arcs:
                self._in_tree_cache[arc_idx] = True

        # Build CSR tree adjacency directly from tree_adj (much faster!)
        total_entries = sum(len(node_arcs) for node_arcs in tree_adj)

        # Allocate or reuse arrays
        if (
            self._tree_adj_indices_cache is None
            or len(self._tree_adj_indices_cache) != total_entries
        ):
            self._tree_adj_indices_cache = np.empty(total_entries, dtype=np.int32)

        if (
            self._tree_adj_offsets_cache is None
            or len(self._tree_adj_offsets_cache) != self.node_count + 1
        ):
            self._tree_adj_offsets_cache = np.empty(self.node_count + 1, dtype=np.int32)

        # Fill CSR structure directly from tree_adj
        offset = 0
        for node in range(self.node_count):
            self._tree_adj_offsets_cache[node] = offset
            node_arcs = tree_adj[node]
            for i, arc_idx in enumerate(node_arcs):
                self._tree_adj_indices_cache[offset + i] = arc_idx
            offset += len(node_arcs)
        self._tree_adj_offsets_cache[self.node_count] = offset

    def collect_cycle(
        self, tree_adj: Sequence[Sequence[int]], arcs: Sequence[ArcState], tail: int, head: int
    ) -> list[tuple[int, int]]:
        """Return tree arcs forming the unique path between tail and head.

        Uses JIT-compiled implementation if use_jit=True, otherwise falls back
        to pure Python implementation.
        """
        if tail == head:
            return []

        # Use JIT version if enabled
        if self._jit_collect_cycle is not None:
            # Update array representations
            self._update_jit_arrays(tree_adj, arcs)

            # Call JIT function
            return self._jit_collect_cycle(
                self._arc_tails_cache,
                self._arc_heads_cache,
                self._in_tree_cache,
                self._tree_adj_indices_cache,
                self._tree_adj_offsets_cache,
                tail,
                head,
                self.node_count,
            )

        # Fallback: Original Python implementation
        prev: dict[int, tuple[int, int]] = {head: (head, -1)}
        queue = deque([head])

        found = False
        while queue:
            node = queue.popleft()
            if node == tail:
                found = True
                break
            for arc_idx in tree_adj[node]:
                arc = arcs[arc_idx]
                if not arc.in_tree:
                    continue
                neighbor = arc.head if arc.tail == node else arc.tail
                if neighbor in prev:
                    continue
                prev[neighbor] = (node, arc_idx)
                queue.append(neighbor)

        if not found:
            raise RuntimeError("Failed to locate cycle path in spanning tree.")

        steps: list[tuple[int, int, int]] = []
        node = tail
        while node != head:
            parent, arc_idx = prev[node]
            steps.append((parent, node, arc_idx))
            node = parent

        path: list[tuple[int, int]] = []
        for parent, child, arc_idx in reversed(steps):
            arc = arcs[arc_idx]
            sign = 1 if arc.tail == parent and arc.head == child else -1
            path.append((arc_idx, sign))
        return path

    def _build_numeric_basis(self, arcs: Sequence[ArcState]) -> None:
        # Build the reduced incidence matrix for the current spanning tree.
        tree_arcs = [idx for idx, arc in enumerate(arcs) if arc.in_tree]
        self.tree_arc_indices = tree_arcs
        expected = self.node_count - 1
        if len(tree_arcs) != expected:
            self.basis_matrix = None
            self.basis_inverse = None
            self.lu_factors = None
            self.ft_engine = None
            return

        matrix = np.zeros((expected, expected), dtype=float)
        row_nodes = [idx for idx in range(self.node_count) if idx != self.root]
        node_to_row = {node: row for row, node in enumerate(row_nodes)}
        self.row_nodes = row_nodes
        self.node_to_row = node_to_row
        self.arc_to_pos = {}

        for col, arc_idx in enumerate(tree_arcs):
            arc = arcs[arc_idx]
            self.arc_to_pos[arc_idx] = col
            if arc.tail != self.root:
                matrix[node_to_row[arc.tail], col] = 1.0
            if arc.head != self.root:
                matrix[node_to_row[arc.head], col] = -1.0

        self.basis_matrix = matrix

        # Only compute dense inverse if explicitly requested (expensive O(n³) operation)
        if self.use_dense_inverse:
            try:
                self.basis_inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                self.basis_inverse = None
        else:
            self.basis_inverse = None

        self.ft_engine = ForrestTomlin(
            matrix,
            tolerance=self.tolerance,
            max_updates=self.ft_update_limit,
            norm_growth_limit=self.ft_norm_limit,
            use_jit=self.use_jit,
        )
        self.lu_factors = build_lu(matrix)

    def _column_vector(self, arc: ArcState) -> np.ndarray:
        vec = np.zeros((self.node_count - 1,), dtype=float)
        if arc.tail != self.root:
            vec[self.node_to_row[arc.tail]] = 1.0
        if arc.head != self.root:
            vec[self.node_to_row[arc.head]] = -1.0
        return vec

    def project_column(self, arc: ArcState) -> np.ndarray | None:
        # Instrumentation: Track projection requests
        arc_key = arc.key
        self.projection_requests[arc_key] = self.projection_requests.get(arc_key, 0) + 1
        self.projection_history.append((self.basis_version, arc_key))

        # Optimized cache: simple dict lookup with arc key
        if self.projection_cache_size > 0:
            # Invalidate cache if basis has changed
            if self.cache_basis_version != self.basis_version:
                self.projection_cache.clear()
                self.cache_basis_version = self.basis_version

            # Check cache using arc_key (tuple is hashable and correct)
            if arc_key in self.projection_cache:
                self.cache_hits += 1
                # Return copy to prevent cache corruption if caller modifies array
                return self.projection_cache[arc_key].copy()

        # Cache miss: compute projection
        self.cache_misses += 1
        column = self._column_vector(arc)
        result = None

        if self.ft_engine is not None:
            # Prefer Forrest–Tomlin solves so Devex stays in sync with incremental updates.
            solved = self.ft_engine.solve(column)
            if solved is not None:
                result = solved
        if result is None and self.basis_inverse is not None:
            result = self.basis_inverse @ column
        if result is None and self.lu_factors is not None:
            solved = solve_lu(self.lu_factors, column)
            if solved is not None:
                result = solved

        # Store in cache if computation succeeded
        if result is not None and self.projection_cache_size > 0:
            # Simple cache: just store, no LRU eviction
            # Cache is cleared on basis change, so size naturally bounded
            self.projection_cache[arc_key] = result

        return result

    def estimate_condition_number(self) -> float | None:
        """Estimate the condition number of the basis matrix.

        Uses the 1-norm condition number estimate: cond(A) ≈ ||A||_1 * ||A^-1||_1
        This is a fast approximation that's suitable for monitoring numerical stability.

        Returns:
            Estimated condition number, or None if basis matrix not available.
            Values > 1e12 typically indicate ill-conditioning.

        Note:
            This is an approximation using matrix norms. For exact condition number,
            use np.linalg.cond, but that's much more expensive (requires SVD).

            When dense inverse is not available, estimates A^-1 norm by solving
            with several random vectors and taking the maximum ratio.
        """
        if self.basis_matrix is None:
            return None

        # Compute 1-norm of basis matrix
        norm_a = np.linalg.norm(self.basis_matrix, ord=1)

        # If we have dense inverse, use it directly
        if self.basis_inverse is not None:
            norm_ainv = np.linalg.norm(self.basis_inverse, ord=1)
            return float(norm_a * norm_ainv)

        # Otherwise, estimate inverse norm using sparse solves
        # This is cheaper than computing the full dense inverse
        if self.lu_factors is None and self.ft_engine is None:
            return None

        # Estimate ||A^-1||_1 by solving A*x = e_i for canonical basis vectors
        n = self.basis_matrix.shape[0]
        max_col_sum = 0.0

        # Sample a few columns to estimate (not all n, for performance)
        num_samples = min(10, n)
        for i in range(0, n, max(1, n // num_samples)):
            e_i = np.zeros(n)
            e_i[i] = 1.0

            # Solve using available method
            x = None
            if self.ft_engine is not None:
                x = self.ft_engine.solve(e_i)
            if x is None and self.lu_factors is not None:
                x = solve_lu(self.lu_factors, e_i)

            if x is not None:
                col_sum = np.sum(np.abs(x))
                max_col_sum = max(max_col_sum, col_sum)

        if max_col_sum == 0.0:
            return None

        # This is an underestimate of the true 1-norm, but sufficient for monitoring
        return float(norm_a * max_col_sum)

    def replace_arc(
        self, leaving_idx: int, entering_idx: int, arcs: Sequence[ArcState], tol: float
    ) -> bool:
        if self.basis_matrix is None or self.arc_to_pos is None:
            return False

        pos = self.arc_to_pos.get(leaving_idx)
        if pos is None:
            return False

        # Instrumentation: Increment basis version on any arc replacement
        # (will be used as cache key to track which basis a projection belongs to)
        self.basis_version += 1

        new_col = self._column_vector(arcs[entering_idx])
        if self.ft_engine is not None:
            try:
                # Attempt a rank-one FT update before falling back to rebuild paths.
                updated = self.ft_engine.update(pos, new_col)
            except ValueError:
                updated = False
            if updated:
                self.basis_matrix[:, pos] = new_col
                self.arc_to_pos.pop(leaving_idx, None)
                self.arc_to_pos[entering_idx] = pos
                self.tree_arc_indices[pos] = entering_idx
                # Clear dense inverse if in dense mode (out of sync after FT update)
                if self.use_dense_inverse:
                    self.basis_inverse = None
                self.lu_factors = None
                return True
            # If FT failed, fall through to other methods (Sherman-Morrison or LU)

        old_col = self.basis_matrix[:, pos]
        u = new_col - old_col
        if np.allclose(u, 0.0, atol=tol):
            self.basis_matrix[:, pos] = new_col
            self.arc_to_pos.pop(leaving_idx, None)
            self.arc_to_pos[entering_idx] = pos
            self.tree_arc_indices[pos] = entering_idx
            return True

        if self.basis_inverse is not None:
            b_inv = self.basis_inverse
            row = b_inv[pos, :]
            denom = 1.0 + row @ u
            if abs(denom) <= tol:
                return False
            b_u = b_inv @ u
            self.basis_inverse = b_inv - np.outer(b_u, row) / denom
            self.basis_matrix[:, pos] = new_col
            self.arc_to_pos.pop(leaving_idx, None)
            self.arc_to_pos[entering_idx] = pos
            self.tree_arc_indices[pos] = entering_idx
            if self.lu_factors is not None:
                reconstructed = reconstruct_matrix(self.lu_factors)
                reconstructed[:, pos] = new_col
                self.lu_factors = build_lu(reconstructed)
            return True
        if self.lu_factors is not None:
            reconstructed = reconstruct_matrix(self.lu_factors)
            reconstructed[:, pos] = new_col
            factors = build_lu(reconstructed)
            if factors.lu is None:
                return False
            self.lu_factors = factors
            self.basis_matrix = reconstructed
            self.arc_to_pos.pop(leaving_idx, None)
            self.arc_to_pos[entering_idx] = pos
            self.tree_arc_indices[pos] = entering_idx
            return True
        return False
