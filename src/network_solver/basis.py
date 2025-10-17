"""Tree basis utilities for network simplex."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .basis_lu import LUFactors, build_lu, reconstruct_matrix, solve_lu
from .core.forrest_tomlin import ForrestTomlin

if TYPE_CHECKING:
    from .simplex import ArcState


class TreeBasis:
    """Encapsulates parent/potential bookkeeping for the spanning-tree basis."""

    def __init__(self, node_count: int, root: int, tolerance: float) -> None:
        self.node_count = node_count
        self.root = root
        self.tolerance = tolerance
        self.parent: list[int | None] = [None] * node_count
        self.parent_arc: list[int | None] = [None] * node_count
        self.parent_dir: list[int] = [0] * node_count
        self.potential: list[float] = [0.0] * node_count
        self.depth: list[int] = [0] * node_count
        self.tree_arc_indices: list[int] = []
        self.basis_matrix: np.ndarray | None = None
        self.basis_inverse: np.ndarray | None = None
        self.lu_factors: LUFactors | None = None
        self.ft_engine: ForrestTomlin | None = None
        self.ft_update_limit = 64
        self.ft_norm_limit: float | None = 1e8
        self.row_nodes: list[int] = []
        self.node_to_row: dict[int, int] = {}
        self.arc_to_pos: dict[int, int] = {}

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

    def collect_cycle(
        self, tree_adj: Sequence[Sequence[int]], arcs: Sequence[ArcState], tail: int, head: int
    ) -> list[tuple[int, int]]:
        """Return tree arcs forming the unique path between tail and head."""
        if tail == head:
            return []

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
        try:
            self.basis_inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.basis_inverse = None
        self.ft_engine = ForrestTomlin(
            matrix,
            tolerance=self.tolerance,
            max_updates=self.ft_update_limit,
            norm_growth_limit=self.ft_norm_limit,
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
        column = self._column_vector(arc)
        if self.ft_engine is not None:
            # Prefer Forrest–Tomlin solves so Devex stays in sync with incremental updates.
            solved = self.ft_engine.solve(column)
            if solved is not None:
                return solved
        if self.basis_inverse is not None:
            result: np.ndarray = self.basis_inverse @ column
            return result
        if self.lu_factors is not None:
            solved = solve_lu(self.lu_factors, column)
            if solved is not None:
                return solved
        return None

    def replace_arc(
        self, leaving_idx: int, entering_idx: int, arcs: Sequence[ArcState], tol: float
    ) -> bool:
        if self.basis_matrix is None or self.arc_to_pos is None:
            return False

        pos = self.arc_to_pos.get(leaving_idx)
        if pos is None:
            return False

        new_col = self._column_vector(arcs[entering_idx])
        if self.ft_engine is not None:
            ft_exception = False
            try:
                # Attempt a rank-one FT update before falling back to rebuild paths.
                updated = self.ft_engine.update(pos, new_col)
            except ValueError:
                updated = False
                ft_exception = True
            if updated:
                self.basis_matrix[:, pos] = new_col
                self.arc_to_pos.pop(leaving_idx, None)
                self.arc_to_pos[entering_idx] = pos
                self.tree_arc_indices[pos] = entering_idx
                self.basis_inverse = None
                self.lu_factors = None
                return True
            if ft_exception or (self.basis_inverse is None and self.lu_factors is None):
                return False

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
