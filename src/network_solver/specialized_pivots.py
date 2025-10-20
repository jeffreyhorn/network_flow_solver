"""Specialized pivot strategies for structured network problems.

This module implements optimized pivot selection and update rules for
special network structures like transportation and assignment problems.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simplex import NetworkSimplex


class TransportationPivotStrategy:
    """Specialized pivot strategy for transportation problems.

    Transportation problems have special structure that allows for
    more efficient pivot operations:
    - Tree basis has exactly m+n-1 arcs (m sources, n sinks)
    - Cycles are simple (only 4 nodes typically)
    - Can use row-column indexing for faster lookups
    """

    def __init__(self, solver: NetworkSimplex, num_sources: int, num_sinks: int):
        """Initialize transportation pivot strategy.

        Args:
            solver: The network simplex solver
            num_sources: Number of source nodes
            num_sinks: Number of sink nodes
        """
        self.solver = solver
        self.num_sources = num_sources
        self.num_sinks = num_sinks

        # Transportation problems can use a more efficient pricing method
        # that exploits the bipartite structure
        self._init_row_column_structure()

    def _init_row_column_structure(self) -> None:
        """Initialize row-column index structure for transportation problem."""
        # Map source nodes to row indices
        self.row_indices: dict[int, int] = {}
        # Map sink nodes to column indices
        self.col_indices: dict[int, int] = {}

        row_idx = 0
        col_idx = 0

        for node_idx, supply in enumerate(self.solver.node_supply):
            if node_idx == self.solver.root:
                continue
            if supply > self.solver.tolerance:  # Source
                self.row_indices[node_idx] = row_idx
                row_idx += 1
            elif supply < -self.solver.tolerance:  # Sink
                self.col_indices[node_idx] = col_idx
                col_idx += 1

    def find_entering_arc_row_scan(self) -> tuple[int, int] | None:
        """Find entering arc using row-scan method for transportation.

        Row-scan pricing is often faster for transportation problems than
        full Devex pricing because it exploits the bipartite structure.

        Returns:
            Tuple of (arc_index, direction) or None if optimal
        """
        best_arc = None
        best_rc = 0.0

        # Scan arcs row by row (source by source)
        for idx, arc in enumerate(self.solver.arcs):
            if arc.in_tree or arc.artificial:
                continue

            # Compute reduced cost
            rc = (
                arc.cost
                + self.solver.basis.potential[arc.tail]
                - self.solver.basis.potential[arc.head]
            )

            # Check forward direction
            forward_res = arc.forward_residual()
            if forward_res > self.solver.tolerance and rc < -self.solver.tolerance:
                if rc < best_rc:
                    best_rc = rc
                    best_arc = (idx, 1)

            # Check backward direction
            backward_res = arc.backward_residual()
            if backward_res > self.solver.tolerance and rc > self.solver.tolerance:
                if -rc < best_rc:
                    best_rc = -rc
                    best_arc = (idx, -1)

        return best_arc

    def compute_cycle_for_transportation(
        self, entering_idx: int, entering_direction: int
    ) -> list[tuple[int, int]]:
        """Compute cycle for transportation problem pivot.

        In transportation problems, cycles have a special structure:
        they alternate between rows (sources) and columns (sinks).
        This can be exploited for faster cycle detection.

        Args:
            entering_idx: Index of entering arc
            entering_direction: Direction (1=forward, -1=backward)

        Returns:
            List of (arc_index, sign) tuples forming the cycle
        """
        # For now, fall back to general cycle detection
        # Future optimization: use row-column structure for O(1) cycle finding
        entering = self.solver.arcs[entering_idx]
        tail = entering.tail if entering_direction == 1 else entering.head
        head = entering.head if entering_direction == 1 else entering.tail

        cycle = self.solver.basis.collect_cycle(self.solver.tree_adj, self.solver.arcs, tail, head)
        cycle.append((entering_idx, entering_direction))

        return cycle


class AssignmentPivotStrategy(TransportationPivotStrategy):
    """Specialized pivot strategy for assignment problems.

    Assignment problems are a special case of transportation where:
    - All supplies and demands are 1
    - Number of sources equals number of sinks
    - Solution is a perfect matching

    This allows for even more specialized optimizations.
    """

    def __init__(self, solver: NetworkSimplex, num_workers: int):
        """Initialize assignment pivot strategy.

        Args:
            solver: The network simplex solver
            num_workers: Number of workers (= number of jobs)
        """
        super().__init__(solver, num_workers, num_workers)
        self.size = num_workers

    def find_entering_arc_min_cost(self) -> tuple[int, int] | None:
        """Find entering arc using minimum-cost selection for assignment.

        For assignment problems, we can often use a simpler pricing rule
        that just picks the most negative reduced cost arc.

        Returns:
            Tuple of (arc_index, direction) or None if optimal
        """
        best_arc = None
        best_rc = 0.0

        for idx, arc in enumerate(self.solver.arcs):
            if arc.in_tree or arc.artificial:
                continue

            rc = (
                arc.cost
                + self.solver.basis.potential[arc.tail]
                - self.solver.basis.potential[arc.head]
            )

            # For assignment, we typically only care about forward direction
            # since all lower bounds are 0 and all capacities are 1
            forward_res = arc.forward_residual()
            if forward_res > self.solver.tolerance and rc < best_rc - self.solver.tolerance:
                best_rc = rc
                best_arc = (idx, 1)

        return best_arc


class BipartiteMatchingPivotStrategy:
    """Specialized pivot strategy for bipartite matching problems.

    Bipartite matching problems can use augmenting path methods
    for faster convergence than general simplex.
    """

    def __init__(self, solver: NetworkSimplex, left_partition: set[int], right_partition: set[int]):
        """Initialize bipartite matching pivot strategy.

        Args:
            solver: The network simplex solver
            left_partition: Node indices in left partition
            right_partition: Node indices in right partition
        """
        self.solver = solver
        self.left_partition = left_partition
        self.right_partition = right_partition

    def find_augmenting_path(self) -> tuple[int, int] | None:
        """Find an augmenting path for bipartite matching.

        Uses BFS/DFS to find augmenting paths, which is often faster
        than pivot selection for matching problems.

        Returns:
            Tuple of (arc_index, direction) representing the path start
        """
        # Find unmatched left nodes
        unmatched_left = set()
        for node_idx in self.left_partition:
            # Check if node has unit supply and no outgoing flow
            if abs(self.solver.node_supply[node_idx] - 1.0) <= self.solver.tolerance:
                has_flow = any(
                    arc.in_tree and arc.flow > self.solver.tolerance and arc.tail == node_idx
                    for arc in self.solver.arcs
                )
                if not has_flow:
                    unmatched_left.add(node_idx)

        # For simplicity, return first eligible arc from unmatched node
        # Full augmenting path algorithm would be more complex
        for node_idx in unmatched_left:
            for idx, arc in enumerate(self.solver.arcs):
                if arc.tail == node_idx and not arc.in_tree:
                    forward_res = arc.forward_residual()
                    if forward_res > self.solver.tolerance:
                        return (idx, 1)

        return None


def select_pivot_strategy(solver: NetworkSimplex, network_type: str) -> any:
    """Select appropriate pivot strategy based on network type.

    Args:
        solver: The network simplex solver
        network_type: Type of network (from NetworkType enum)

    Returns:
        Specialized pivot strategy instance or None for general
    """
    from .specializations import NetworkType

    # Import here to avoid circular dependency
    if network_type == NetworkType.TRANSPORTATION.value:
        # Count sources and sinks
        num_sources = sum(1 for supply in solver.node_supply if supply > solver.tolerance)
        num_sinks = sum(1 for supply in solver.node_supply if supply < -solver.tolerance)
        return TransportationPivotStrategy(solver, num_sources, num_sinks)

    elif network_type == NetworkType.ASSIGNMENT.value:
        num_workers = sum(1 for supply in solver.node_supply if supply > solver.tolerance)
        return AssignmentPivotStrategy(solver, num_workers)

    elif network_type == NetworkType.BIPARTITE_MATCHING.value:
        # Would need to pass partition information from network analysis
        # For now, return None and use general strategy
        return None

    # For other types, use general simplex
    return None
