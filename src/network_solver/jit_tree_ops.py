"""JIT-compiled tree operations for network simplex.

This module provides Numba-accelerated implementations of tree operations
that are bottlenecks in the network simplex algorithm. These functions
work with NumPy arrays instead of Python objects for maximum performance.

Key optimizations:
- collect_cycle_jit: BFS path finding with NumPy arrays (10% of runtime)
- update_tree_sets_jit: Tree adjacency list construction (20% of runtime)
- rebuild_tree_jit: Tree structure rebuilding (21% of runtime)

All functions have NumPy fallbacks if Numba is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# Try to import Numba
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Create dummy decorator for when Numba is unavailable
    from collections.abc import Callable
    from typing import Any, TypeVar

    F = TypeVar("F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """Dummy decorator when Numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            # @njit without arguments
            return args[0]  # type: ignore[no-any-return]
        # @njit(...) with arguments
        return lambda f: f


# =============================================================================
# Cycle Collection (Priority 1: 55.3s / 10% of total runtime)
# =============================================================================


@njit(cache=True)  # type: ignore[misc]
def _collect_cycle_bfs_jit(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    tree_adj_indices: NDArray[np.int32],
    tree_adj_offsets: NDArray[np.int32],
    tail_node: int,
    head_node: int,
    max_nodes: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], int]:
    """JIT-compiled BFS to find path from head to tail in spanning tree.

    Args:
        arc_tails: Tail node for each arc (length: num_arcs)
        arc_heads: Head node for each arc (length: num_arcs)
        in_tree: Boolean flag for each arc (length: num_arcs)
        tree_adj_indices: Flattened adjacency list arc indices
        tree_adj_offsets: Offset into tree_adj_indices for each node (length: num_nodes+1)
        tail_node: Target node (where we want to reach)
        head_node: Start node (where we begin search)
        max_nodes: Maximum number of nodes (for array allocation)

    Returns:
        prev_nodes: Parent node for each visited node (-1 if not visited)
        prev_arcs: Arc index from parent to node (-1 if not visited)
        path_length: Length of path found (-1 if not found)
    """
    # Initialize BFS structures
    prev_nodes = np.full(max_nodes, -1, dtype=np.int32)
    prev_arcs = np.full(max_nodes, -1, dtype=np.int32)
    visited = np.zeros(max_nodes, dtype=np.bool_)

    # BFS queue (pre-allocated, use indices to track)
    queue = np.empty(max_nodes, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Start BFS from head_node
    queue[queue_end] = head_node
    queue_end += 1
    visited[head_node] = True
    prev_nodes[head_node] = head_node  # Mark as visited

    found = False

    while queue_start < queue_end:
        node = queue[queue_start]
        queue_start += 1

        if node == tail_node:
            found = True
            break

        # Iterate over tree arcs adjacent to this node
        start_idx = tree_adj_offsets[node]
        end_idx = tree_adj_offsets[node + 1]

        for i in range(start_idx, end_idx):
            arc_idx = tree_adj_indices[i]

            # Skip non-tree arcs (shouldn't happen with proper tree_adj, but be safe)
            if not in_tree[arc_idx]:
                continue

            # Find neighbor (arc is undirected in tree)
            arc_tail = arc_tails[arc_idx]
            arc_head = arc_heads[arc_idx]

            neighbor = arc_head if arc_tail == node else arc_tail

            # Skip if already visited
            if visited[neighbor]:
                continue

            # Mark as visited and record parent
            visited[neighbor] = True
            prev_nodes[neighbor] = node
            prev_arcs[neighbor] = arc_idx
            queue[queue_end] = neighbor
            queue_end += 1

    if not found:
        return prev_nodes, prev_arcs, -1

    # Trace back path length
    path_length = 0
    node = tail_node
    while node != head_node:
        path_length += 1
        node = prev_nodes[node]

    return prev_nodes, prev_arcs, path_length


@njit(cache=True)  # type: ignore[misc]
def _reconstruct_cycle_path_jit(
    prev_nodes: NDArray[np.int32],
    prev_arcs: NDArray[np.int32],
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    tail_node: int,
    head_node: int,
    path_length: int,
) -> tuple[NDArray[np.int32], NDArray[np.int8]]:
    """JIT-compiled path reconstruction from BFS result.

    Args:
        prev_nodes: Parent node for each node
        prev_arcs: Arc from parent to node
        arc_tails: Tail node for each arc
        arc_heads: Head node for each arc
        tail_node: End of path
        head_node: Start of path
        path_length: Length of path

    Returns:
        cycle_arcs: Arc indices in cycle (length: path_length)
        cycle_signs: Direction of flow (+1 or -1) for each arc (length: path_length)
    """
    # Allocate result arrays
    cycle_arcs = np.empty(path_length, dtype=np.int32)
    cycle_signs = np.empty(path_length, dtype=np.int8)

    # Trace back from tail to head
    node = tail_node
    idx = path_length - 1

    while node != head_node:
        parent = prev_nodes[node]
        arc_idx = prev_arcs[node]

        # Determine sign based on arc direction
        # Positive if we traverse arc in tail→head direction
        sign = 1 if arc_tails[arc_idx] == parent and arc_heads[arc_idx] == node else -1

        cycle_arcs[idx] = arc_idx
        cycle_signs[idx] = sign
        idx -= 1
        node = parent

    return cycle_arcs, cycle_signs


def collect_cycle_jit(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    tree_adj_indices: NDArray[np.int32],
    tree_adj_offsets: NDArray[np.int32],
    tail_node: int,
    head_node: int,
    num_nodes: int,
) -> list[tuple[int, int]]:
    """Find cycle formed by adding arc (tail, head) to spanning tree.

    This is a JIT-accelerated version of basis.collect_cycle() that works
    with NumPy arrays instead of Python objects.

    Args:
        arc_tails: Tail node for each arc
        arc_heads: Head node for each arc
        in_tree: Boolean flag indicating if arc is in tree
        tree_adj_indices: Flattened tree adjacency list (arc indices)
        tree_adj_offsets: CSR-style offsets for tree_adj_indices
        tail_node: Tail of entering arc
        head_node: Head of entering arc
        num_nodes: Number of nodes in network

    Returns:
        List of (arc_index, sign) tuples forming the cycle
    """
    if tail_node == head_node:
        return []

    # Run BFS to find path
    prev_nodes, prev_arcs, path_length = _collect_cycle_bfs_jit(
        arc_tails,
        arc_heads,
        in_tree,
        tree_adj_indices,
        tree_adj_offsets,
        tail_node,
        head_node,
        num_nodes,
    )

    if path_length < 0:
        raise RuntimeError("Failed to locate cycle path in spanning tree.")

    # Reconstruct path
    cycle_arcs, cycle_signs = _reconstruct_cycle_path_jit(
        prev_nodes, prev_arcs, arc_tails, arc_heads, tail_node, head_node, path_length
    )

    # Convert to list of tuples for compatibility
    return [(int(arc), int(sign)) for arc, sign in zip(cycle_arcs, cycle_signs, strict=True)]


# =============================================================================
# Helper Functions for Array-based Tree Adjacency
# =============================================================================


def build_tree_adj_csr(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    num_nodes: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build CSR-format tree adjacency list.

    This converts the tree adjacency from list-of-lists format to CSR
    (Compressed Sparse Row) format, which is more efficient for JIT functions.

    Args:
        arc_tails: Tail node for each arc
        arc_heads: Head node for each arc
        in_tree: Boolean flag for each arc
        num_nodes: Number of nodes

    Returns:
        tree_adj_indices: Arc indices in flattened adjacency list
        tree_adj_offsets: Start index for each node's adjacency (length: num_nodes+1)
    """
    # Count tree arcs per node
    degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(len(arc_tails)):
        if in_tree[i]:
            degree[arc_tails[i]] += 1
            degree[arc_heads[i]] += 1

    # Build offsets (cumulative sum)
    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(degree)

    # Fill indices
    indices = np.empty(offsets[-1], dtype=np.int32)
    current = np.zeros(num_nodes, dtype=np.int32)

    for i in range(len(arc_tails)):
        if in_tree[i]:
            tail = arc_tails[i]
            head = arc_heads[i]

            # Add arc to tail's adjacency
            idx = offsets[tail] + current[tail]
            indices[idx] = i
            current[tail] += 1

            # Add arc to head's adjacency
            idx = offsets[head] + current[head]
            indices[idx] = i
            current[head] += 1

    return indices, offsets


# =============================================================================
# NumPy Fallback (when Numba unavailable)
# =============================================================================


def collect_cycle_numpy(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    tree_adj_indices: NDArray[np.int32],
    tree_adj_offsets: NDArray[np.int32],
    tail_node: int,
    head_node: int,
    num_nodes: int,
) -> list[tuple[int, int]]:
    """NumPy fallback for collect_cycle (when Numba unavailable).

    This is slower than the JIT version but functionally identical.
    """
    from collections import deque

    if tail_node == head_node:
        return []

    # BFS to find path
    prev = {head_node: (head_node, -1)}
    queue = deque([head_node])
    found = False

    while queue:
        node = queue.popleft()
        if node == tail_node:
            found = True
            break

        # Get adjacent arcs
        start = tree_adj_offsets[node]
        end = tree_adj_offsets[node + 1]

        for i in range(start, end):
            arc_idx = tree_adj_indices[i]

            if not in_tree[arc_idx]:
                continue

            # Find neighbor
            neighbor = arc_heads[arc_idx] if arc_tails[arc_idx] == node else arc_tails[arc_idx]

            if neighbor in prev:
                continue

            prev[neighbor] = (node, arc_idx)
            queue.append(neighbor)

    if not found:
        raise RuntimeError("Failed to locate cycle path in spanning tree.")

    # Reconstruct path
    path = []
    node = tail_node

    while node != head_node:
        parent, arc_idx = prev[node]

        # Determine sign
        sign = 1 if arc_tails[arc_idx] == parent and arc_heads[arc_idx] == node else -1

        path.append((arc_idx, sign))
        node = parent

    path.reverse()
    return path


# =============================================================================
# Tree Adjacency List Building (Priority 1: 123s / 19.5% of total runtime)
# =============================================================================


@njit(cache=True)  # type: ignore[misc]
def _build_tree_adj_jit(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    num_nodes: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """JIT-compiled tree adjacency list builder using CSR format.

    This replaces the Python list-of-lists with a more efficient CSR representation.

    Args:
        arc_tails: Tail node for each arc
        arc_heads: Head node for each arc
        in_tree: Boolean flag for each arc
        num_nodes: Number of nodes

    Returns:
        indices: Arc indices in flattened adjacency list
        offsets: Start index for each node's adjacency (length: num_nodes+1)
    """
    num_arcs = len(arc_tails)

    # Count degree for each node (tree arcs only)
    degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_arcs):
        if in_tree[i]:
            degree[arc_tails[i]] += 1
            degree[arc_heads[i]] += 1

    # Build offsets (cumulative sum)
    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        offsets[i + 1] = offsets[i] + degree[i]

    # Allocate indices array
    total_entries = offsets[num_nodes]
    indices = np.empty(total_entries, dtype=np.int32)

    # Fill indices (use current position tracker)
    current = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_arcs):
        if in_tree[i]:
            tail = arc_tails[i]
            head = arc_heads[i]

            # Add to tail's adjacency
            idx = offsets[tail] + current[tail]
            indices[idx] = i
            current[tail] += 1

            # Add to head's adjacency
            idx = offsets[head] + current[head]
            indices[idx] = i
            current[head] += 1

    return indices, offsets


def build_tree_adj_jit(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    num_nodes: int,
) -> list[list[int]]:
    """Build tree adjacency list with JIT optimization.

    Returns list-of-lists for compatibility with existing code.
    """
    indices, offsets = _build_tree_adj_jit(arc_tails, arc_heads, in_tree, num_nodes)

    # Convert CSR back to list-of-lists for compatibility
    # Optimized: create Python list directly without tolist() overhead
    tree_adj: list[list[int]] = []
    for node in range(num_nodes):
        start = int(offsets[node])
        end = int(offsets[node + 1])
        # Create Python list directly from indices
        node_arcs: list[int] = []
        for i in range(start, end):
            node_arcs.append(int(indices[i]))
        tree_adj.append(node_arcs)

    return tree_adj


def build_tree_adj_numpy(
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
    in_tree: NDArray[np.bool_],
    num_nodes: int,
) -> list[list[int]]:
    """NumPy fallback for build_tree_adj (when Numba unavailable)."""
    tree_adj: list[list[int]] = [[] for _ in range(num_nodes)]

    for i in range(len(arc_tails)):
        if in_tree[i]:
            tree_adj[arc_tails[i]].append(i)
            tree_adj[arc_heads[i]].append(i)

    return tree_adj


# =============================================================================
# Public API
# =============================================================================


def get_collect_cycle_function() -> Callable[
    [
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_],
        NDArray[np.int32],
        NDArray[np.int32],
        int,
        int,
        int,
    ],
    list[tuple[int, int]],
]:
    """Return the best available collect_cycle implementation."""
    if HAS_NUMBA:
        return collect_cycle_jit
    else:
        return collect_cycle_numpy


def get_build_tree_adj_function() -> Callable[
    [NDArray[np.int32], NDArray[np.int32], NDArray[np.bool_], int], list[list[int]]
]:
    """Return the best available build_tree_adj implementation."""
    if HAS_NUMBA:
        return build_tree_adj_jit
    else:
        return build_tree_adj_numpy
