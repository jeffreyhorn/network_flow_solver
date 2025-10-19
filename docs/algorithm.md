# Network Simplex Algorithm

This document explains the network simplex algorithm implementation used in this library.

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Algorithm Structure](#algorithm-structure)
- [Data Structures](#data-structures)
- [Phase 1: Finding a Feasible Solution](#phase-1-finding-a-feasible-solution)
- [Phase 2: Optimizing the Solution](#phase-2-optimizing-the-solution)
- [Pricing Strategies](#pricing-strategies)
- [Basis Management](#basis-management)
- [Complexity Analysis](#complexity-analysis)

## Overview

The **network simplex algorithm** is a specialized version of the simplex method designed for minimum-cost flow problems. It exploits the network structure to achieve significant computational advantages over the general linear programming simplex method.

### Key Advantages

- **Efficiency**: Orders of magnitude faster than general simplex for network problems
- **Integrality**: If all supplies, demands, and capacities are integers, the optimal solution will have integer flows
- **Scalability**: Can handle large-scale networks with thousands of nodes and arcs
- **Sparsity**: Works directly with the network structure, avoiding dense matrix operations

## Problem Formulation

### Minimum-Cost Flow Problem

Given a directed graph G = (N, A) where:
- N is the set of nodes
- A is the set of arcs

**Minimize:**
```
∑(i,j)∈A c_ij · x_ij
```

**Subject to:**
```
∑(i,j)∈A x_ij - ∑(j,i)∈A x_ji = b_i    for all i ∈ N  (flow conservation)
l_ij ≤ x_ij ≤ u_ij                      for all (i,j) ∈ A  (capacity bounds)
```

Where:
- `x_ij` = flow on arc (i,j)
- `c_ij` = cost per unit of flow on arc (i,j)
- `b_i` = supply (if positive) or demand (if negative) at node i
- `l_ij` = lower bound on flow for arc (i,j)
- `u_ij` = upper bound (capacity) on flow for arc (i,j)

### Balance Constraint

For a feasible problem:
```
∑(i∈N) b_i = 0
```

The total supply must equal the total demand.

### Undirected Graphs

While the algorithm operates on directed networks, the solver supports undirected graphs through automatic transformation during preprocessing:

- **Input:** Undirected edge `{i, j}` with capacity `C` and cost `c`
- **Transformation:** Becomes directed arc `(i, j)` with:
  - Lower bound: `l_ij = -C`
  - Upper bound: `u_ij = C`
  - Cost: `c_ij = c`
- **Interpretation:** Flow `x_ij > 0` means flow goes `i → j`, while `x_ij < 0` means flow goes `j → i`

This allows the network simplex algorithm to handle bidirectional edges while maintaining the standard directed formulation. See [API Reference - Undirected Graphs](api.md#working-with-undirected-graphs) for details.

## Algorithm Structure

The network simplex algorithm maintains a **spanning tree solution** (a basis) and iteratively improves it through pivots.

```
┌─────────────────────────────────────┐
│  Initialize with feasible tree     │
│  (using artificial arcs)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 1: Minimize artificial flow  │
│  - Use modified costs               │
│  - Drive artificial arcs to zero    │
└──────────────┬──────────────────────┘
               │
               ▼
         ┌─────────┐
         │ Feasible?│
         └────┬────┘
              │ Yes
              ▼
┌─────────────────────────────────────┐
│  PHASE 2: Minimize actual cost      │
│  - Restore original costs           │
│  - Continue pivoting to optimality  │
└──────────────┬──────────────────────┘
               │
               ▼
         ┌─────────┐
         │ Optimal! │
         └──────────┘
```

## Data Structures

### Spanning Tree Basis

The algorithm maintains a **spanning tree** of the network containing exactly |N| - 1 arcs, where |N| is the number of nodes.

```
Example Network:          Spanning Tree Basis:

  s ──→ a ──→ d             s ──→ a ──→ d
  │     │                   │
  └──→ b ──→ t             b ───────→ t
```

**Properties:**
- Uniquely determines flows on all tree arcs
- Defines dual values (node potentials) uniquely
- Forms the current basic feasible solution

### Node Potentials (Dual Variables)

Each node i has a potential π_i that satisfies:
```
c_ij = π_i - π_j    for all tree arcs (i,j)
```

**Interpretation:** Shadow prices representing the marginal cost of supply/demand changes.

### Reduced Costs

For non-tree arcs (i,j):
```
r_ij = c_ij - (π_i - π_j)
```

**Optimality condition:** All reduced costs must be non-negative for non-basic arcs at their lower bound, and non-positive for arcs at their upper bound.

## Phase 1: Finding a Feasible Solution

### Initial Tree Construction

1. Add an **artificial root node**
2. Connect all original nodes to the root with artificial arcs
3. Set artificial arc costs to be large (M-method)

```
        Artificial Root
          ╱  │  │  ╲
         /   │  │   \
        s    a  b    t     (original nodes)
```

**Artificial arc properties:**
- Supply nodes: arc from node to root with capacity = supply
- Demand nodes: arc from root to node with capacity = demand
- Zero-supply nodes: arc from root to node with capacity = ∞

### Phase 1 Objective

Minimize the sum of flows on artificial arcs. If this minimum is zero, we have a feasible solution for the original problem.

**Modified costs:**
```
c'_ij = c_ij - 1 - ε·j    for original arcs (lexicographic perturbation)
c'_artificial = M         for artificial arcs (M = large penalty)
```

The lexicographic perturbation prevents cycling by making all reduced costs unique.

## Phase 2: Optimizing the Solution

### Pivot Operation

Each iteration performs a **pivot** that:
1. Selects an entering arc (non-tree arc with negative reduced cost)
2. Determines the leaving arc (tree arc that reaches its bound)
3. Updates the tree structure
4. Recomputes flows and potentials

```
Before Pivot:               After Pivot:

Tree arcs: ────            Tree arcs: ────
Non-tree: ┄┄┄┄            Non-tree: ┄┄┄┄

  s ──→ a ┄┄→ d              s ┄┄→ a ──→ d
  │                          │
  └──→ b ──→ t              └──→ b ──→ t

Entering arc: (a,d)         Leaving arc: (s,a)
```

### Flow Update

When arc (k,l) enters the tree:
1. Identify the **cycle** formed by adding (k,l) to the tree
2. Compute **θ** = maximum flow that can be sent around the cycle
3. Update flows on all cycle arcs by ±θ
4. Remove the leaving arc (the one that reaches its bound)

```
Cycle: s → a → d → t → b → s
Flow change: +θ on forward arcs, -θ on backward arcs
```

## Pricing Strategies

The algorithm supports two strategies for selecting the entering arc:

### 1. Dantzig's Rule (Most Negative Reduced Cost)

Select the non-tree arc with the most negative reduced cost:
```
(k,l) = argmin{r_ij : r_ij < 0}
```

**Pros:**
- Simple to implement
- Proven convergence

**Cons:**
- May require many iterations
- Can be slow on large problems

### 2. Devex Pricing (Default)

Uses **normalized reduced costs** based on steepest edge estimates:
```
merit_ij = (r_ij)² / w_ij
```

where w_ij is a weight that approximates the squared norm of the entering column.

**Devex weight update:**
```
w_entering = ∑(arc in cycle) w_arc
```

**Pros:**
- Typically 2-5x fewer iterations than Dantzig
- Better performance on large problems

**Cons:**
- More complex implementation
- Requires weight maintenance

### Block Pricing

Both strategies use **block pricing** to reduce per-iteration overhead:
- Divide non-tree arcs into blocks
- Search one block per iteration
- Rotate through blocks cyclically

**Default block size:** `num_arcs / 8`

This amortizes the cost of basis updates across multiple pivots.

## Basis Management

### Forrest-Tomlin Update

The algorithm uses **Forrest-Tomlin (FT) updates** to maintain the LU factorization of the basis matrix without full refactorization.

```
Initial: B = LU

After pivot: B' = LU + e_l(a_k - B⁻¹a_k)ᵀ
```

**FT Update Limit:**
- Default: 64 updates before full refactorization
- Configurable via `SolverOptions.ft_update_limit`
- Trade-off: More updates = faster but less numerically stable

### Numerical Stability

**Techniques used:**
1. **Cost perturbation**: Prevents degeneracy and cycling
2. **Periodic refactorization**: Maintains numerical accuracy
3. **Tolerance checks**: Flow conservation within ε = 1e-6
4. **Weight clamping**: Devex weights bounded in [1e-12, 1e12]

## Complexity Analysis

### Theoretical Complexity

**Worst case:** O(|A| · |N|² · log|N|) for integer data

**Average case:** Much better in practice, often O(|A| · |N|)

### Practical Performance

**Per iteration:**
- Pricing: O(block_size) for candidate selection
- Cycle detection: O(|N|) using parent pointers
- Flow update: O(cycle_length) ≤ O(|N|)
- Basis update: O(|N|²) with FT, O(|N|³) with refactorization

**Total iterations:**
- Typically O(|A|) to O(|A| · log|A|)
- Strongly depends on problem structure
- Devex pricing reduces iterations by 2-5x

### Comparison to General Simplex

| Aspect | Network Simplex | General Simplex |
|--------|----------------|----------------|
| Matrix storage | O(\|N\|) (tree structure) | O(\|N\|·\|A\|) (dense/sparse) |
| Pivot operation | O(\|N\|²) | O(\|A\|³) |
| Typical iterations | O(\|A\|) | O(\|A\|·\|N\|) |
| Memory | O(\|N\| + \|A\|) | O(\|N\|·\|A\|) |

**Speed improvement:** Often 10-1000x faster than general simplex on network problems.

## Implementation Details

### Cost Perturbation

```python
ε_base = 1e-10
growth = 1.00001

for i, arc in enumerate(arcs):
    perturb = ε_base * (growth ** i)
    perturbed_cost[i] = original_cost[i] + perturb
```

This ensures all reduced costs are strictly different, preventing cycling.

### Cycle Detection

```python
def find_cycle(tree, tail, head):
    """Find cycle formed by adding arc (tail, head) to tree."""
    # Find LCA (lowest common ancestor)
    depth_tail = depth[tail]
    depth_head = depth[head]
    
    # Walk up from deeper node
    while depth_tail > depth_head:
        cycle.append(arc to parent of tail)
        tail = parent[tail]
        depth_tail -= 1
    
    # Walk up together until LCA
    while tail != head:
        cycle.append(arc to parent of tail)
        cycle.append(arc to parent of head)
        tail = parent[tail]
        head = parent[head]
    
    return cycle
```

### Potential Computation

```python
def compute_potentials(root):
    """DFS from root to compute node potentials."""
    potential[root] = 0
    
    def dfs(node):
        for arc in tree_arcs[node]:
            if not visited[arc.head]:
                # c_ij = potential[i] - potential[j]
                potential[arc.head] = potential[node] - arc.cost
                visited[arc.head] = True
                dfs(arc.head)
    
    dfs(root)
```

## References

1. **Ahuja, R.K., Magnanti, T.L., and Orlin, J.B.** (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.

2. **Bazaraa, M.S., Jarvis, J.J., and Sherali, H.D.** (2010). *Linear Programming and Network Flows*. Wiley.

3. **Orlin, J.B.** (1997). "A polynomial time primal network simplex algorithm for minimum cost flows." *Mathematical Programming*, 78(2):109-129.

4. **Forrest, J.J.H. and Tomlin, J.A.** (1972). "Updated triangular factors of the basis to maintain sparsity in the product form simplex method." *Mathematical Programming*, 2:263-278.

5. **Harris, P.M.J.** (1973). "Pivot selection methods of the Devex LP code." *Mathematical Programming*, 5:1-28.
