# Phase 7: Memory Profiling Analysis

## Problem Overview

**Test Problem**: `gridgen_8_12a.min`
- Nodes: 4,097
- Arcs: 32,776
- Iterations: 8,903
- Peak Memory: **1.02 GB**
- Residual Memory: 17 MB
- **Memory efficiency issue**: 1 GB peak for a 4K node problem indicates poor memory usage

## Key Findings

### 1. Peak vs Residual Memory Gap

The **1.02 GB peak** vs **17 MB residual** reveals massive temporary allocations during solving:

```
Peak memory: 1.02 GB
Residual:    17 MB
Temporary:   ~1.00 GB (freed during solve)
```

This indicates we're creating and destroying large temporary arrays during pivots.

### 2. Memory Allocation Breakdown

**Top allocations in our code (simplex.py):**

| Line | Size | Count | What | When |
|------|------|-------|------|------|
| 1713 | 1.28 MB | 1 | `flows` dict building | End (result) |
| 1728 | 197 KB | 4,098 | `duals` dict building | End (result) |
| 1037 | 144 KB | 1 | `tree_arcs` set | Snapshot |
| 1036 | 128 KB | 1 | `arc_flows` dict | Snapshot |
| 426 | 246 KB | 4,502 | Arc object creation | Init |

**Total from simplex.py: 2.3 MB** (only 0.2% of peak!)

### 3. The Real Culprit: Basis Matrix Operations

For a 4,097 node problem:
- Basis matrix size: `(n-1) × (n-1) = 4,096 × 4,096`
- Storage: `4,096² × 8 bytes = 134 MB per matrix`
- **Each basis rebuild allocates 134 MB**
- **Each Forrest-Tomlin update may create temporary copies**

With 8,903 iterations and adaptive refactorization, we're likely doing:
- ~100-300 full basis rebuilds
- ~8,000+ Forrest-Tomlin updates
- Each operation may create 1-3 temporary arrays

**Estimated peak during rebuild:** 
- Original matrix: 134 MB
- LU factors: 134 MB × 2 = 268 MB
- Temporary arrays during factorization: 134-400 MB
- **Total per rebuild: 400-800 MB**

### 4. Secondary Issues

**Import/JIT overhead (9+ MB):**
- Python imports: 4.6 MB
- Numba JIT compilation: 5+ MB
- These are one-time costs, not the bottleneck

## Root Causes

### Primary: Dense Matrix Storage

```python
# basis.py line 254
matrix = np.zeros((expected, expected), dtype=float)  # 134 MB allocation!
```

For network simplex, the basis matrix is **99.9% sparse**:
- Only 2 nonzeros per column (tail +1, head -1)
- Storing as dense wastes >99% memory

### Secondary: Array Copies During Operations

NumPy operations often create temporary copies:
```python
# Example operations that may copy:
matrix @ vector              # May create temporary
matrix.T                     # May create copy
np.linalg.solve(A, b)       # Creates LU factors
```

### Tertiary: Python Object Overhead

Dictionary/set operations at the end (lines 1713, 1728) use Python objects which have 200+ bytes overhead per item.

## Optimization Opportunities

### 1. Sparse Basis Matrix (HIGH IMPACT - Est. 80% memory reduction)

**Current:** Dense `(n-1) × (n-1)` matrix (134 MB for n=4097)
**Proposed:** Sparse CSC/CSR format (267 KB for n=4097)

```python
# Instead of:
matrix = np.zeros((n, n), dtype=float)  # 134 MB

# Use:
from scipy.sparse import csc_matrix
data, row_ind, col_ptr = build_sparse_basis(arcs)
matrix = csc_matrix((data, row_ind, col_ptr), shape=(n, n))  # 267 KB
```

**Impact:**
- Basis storage: 134 MB → 0.27 MB (**500x reduction**)
- Also speeds up basis operations (sparse solve is faster)

### 2. In-Place Basis Updates (MEDIUM IMPACT - Est. 40% reduction)

Forrest-Tomlin updates should modify basis in-place instead of creating copies.

**Check if we're copying:**
```python
# In basis.py, look for:
new_matrix = old_matrix.copy()  # BAD: creates copy
old_matrix[i, j] = value        # GOOD: in-place
```

### 3. Reuse Temporary Arrays (MEDIUM IMPACT - Est. 20% reduction)

Pre-allocate temporary arrays and reuse them:

```python
class TreeBasis:
    def __init__(self, node_count, ...):
        # Pre-allocate temporary arrays for pivot operations
        self._temp_vector = np.zeros(node_count)
        self._temp_projection = np.zeros(node_count)
    
    def project_onto_basis(self, ...):
        # Reuse self._temp_projection instead of allocating each time
        result = self._temp_projection
        # ... compute into result
```

### 4. Optimize Result Building (LOW IMPACT - Est. 5% reduction)

Lines 1713 and 1728 build dictionaries at the end. Could use NumPy arrays:

```python
# Instead of:
flows = {}
for arc in self.arcs:
    flows[arc.key] = arc.flow  # Python dict overhead

# Consider:
flow_array = np.array([arc.flow for arc in self.arcs])  # Compact
arc_keys = [arc.key for arc in self.arcs]  # Separate index
```

But this is only 1.5 MB and happens once at the end, so **low priority**.

## Recommended Implementation Order

### Phase 7.1: Sparse Basis Matrix (HIGH PRIORITY)
- **Expected impact**: 80% memory reduction (1 GB → 200 MB)
- **Effort**: Medium (need to update basis_lu.py to use sparse)
- **Risk**: Medium (need to validate numerical stability)

### Phase 7.2: In-Place Basis Updates (MEDIUM PRIORITY)
- **Expected impact**: 40% reduction (200 MB → 120 MB)
- **Effort**: Low-Medium
- **Risk**: Low

### Phase 7.3: Reuse Temporary Arrays (MEDIUM PRIORITY)
- **Expected impact**: 20% reduction (120 MB → 100 MB)
- **Effort**: Low
- **Risk**: Very low

### Phase 7.4: Optimize Result Building (LOW PRIORITY)
- **Expected impact**: Minimal (<1%)
- **Effort**: Low
- **Risk**: Very low
- **Skip for now**: Focus on high-impact optimizations first

## Success Criteria

For the `gridgen_8_12a.min` benchmark:

- **Target Peak Memory**: < 150 MB (85% reduction from 1 GB)
- **Baseline**: 1.02 GB
- **After Phase 7.1**: ~200 MB (80% reduction)
- **After Phase 7.2**: ~120 MB (88% reduction)
- **After Phase 7.3**: ~100 MB (90% reduction)

**Correctness requirement**: All 576 tests must pass, identical objective values.

## Next Steps

1. Implement Phase 7.1 (sparse basis matrix)
2. Run memory profiling to measure improvement
3. Validate all tests pass
4. Implement Phase 7.2 (in-place updates)
5. Continue iteratively

---
*Analysis Date: 2025-10-27*
*Profiled Problem: gridgen_8_12a.min (4,097 nodes, 32,776 arcs)*
