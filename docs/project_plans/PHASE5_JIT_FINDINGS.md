# Phase 5 JIT Compilation - Initial Findings

**Date**: 2025-10-26  
**Status**: Strategic Pivot Required  
**Problem**: GRIDGEN 8_12a (4097 nodes, 32776 arcs, 8941 iterations)

---

## Executive Summary

**Initial JIT attempt on `collect_cycle()` FAILED due to conversion overhead.**

- ❌ Original time: 560s
- ❌ With JIT: 751s (34% SLOWER)
- ❌ Net overhead: +191s despite 55s → 3s JIT speedup

**Root cause**: Data stored in Python objects (ArcState dataclass, list-of-lists tree_adj), but JIT requires NumPy arrays. Conversion overhead (106s) exceeds JIT benefit (52s savings).

**Strategic pivot**: 
1. Disable JIT for `collect_cycle()` 
2. Focus on bigger bottlenecks that don't require conversion: `rebuild()` (147s), `_update_tree_sets()` (134s)
3. Consider architectural refactor to use NumPy arrays throughout (major change)

---

## Profiling Results Comparison

### Baseline (No JIT) - 560.8s total

| Function | Time (s) | % Total | Calls | Notes |
|----------|----------|---------|-------|-------|
| `basis.rebuild()` | 117.2 | 20.9% | 8855 | Tree structure rebuild |
| `simplex._update_tree_sets()` | 110.5 | 19.7% | 8852 | Tree adjacency lists |
| `basis.collect_cycle()` | 55.3 | 9.9% | 8941 | **JIT target** |
| `numpy.ndarray.nonzero()` | 58.8 | 10.5% | 278 | Scipy overhead (native) |

**Top 3 targetable bottlenecks: 282.9s (50.4% of runtime)**

### With JIT (Initial Implementation) - 903.2s total

| Function | Time (s) | % Total | Calls | Notes |
|----------|----------|---------|-------|-------|
| `jit_tree_ops.build_tree_adj_csr()` | 299.1 | 33.1% | 8941 | ⚠️ **NEW OVERHEAD** |
| `basis.rebuild()` | 127.5 | 14.1% | 8855 | Slightly worse |
| `simplex._update_tree_sets()` | 119.3 | 13.2% | 8852 | Slightly worse |
| `basis._build_jit_arrays()` | 84.0 | 9.3% | 8941 | ⚠️ **NEW OVERHEAD** |
| `basis.collect_cycle()` | 387.9 | 43.0% | 8941 | 7x WORSE (overhead!) |
| `collect_cycle_jit()` | 2.4 | 0.3% | 8941 | ✅ Fast, but overhead dominates |

**Total JIT overhead: 384s**  
**JIT speedup: 52s**  
**Net effect: -332s (59% SLOWER)**

### With JIT (Optimized CSR Building) - 751.4s total

| Function | Time (s) | % Total | Calls | Notes |
|----------|----------|---------|-------|-------|
| `basis.rebuild()` | 146.5 | 19.5% | 8855 | 25% worse than baseline |
| `simplex._update_tree_sets()` | 133.6 | 17.8% | 8852 | 21% worse |
| `basis._update_jit_arrays()` | 106.5 | 14.2% | 8941 | ⚠️ Still huge overhead |
| `basis.collect_cycle()` | 167.4 | 22.3% | 8941 | 3x worse than baseline |
| `collect_cycle_jit()` | 2.8 | 0.4% | 8941 | ✅ Very fast |

**Total JIT overhead: ~110s**  
**JIT speedup: ~52s**  
**Net effect: -58s (10% SLOWER after optimization)**

---

## Why JIT Failed for `collect_cycle()`

### The Fundamental Problem

The solver architecture uses **Python-native data structures**:
```python
# ArcState: dataclass with tail, head, cost, flow, in_tree, etc.
arcs: list[ArcState]

# tree_adj: list of lists, each containing arc indices
tree_adj: list[list[int]]
```

But Numba JIT requires **NumPy arrays**:
```python
arc_tails: NDArray[np.int32]
arc_heads: NDArray[np.int32]
in_tree: NDArray[np.bool_]
tree_adj_indices: NDArray[np.int32]  # CSR format
tree_adj_offsets: NDArray[np.int32]
```

### Conversion Overhead Breakdown

**What `_update_jit_arrays()` does** (called 8941 times, once per pivot):

1. **Build arc arrays** (one-time, amortized):
   ```python
   arc_tails = np.array([arc.tail for arc in arcs])  # ~20ms per call initially
   arc_heads = np.array([arc.head for arc in arcs])  # cached after first call
   ```

2. **Mark in_tree flags** (every call):
   ```python
   in_tree.fill(False)  # Reset array
   for node_arcs in tree_adj:
       for arc_idx in node_arcs:
           in_tree[arc_idx] = True  # ~4ms
   ```

3. **Build CSR tree adjacency** (every call):
   ```python
   # Count entries, allocate arrays, fill CSR structure
   # Originally: 299s (using build_tree_adj_csr)
   # Optimized: ~50s (direct conversion)
   # Still expensive: ~6ms per call × 8941 calls = 50s
   ```

**Total overhead per call**: ~10-12ms  
**Total overhead**: 10ms × 8941 = 90-110s

**JIT benefit**: 55s → 3s = 52s saved

**Net result**: -58s (still losing!)

### Why the Overhead is So High

1. **Data fragmentation**: ArcState objects are not contiguous in memory
2. **Repeated iterations**: Must iterate 8941 times to extract tail/head
3. **CSR conversion**: tree_adj (list-of-lists) → CSR arrays requires counting and copying
4. **Python → NumPy boundary**: Even "fast" conversions have overhead

### Why Original Python is So Fast

The original `collect_cycle()` uses Python-native BFS:
- `deque` for queue: highly optimized C implementation
- `dict` for predecessors: O(1) lookups, no conversion needed
- Works directly with existing data structures
- No conversion overhead

**Lesson**: Sometimes Python is fast enough, especially when data is already in Python format.

---

## Strategic Options

### Option 1: Disable JIT for `collect_cycle()` ✅ **CHOSEN**

**Pros**:
- Immediate fix (back to 560s baseline)
- Preserves code clarity
- No conversion overhead

**Cons**:
- Doesn't improve performance
- Leaves 55s (10%) on the table

**Recommendation**: Do this now while exploring other options.

---

### Option 2: Focus JIT on Bigger Bottlenecks

Target `rebuild()` (147s) and `_update_tree_sets()` (134s) which together are 50% of runtime.

**`_update_tree_sets()` - 134s (24% of runtime)**:
```python
def _update_tree_sets(self) -> None:
    self.tree_adj = [[] for _ in range(self.node_count)]
    for idx, arc in enumerate(self.arcs):
        if arc.in_tree:
            self.tree_adj[arc.tail].append(idx)
            self.tree_adj[arc.head].append(idx)
```

**JIT potential**: High!
- Simple loops over arcs
- No complex data structures
- Could use NumPy to build tree_adj more efficiently
- Or pre-allocate and track sizes

**Expected speedup**: 2-3x → save 45-90s

**`rebuild()` - 147s (26% of runtime)**:
```python
def rebuild(self, tree_adj, arcs, build_numeric=True):
    # Reset parent arrays
    # BFS to rebuild tree structure
    # Compute potentials
    # Build numeric basis (if needed)
```

**JIT potential**: Medium
- More complex (BFS, potential updates)
- Works with tree_adj and arcs (same conversion problem?)
- `build_numeric` calls scipy (can't JIT that part)

**Expected speedup**: 1.5-2x → save 50-75s

**Combined potential**: 95-165s savings (17-29% overall speedup)

---

### Option 3: Architectural Refactor (Long-term)

**Refactor solver to use NumPy arrays throughout**:

```python
# Instead of:
arcs: list[ArcState]

# Use parallel arrays:
arc_tails: NDArray[np.int32]
arc_heads: NDArray[np.int32]
arc_costs: NDArray[np.float64]
arc_flows: NDArray[np.float64]
arc_in_tree: NDArray[np.bool_]
# etc.
```

**Pros**:
- Eliminates all conversion overhead
- Unlocks full JIT potential
- More cache-friendly (better memory locality)
- Easier to vectorize operations

**Cons**:
- **MAJOR REFACTOR** (100s of lines changed)
- Breaks existing API (users rely on Arc/Node objects)
- Harder to read/debug (indices instead of objects)
- Type safety issues (can't use dataclasses)

**Effort**: 2-4 weeks  
**Risk**: High (breaking changes, bugs)  
**Payoff**: 2-5x overall speedup (if done well)

**Recommendation**: Consider for Phase 6+, not Phase 5.

---

### Option 4: Hybrid Approach

Keep current API, but convert to NumPy internally:

```python
class SimplexSolver:
    def __init__(self, problem):
        # Store user-facing Arc objects
        self.arcs_objects = [Arc(...) for ...]
        
        # Build NumPy arrays once
        self._arc_tails = np.array([arc.tail for arc in self.arcs_objects])
        self._arc_heads = np.array([arc.head for arc in self.arcs_objects])
        # etc.
        
        # Keep arrays in sync with objects
        self._sync_arrays_to_objects()  # When needed
        self._sync_objects_to_arrays()  # When needed
```

**Pros**:
- Preserves API
- Enables JIT for hot loops
- Gradual migration path

**Cons**:
- Synchronization overhead (when to sync?)
- Two sources of truth (error-prone)
- Complexity

**Recommendation**: Prototype and measure carefully.

---

## Next Steps (Immediate)

1. ✅ **Disable JIT for `collect_cycle()`** - get back to baseline
2. **Profile `_update_tree_sets()` in detail** - understand the 134s
3. **Implement JIT for `_update_tree_sets()`** - simple loop, high impact
4. **Measure and validate** - does it actually help?
5. **Document findings** - what works, what doesn't

## Next Steps (Phase 5 Continued)

### Week 1: Focus on `_update_tree_sets()` (134s target)

**Current implementation**:
```python
self.tree_adj = [[] for _ in range(self.node_count)]  # List of lists
for idx, arc in enumerate(self.arcs):
    if arc.in_tree:
        self.tree_adj[arc.tail].append(idx)
        self.tree_adj[arc.head].append(idx)
```

**JIT strategy**:
- Extract arc data to NumPy arrays (tails, heads, in_tree) - one-time cost
- JIT-compile the loop that builds tree_adj
- Or: Change tree_adj to NumPy-friendly format (CSR from start)

**Expected**: 2-3x speedup → 134s → 45-65s, save 70-90s

### Week 2: Profile and Optimize `rebuild()` (147s target)

**More complex**, involves:
- BFS (similar to collect_cycle)
- Parent/potential updates
- Numeric basis building (scipy calls)

**JIT strategy**:
- JIT-compile BFS portion
- Keep numeric basis building as-is (can't JIT scipy)
- Expected: 1.5-2x speedup → save 50-75s

### Week 3-4: Other Quick Wins

- `_compute_reduced_costs_vectorized()`: 9.8s → already vectorized, but could JIT
- Pricing operations: already fast, low priority

### Success Criteria (Revised)

- **Minimum**: 1.3x overall speedup (560s → 430s) via `_update_tree_sets()` alone
- **Target**: 1.5x overall speedup (560s → 375s) via both `_update_tree_sets()` and `rebuild()`
- **Stretch**: 1.7x overall speedup (560s → 330s) with additional optimizations

**No longer targeting collect_cycle JIT** - overhead too high.

---

## Lessons Learned

1. **JIT is not a silver bullet** - conversion overhead can exceed benefit
2. **Profile before AND after** - assumptions can be wrong
3. **Data layout matters** - Python objects vs NumPy arrays is a fundamental choice
4. **Original code can be well-optimized** - deque/dict are fast!
5. **Focus on biggest bottlenecks** - 50% of time in 2 functions, not 1

6. **Measure everything**:
   - JIT compilation time (first call slower)
   - Conversion overhead
   - Actual JIT speedup
   - Net effect

7. **Sometimes Python is fast enough**:
   - Native data structures (dict, deque, set) are highly optimized
   - List comprehensions are fast
   - Function call overhead is low compared to algorithm complexity

8. **Choose battles wisely**:
   - JIT simple loops: high ROI
   - JIT complex data structures: low ROI (conversion overhead)
   - JIT entire algorithms: only if data already in NumPy format

---

## Conclusion

The initial Phase 5 JIT attempt on `collect_cycle()` revealed an important lesson: **data format mismatch creates conversion overhead that can exceed JIT benefits**.

**Going forward**:
- Focus JIT efforts on `_update_tree_sets()` and `rebuild()` (281s combined, 50% of runtime)
- These are simpler loops that might JIT well
- If they also have conversion overhead, reconsider architectural refactor
- Be very careful to measure overhead vs benefit for each optimization

**Current status**: collect_cycle JIT disabled, back to ~560s baseline, ready to target bigger bottlenecks.

**Phase 5 revised goal**: 1.3-1.7x speedup by JIT-compiling `_update_tree_sets()` and `rebuild()`, avoiding conversion overhead.

---

**End of Phase 5 JIT Findings**
