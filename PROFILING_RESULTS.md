# Profiling Results - Performance Bottleneck Analysis

**Date**: 2025-10-25  
**Branch**: `performance/profiling-analysis`  
**Test Case**: `gridgen_8_09a.min` (507 nodes, 4056 arcs, 1204 iterations, 19.9 seconds)

## Executive Summary

Profiling identified **three critical bottlenecks** accounting for 50% of solver time:

1. **`_sync_vectorized_arrays()`: 4.26s (21%)** - Unnecessary full array sync every iteration
2. **`rebuild()`: 1.66s (8%)** - Basis rebuilt 98% of iterations (way too often!)
3. **`estimate_condition_number()`: 4.58s cumulative (23%)** - Expensive adaptive refactorization check

**Quick wins available**: Fixing just #1 and #2 could provide **30-50% speedup** with relatively simple changes.

---

## Detailed Analysis

### Performance Breakdown (19.9 seconds total)

| Function | Time | % | Calls | Per Call | Issue |
|----------|------|---|-------|----------|-------|
| `_sync_vectorized_arrays()` | 4.26s | 21% | 1,205 | 3.5ms | ⚠️ CRITICAL |
| `estimate_condition_number()` | 4.58s | 23% | 1,176 | 3.9ms | ⚠️ CRITICAL |
| `rebuild()` | 1.66s | 8% | 1,179 | 1.4ms | ⚠️ CRITICAL |
| Forrest-Tomlin `solve()` | 1.27s | 6% | 15,299 | 0.08ms | Expected |
| `_update_tree_sets()` | 1.21s | 6% | 1,176 | 1.0ms | Related to rebuild |
| `any()` builtin | 1.32s | 7% | 1,402 | 0.94ms | Likely artificial arc checks |
| `collect_cycle()` | 0.81s | 4% | 1,204 | 0.67ms | Expected (pivot) |
| **Other** | 4.85s | 24% | - | - | Various |

### Critical Bottleneck #1: `_sync_vectorized_arrays()` - 21% of time

**Location**: `simplex.py:423`

**Problem**: This function copies **all 4056 arcs** from Python `ArcState` objects to NumPy arrays on EVERY iteration:

```python
for i in range(n):  # n = 4056
    self.arc_flows[i] = self.arcs[i].flow
    self.arc_in_tree[i] = self.arcs[i].in_tree
    self.arc_artificial[i] = self.arcs[i].artificial
```

**Analysis**:
- Called: 1,205 times (once per iteration)
- Copies: 4056 arcs × 3 fields = 12,168 values per call
- Total copies: 14.7 million values!
- Time: 4.26 seconds (3.5ms per call)

**Why this is wasteful**:
- Each pivot only modifies ~10-50 arcs in the cycle
- 99% of arc data is unchanged
- We're copying everything just to update a handful of values

**Solution**: 
1. **Quick fix**: Only sync arcs that changed (track modified arcs in pivot)
2. **Better fix**: Update NumPy arrays directly during pivots, eliminate sync entirely
3. **Best fix**: Use NumPy arrays as primary storage, eliminate `ArcState` list duplication

**Expected impact**: 2-4 second reduction (10-20% speedup)

---

### Critical Bottleneck #2: Excessive Basis Rebuilds - 8% + overhead

**Location**: `basis.py:68`

**Problem**: Basis is rebuilt 1,179 times in 1,204 iterations = **98% rebuild rate!**

**Analysis**:
- Direct rebuild time: 1.66 seconds
- Condition number estimation: 4.58 seconds (for deciding when to rebuild)
- Tree structure updates: 1.21 seconds (after rebuild)
- **Total overhead: ~7.5 seconds (38% of total time!)**

**Why this is excessive**:
- Literature suggests rebuild every 50-200 pivots for stability
- We're rebuilding almost EVERY pivot!
- Adaptive refactorization is too aggressive

**Root cause**: Checking condition number every iteration:

```python
# In _pivot() - runs every iteration
if self.options.adaptive_refactorization:
    condition_number = self.basis.estimate_condition_number()
    if condition_number > threshold:
        force_rebuild = True
```

The condition number estimation itself is expensive (numpy.linalg.norm operations).

**Solutions**:
1. **Quick fix**: Reduce rebuild frequency threshold
2. **Better fix**: Only check condition number every N pivots (e.g., N=10)
3. **Best fix**: Use cheaper stability indicators (e.g., Forrest-Tomlin spike size)

**Expected impact**: 4-6 second reduction (20-30% speedup)

---

### Critical Bottleneck #3: Condition Number Estimation - 23% cumulative

**Location**: `basis.py:255`

**Problem**: Computing matrix condition number is very expensive:

```python
def estimate_condition_number(self):
    # ... builds dense matrix from sparse basis ...
    col_norms = np.linalg.norm(basis_matrix, axis=0)  # Expensive!
    # ... more numpy operations ...
```

**Analysis**:
- Called: 1,176 times (every potential rebuild check)
- Time: 4.58 seconds cumulative
- Uses: 3,494 calls to `numpy.linalg.norm()` (1.17s)

**Why this is wasteful**:
- We're computing exact condition number to decide IF we should rebuild
- The check is almost as expensive as the rebuild itself!
- Classic case of "the cure is worse than the disease"

**Solutions**:
1. **Quick fix**: Cache condition number, only recompute every N pivots
2. **Better fix**: Use cheaper stability proxy (Forrest-Tomlin update sizes)
3. **Best fix**: Fixed rebuild schedule + occasional condition checks

**Expected impact**: 3-4 second reduction (15-20% speedup) when combined with #2

---

## Secondary Issues

### Issue: Forrest-Tomlin Overhead

- 15,299 linear system solves in 1204 iterations = **12.7 solves per iteration**
- Time: 1.27s (6.4% of total)
- Per solve: 0.08ms (actually quite good!)

**Analysis**: This is expected and reasonable. Each iteration needs:
- 1 solve for dual variables (node potentials)
- Multiple solves for cycle collection
- Solves for Devex weight updates

**Conclusion**: Not a priority optimization target.

### Issue: Generic Expression Overhead

- 4.8 million calls to generator expression at line 1073
- Time: 1.14 seconds

Let me check what this is:

```python
# simplex.py:1073 - likely in _pivot()
# Probably: any(arc.artificial and arc.flow > tolerance for arc in self.arcs)
```

This is checking for artificial flow on every iteration. With 4056 arcs, this scans all arcs repeatedly.

**Solution**: Track artificial arc count instead of scanning.

---

## Optimization Roadmap

### Phase 1: Quick Wins (Target: 40-50% speedup)

**Priority 1: Fix `_sync_vectorized_arrays()` overhead**
- **Impact**: ~20% speedup (4 seconds)
- **Effort**: Medium
- **Approach**: Only sync changed arcs, or update arrays directly in pivot

**Priority 2: Reduce basis rebuild frequency**
- **Impact**: ~25% speedup (5 seconds)  
- **Effort**: Low
- **Approach**: 
  - Option A: Check condition number every 10 pivots instead of every pivot
  - Option B: Use fixed rebuild schedule (every 100 pivots)
  - Option C: Disable adaptive refactorization temporarily

**Priority 3: Eliminate artificial arc scanning**
- **Impact**: ~5% speedup (1 second)
- **Effort**: Low
- **Approach**: Track count of artificial arcs with non-zero flow

**Combined Phase 1 impact**: 10 seconds reduction (19.9s → ~10s), **2x speedup**

### Phase 2: Structural Improvements (Target: additional 2-3x)

**1. Eliminate data duplication**
- Use NumPy arrays as primary storage
- Remove `ArcState` list entirely
- Keep only mapping for node IDs

**2. Reduce iteration count**
- Profile degeneracy rate
- Improve pricing strategy
- Better initial basis

**3. Optimize hot functions**
- JIT compile critical paths with Numba
- Optimize cycle collection
- Cache more intermediate results

### Phase 3: Algorithm Improvements

- Better pricing strategies (candidate lists, partial pricing)
- Improved Phase 1 initialization
- Cost scaling
- Dual simplex option

---

## Immediate Next Steps

1. **Implement Quick Fix for Rebuild Frequency**
   - Add option to check condition number every N pivots
   - Test with N=10, N=50, N=100
   - Measure impact on solve time and numerical stability

2. **Optimize `_sync_vectorized_arrays()`**
   - Track which arcs were modified in pivot
   - Only sync those arcs
   - Or update arrays directly, skip sync

3. **Add Degeneracy Tracking**
   - Count degenerate pivots
   - Measure if this correlates with rebuild frequency
   - Understand why we need so many iterations

4. **Benchmark Improvements**
   - Re-run profiling after each fix
   - Measure cumulative speedup
   - Verify solution correctness maintained

---

## Testing Strategy

For each optimization:
1. Run profiler on `gridgen_8_09a.min` to measure speedup
2. Run full benchmark suite to verify correctness
3. Check that all problems still validate
4. Monitor memory usage (shouldn't increase)

**Success Criteria for Phase 1**:
- `gridgen_8_09a`: 19.9s → <10s (2x speedup)
- All solutions still validate correctly
- No increase in memory usage

---

## Conclusion

The profiling clearly shows that **overhead from data synchronization and excessive basis rebuilds** is killing performance, not the core simplex algorithm itself.

The good news: These are **fixable inefficiencies**, not fundamental algorithm limitations.

**Conservative estimate**: Phase 1 optimizations could achieve **2x speedup** (20s → 10s for this problem), which would:
- Make all _09 variants solvable (currently timeout)
- Improve success rate from 39% to ~60-70%
- Make the solver usable for small benchmarks

**Optimistic estimate**: Combined Phase 1+2 could achieve **4-5x speedup**, making solver competitive for problems up to 2K-4K nodes.

---

**Status**: Analysis complete, ready to implement Phase 1 optimizations
