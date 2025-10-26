# Phase 3 Profiling Analysis - Current Bottlenecks After Phase 1+2

**Date**: 2025-10-25  
**Branch**: `performance/profiling-phase3-analysis`  
**Test Case**: `gridgen_8_09a.min` (507 nodes, 4056 arcs, 1204 iterations)

## Executive Summary

After Phase 1 and Phase 2 optimizations, the solver is **2.40x faster** than the original baseline:
- **Original**: 19.9 seconds
- **After Phase 1+2**: 8.29 seconds
- **Speedup**: 2.40x (58% reduction)

The top bottlenecks have shifted. The new performance profile reveals:
1. **Basis operations still dominate**: 30% of time (rebuild + updates)
2. **Tree operations emerged as #2 bottleneck**: 13% of time
3. **Cycle collection**: 9% of time
4. **Array scanning eliminated**: ✅ No longer in top 30 functions

---

## Performance Comparison: Then vs Now

| Metric | Original | After Phase 1 | After Phase 2 | Improvement |
|--------|----------|---------------|---------------|-------------|
| **Total Time** | 19.9s | 12.78s | 8.29s | **2.40x** |
| **Function Calls** | 9.3M | 4.5M | 4.5M | 2.07x fewer |
| `_sync_vectorized_arrays()` | 4.26s (21%) | 0s | 0s | ✅ Eliminated |
| `any()` builtin | 1.43s (7%) | 1.43s | 0s | ✅ Eliminated |
| `rebuild()` | 1.66s (8%) | 1.85s | 1.52s (18%) | Slightly better |
| `estimate_condition_number()` | 4.58s (23%) | 0.49s | 0.43s (5%) | 10.6x faster |

---

## Current Bottleneck Analysis

### Top Time Consumers (by cumulative time)

| Function | Time | % | Calls | Per Call | Category |
|----------|------|---|-------|----------|----------|
| `_pivot()` | 6.00s | 72% | 1,204 | 4.98ms | **Pivot Operations** |
| `rebuild()` | 2.52s | 30% | 1,179 | 2.14ms | **Basis Operations** |
| `_find_entering_arc()` | 1.89s | 23% | 1,205 | 1.57ms | **Pricing** |
| `_update_tree_sets()` | 1.51s | 18% | 1,176 | 1.29ms | **Tree Operations** |
| `solve()` (Forrest-Tomlin) | 1.39s | 17% | 3,650 | 0.38ms | **Linear Algebra** |
| `_update_weight()` | 1.04s | 13% | 1,204 | 0.86ms | **Devex Pricing** |
| `project_column()` | 1.03s | 12% | 2,408 | 0.43ms | **Basis Operations** |
| `collect_cycle()` | 1.00s | 12% | 1,204 | 0.83ms | **Cycle Detection** |
| `estimate_condition_number()` | 0.43s | 5% | 117 | 3.70ms | **Adaptive Refactorization** |

### Top Time Consumers (by self time, excluding subcalls)

| Function | Self Time | % | What It Does |
|----------|-----------|---|--------------|
| `rebuild()` | 1.52s | 18% | LU factorization of basis matrix |
| `_update_tree_sets()` | 1.11s | 13% | Update parent/depth after basis change |
| `collect_cycle()` | 0.71s | 9% | Find cycle in spanning tree (BFS) |
| `append` (list) | 0.43s | 5% | Python list operations |
| `append` (deque) | 0.38s | 5% | Deque operations in BFS |
| `_compute_reduced_costs_vectorized()` | 0.37s | 4% | Calculate reduced costs |
| `solve()` (Forrest-Tomlin) | 0.33s | 4% | Linear system solves |
| `popleft` (deque) | 0.33s | 4% | Deque operations in BFS |

---

## Detailed Bottleneck Analysis

### Bottleneck #1: Basis Operations (30% of time)

**Total overhead**: ~2.5 seconds (30% of total time)

**Components**:
- `rebuild()`: 1.52s self-time, 2.52s cumulative (18%)
- `project_column()`: 1.03s cumulative (12%)
- `estimate_condition_number()`: 0.43s (5%)

**Analysis**:
- Still rebuilding basis 1,179 times in 1,204 iterations (98% rebuild rate!)
- Even with condition number checks every 10 pivots, still rebuilding almost every time
- This suggests the basis is becoming poorly conditioned frequently

**Why this happens**:
- Condition number threshold (1e12) is likely being exceeded frequently
- Fixed `ft_update_limit` of 64 might be too aggressive
- Network structure might lead to rapid conditioning deterioration

**Potential solutions**:
1. Increase condition number threshold (e.g., 1e14)
2. Increase `ft_update_limit` baseline (e.g., 100 instead of 64)
3. Reduce condition check frequency (e.g., every 20-50 pivots)
4. Use cheaper stability indicators instead of full condition number

**Expected impact**: Could reduce rebuild time by 50% if we can cut rebuild rate to 50%

---

### Bottleneck #2: Tree Operations (13% of time)

**Function**: `_update_tree_sets()` - 1.11s self-time (13%)

**What it does**: After each basis change, recomputes:
- Parent pointers for all nodes
- Depth of each node in spanning tree
- Uses BFS from root through tree arcs

**Called**: 1,176 times (once per rebuild)

**Analysis**:
Looking at the code, this is essentially doing a full tree traversal every time we rebuild. For 507 nodes, this isn't inherently expensive, but doing it 1,176 times adds up.

**Why it's expensive**:
- BFS requires queue operations (deque append/popleft: 0.71s combined)
- Set operations for visited tracking (0.25s)
- Directly tied to rebuild frequency

**Potential solutions**:
1. **Primary**: Reduce rebuild frequency (helps both #1 and #2)
2. **Secondary**: Cache parent/depth when they don't change
3. **Advanced**: Incremental tree updates instead of full BFS

**Expected impact**: Reduces proportionally with rebuild frequency reduction

---

### Bottleneck #3: Cycle Collection (9% of time)

**Function**: `collect_cycle()` - 0.71s self-time, 1.00s cumulative (9%)

**What it does**: For each pivot, finds the cycle created by adding entering arc to spanning tree

**Called**: 1,204 times (once per pivot - unavoidable)

**Analysis**:
- Uses BFS to find path between arc endpoints
- Returns list of (arc_idx, direction) tuples
- Per-call time: 0.83ms (actually quite good!)

**Why it takes time**:
- BFS queue operations
- Path reconstruction
- Called on every single pivot (can't skip)

**Potential solutions**:
1. **JIT compilation with Numba**: Could speed up BFS traversal
2. **Algorithm improvements**: Specialized tree path-finding
3. **Caching**: If we could predict which arcs will enter, could pre-compute

**Expected impact**: JIT could provide 20-30% speedup → ~0.2s savings

**Priority**: Lower - this is actually reasonably efficient, and we can't avoid calling it

---

### Bottleneck #4: Pricing Strategy (23% cumulative)

**Total overhead**: ~1.9 seconds (23% of total time)

**Components**:
- `_find_entering_arc()`: 1.89s cumulative
- `_select_entering_arc_vectorized()`: 1.89s
- `_compute_reduced_costs_vectorized()`: 0.37s
- `_update_weight()` (Devex): 1.04s

**Analysis**:
Pricing is working well (using vectorized operations), but it's called 1,205 times. The question is: are we making good choices?

**Key metric**: Iterations per solve
- gridgen_8_09a: 1,204 iterations
- goto_8_09a: 5,513 iterations (from benchmark)

**Potential issue**: We might be making poor arc selections, leading to more iterations than necessary.

**Degeneracy check**: Are many pivots degenerate (theta=0)?
- Would need additional profiling to measure
- Degenerate pivots make no progress but consume time

**Potential solutions**:
1. **Measure degeneracy rate**: Add counter for theta=0 pivots
2. **Better pricing**: Steepest edge instead of Devex?
3. **Candidate lists**: Don't scan all arcs, maintain small candidate set
4. **Phase 1 improvements**: Better initial basis → fewer Phase 2 iterations

**Expected impact**: If we can reduce iteration count 20-30%, save ~2-3 seconds

---

## What Got Better Since Original Profiling

✅ **Array synchronization overhead**: Eliminated (was 4.26s, 21%)
✅ **Artificial arc scanning**: Eliminated (was 1.43s, 7%)
✅ **Condition number checks**: 10.6x reduction (4.58s → 0.43s)
✅ **Total function calls**: 2.07x fewer (9.3M → 4.5M)
✅ **Generator expressions**: Eliminated (was 4.8M calls)

---

## Where Time Goes Now (8.29 seconds total)

```
Pivot Operations (6.00s, 72%):
  ├─ Basis operations: 2.52s (30%)
  │   ├─ rebuild(): 1.52s
  │   ├─ project_column(): 1.03s
  │   └─ estimate_condition_number(): 0.43s
  ├─ Pricing: 1.89s (23%)
  │   ├─ _select_entering_arc_vectorized(): 1.89s
  │   └─ _compute_reduced_costs_vectorized(): 0.37s
  ├─ Tree updates: 1.51s (18%)
  │   └─ _update_tree_sets(): 1.11s
  ├─ Cycle collection: 1.00s (12%)
  │   └─ collect_cycle(): 0.71s
  └─ Devex updates: 1.04s (13%)
      └─ _update_weight(): 1.04s

Linear Algebra (1.39s, 17%):
  └─ Forrest-Tomlin solve(): 0.33s × 3,650 calls

Overhead (0.90s, 11%):
  ├─ List/deque operations: 0.71s
  ├─ NumPy operations: 0.19s
  └─ Other: varies
```

---

## Phase 3 Optimization Opportunities

### Option A: Reduce Rebuild Frequency (High Impact, Medium Effort)

**Target**: Cut rebuild rate from 98% to 50% or less

**Approaches**:
1. Increase `ft_update_limit` from 64 to 100-150
2. Increase condition number threshold from 1e12 to 1e14
3. Reduce condition check interval from 10 to 20-50 pivots
4. Try fixed rebuild schedule instead of adaptive

**Expected Impact**: 
- Rebuild time: 1.52s → 0.76s (save 0.76s)
- Tree updates: 1.11s → 0.55s (save 0.56s)
- **Total savings: ~1.3s (16% speedup)**

**Risks**: Lower numerical stability, potential accuracy issues

---

### Option B: Reduce Iteration Count (Very High Impact, High Effort)

**Target**: Reduce iterations by 20-30% through better pricing

**Approaches**:
1. Measure and reduce degeneracy
2. Implement steepest-edge pricing (more expensive per iteration, but fewer iterations)
3. Improve Phase 1 basis quality
4. Add anti-cycling perturbations

**Expected Impact**: 
- 20% fewer iterations: 8.29s → 6.63s (save 1.66s, 20% speedup)
- 30% fewer iterations: 8.29s → 5.80s (save 2.49s, 30% speedup)

**Risks**: Higher complexity, might make each iteration more expensive

---

### Option C: JIT Compilation (Low-Medium Impact, Medium Effort)

**Target**: Speed up hot inner loops with Numba

**Functions to JIT**:
- `collect_cycle()`: 0.71s → ~0.50s (save 0.21s)
- `_update_tree_sets()`: 1.11s → ~0.77s (save 0.34s)
- `_compute_reduced_costs_vectorized()`: Already vectorized, might not benefit

**Expected Impact**: ~0.5s savings (6% speedup)

**Risks**: JIT warmup overhead, debugging harder, not all code JIT-compatible

---

### Option D: Algorithm Improvements (Variable Impact, High Effort)

**Possibilities**:
- Incremental tree updates instead of full BFS
- Better data structures (e.g., Euler tour tree)
- Specialized cycle-finding for network simplex
- Cache/memoize expensive computations

**Expected Impact**: Highly variable, 5-20% possible

**Risks**: High complexity, might introduce bugs

---

## Recommendation for Phase 3

**Primary recommendation: Option A - Reduce Rebuild Frequency**

**Rationale**:
1. **Clear, measurable bottleneck**: 30% of time spent on rebuilds
2. **Simple to implement**: Tune 2-3 parameters
3. **Low risk**: Easy to test and revert if numerical issues arise
4. **Compounds with other optimizations**: Helps both rebuild() and _update_tree_sets()

**Secondary recommendation: Option B - Measure Degeneracy**

**Rationale**:
1. **Data-driven**: Before optimizing iteration count, measure if degeneracy is an issue
2. **Low effort**: Add simple counters
3. **Informs future work**: Tells us if iteration reduction is the right path

**Implementation Plan**:
1. Add degeneracy tracking (theta=0 count)
2. Experiment with rebuild frequency parameters
3. Benchmark to find sweet spot between performance and stability
4. If degeneracy is high (>10%), investigate anti-degeneracy strategies

**Not recommended for Phase 3**: JIT compilation (Option C)
- Smaller impact (~6%)
- Adds complexity
- Better to exhaust simpler optimizations first

---

## Success Criteria for Phase 3

**Target performance**: 6-7 seconds on gridgen_8_09a (20-30% speedup)

**Metrics to track**:
- Solve time
- Rebuild frequency (should drop from 98% to 50-70%)
- Degeneracy rate (what % of pivots have theta ≈ 0)
- Solution accuracy (condition number, objective value precision)
- Benchmark success rate (8/18 → ?/18)

**Validation**:
- All existing problems still solve
- Solutions still validate correctly
- No accuracy degradation (compare objectives to LEMON)

---

## Conclusion

After Phase 1+2 optimizations, we've achieved **2.40x speedup** (19.9s → 8.29s), eliminating the most egregious inefficiencies:
- ✅ Array synchronization overhead (eliminated)
- ✅ Artificial arc scanning (eliminated)
- ✅ Excessive condition number checking (10x reduction)

The remaining bottlenecks are more fundamental:
1. **Basis operations**: 30% of time (rebuild + updates)
2. **Tree operations**: 13% of time (tied to rebuilds)
3. **Cycle collection**: 9% of time (unavoidable per-pivot cost)
4. **Pricing**: 23% of time (but is this buying us good progress?)

**Next frontier**: Either reduce rebuild frequency (easier, ~16% gain) or reduce iteration count (harder, ~20-30% gain).

The solver has matured from "obviously inefficient" to "reasonably optimized" - further gains will require either tuning trade-offs or algorithmic improvements.
