# Performance Optimization Roadmap

## Executive Summary

This document outlines the performance profiling results and optimization strategy for the network simplex solver. Through systematic profiling, we identified that **74% of runtime** is spent in just 5 functions, with the top optimization opportunities in linear algebra operations and pricing algorithms.

## Profiling Results

### Baseline Performance Metrics

| Problem Size | Arcs | Time (s) | Iterations |
|-------------|------|----------|------------|
| Small (5x5) | 25 | 0.047 | 26 |
| Medium (20x20) | 400 | 0.594 | 42 |
| Large (50x50) | 2,500 | 7.758 | 253 |
| Complex Network | ~180 | 0.853 | - |

### Hot Path Analysis (Large Problem)

**Top 5 Bottlenecks (74.3% of total runtime):**

1. **Forrest-Tomlin solve** - 3.324s (42.8%) - 20,729 calls
2. **SuperLU solve** - 1.165s (15.0%) - 20,729 calls  
3. **Devex pricing** - 0.705s (9.1%) - 254 calls
4. **Devex weight updates** - 0.354s (4.6%) - 20,225 calls
5. **Residual calculations** - 0.215s (2.8%) - 188,583 calls

## Optimization Strategy

### Phase 1: Vectorization (PRIORITY)

**Target Functions:**
- `forward_residual()` / `backward_residual()` 
- `_update_devex_weight()`
- `_find_entering_arc_devex()`

**Approach:**
- Replace per-arc Python loops with NumPy array operations
- Create parallel array representations of arc data
- Use boolean masks for arc eligibility

**Expected Speedup:** 2-5x on large problems

**Status:** ✅ Prototype implemented in `simplex_vectorized.py`

**Challenges Identified:**
- Exact tie-breaking behavior must match original for test compatibility
- Array synchronization overhead when switching between representations
- Need careful handling of edge cases (inf capacities, zero costs, etc.)

### Phase 2: Numba JIT Compilation

**Target Functions:**
- Vectorized pricing loops
- Residual calculation kernels
- Hot paths in Forrest-Tomlin operations

**Approach:**
- Apply `@numba.jit` decorators to pure numerical functions
- Ensure NumPy array inputs for compatibility
- Profile to verify speedup

**Expected Speedup:** Additional 1.5-3x

**Status:** 🔄 Not yet implemented

**Dependencies:**
- Add numba to project dependencies
- Complete Phase 1 vectorization first

### Phase 3: Algorithmic Improvements

**Opportunities:**
1. **Reduce basis solve frequency**
   - Cache repeated calculations
   - Batch multiple pricing operations
   
2. **Smarter Devex weight caching**
   - Avoid recalculating unchanged weights
   - Use incremental updates

3. **Parallel block pricing**
   - Evaluate multiple blocks concurrently
   - Use multiprocessing for independent blocks

**Expected Speedup:** 10-30% improvement

**Status:** 📋 Planned for future

## Implementation Status

### ✅ Completed

- [x] Comprehensive profiling with cProfile
- [x] Identified hot paths and bottlenecks
- [x] Created vectorized helper functions module (`simplex_vectorized.py`)
- [x] Prototyped NumPy array-based pricing
- [x] Documented optimization opportunities

### 🔄 In Progress

- [ ] Refine vectorized pricing to match exact tie-breaking behavior
- [ ] Add feature flag for enabling/disabling vectorization

### 📋 Planned

- [ ] Implement Numba JIT compilation
- [ ] Add performance regression tests
- [ ] Create benchmark suite
- [ ] Optimize Forrest-Tomlin update operations
- [ ] Investigate BLAS/LAPACK optimization opportunities

## Key Findings

### 1. Linear Algebra Dominates (58% of runtime)

The Forrest-Tomlin basis update system and SuperLU factorization account for the majority of time. These are already using optimized C/Fortran libraries (scipy.sparse), so gains here are limited.

**Recommendation:** Focus on reducing call frequency rather than optimizing the solves themselves.

### 2. Pricing is Highly Vectorizable (15% of runtime)

The Devex pricing loop iterates over arcs checking eligibility and computing merits. This is a perfect candidate for NumPy vectorization.

**Recommendation:** High priority for Phase 1 implementation.

### 3. Small Functions, High Call Frequency

Functions like `forward_residual()` are called 188k+ times but are individually fast (1µs each). The overhead is cumulative.

**Recommendation:** Batch/vectorize these operations.

### 4. Test Compatibility Challenge

Our vectorized implementation works correctly but produces slightly different iteration counts due to subtle differences in tie-breaking when multiple arcs have equal merit.

**Recommendation:** Either:
- Refine vectorized version to match exact behavior
- Update tests to be less strict about iteration counts
- Add tolerance-based comparison for iteration counts

## Lessons Learned

1. **Profiling First:** The profiling showed Forrest-Tomlin dominates runtime, not pricing as initially assumed. Data-driven optimization is critical.

2. **NumPy Arrays vs Python Objects:** Converting from list-of-objects to parallel arrays has overhead, but the vectorization benefits outweigh this for large problems.

3. **Exact Behavioral Matching:** Optimization can't change behavior in unexpected ways. Our vectorized version is "correct" but different enough to fail deterministic tests.

4. **Incremental Approach:** Building infrastructure (arrays, sync functions) separate from integration allows iterative refinement.

## Next Steps

### Immediate (Next PR)

1. Add feature flag for vectorization (default: off)
2. Create performance benchmark suite
3. Document array synchronization overhead
4. Add integration tests for vectorized mode

### Short Term (Next Quarter)

1. Complete vectorized pricing refinement
2. Implement Numba JIT for kernel functions  
3. Add performance regression tracking to CI/CD
4. Optimize Devex weight caching

### Long Term (Future)

1. Explore parallel pricing strategies
2. Investigate custom sparse linear algebra
3. Consider GPU acceleration for very large problems
4. Profile and optimize other solver components (basis updates, tree rebuilds)

## Recommendations for Contributors

If you want to work on performance optimization:

1. **Start with profiling** - Always profile before optimizing
2. **Use the existing infrastructure** - `simplex_vectorized.py` has ready-to-use functions
3. **Maintain test compatibility** - All 287 tests must pass
4. **Benchmark thoroughly** - Use `profile_solver.py` to verify improvements
5. **Document trade-offs** - Note any behavioral changes or edge cases

## Conclusion

Through systematic profiling, we've identified clear optimization opportunities that could yield 3-10x speedups on large problems. The infrastructure is in place for vectorization, and a clear roadmap exists for Numba JIT and algorithmic improvements.

The main challenge is maintaining exact behavioral compatibility with existing tests while achieving performance gains. This can be addressed through:
- Careful refinement of tie-breaking logic
- Feature flags for gradual rollout
- Updated test expectations for non-deterministic optimizations

**Estimated Total Speedup Potential:** 5-15x on large problems (>1000 arcs)
