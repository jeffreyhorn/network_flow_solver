# Performance Optimization Strategy 2025

## Executive Summary

This document presents a comprehensive, actionable optimization strategy for the network simplex solver based on profiling analysis and modern Python performance techniques. We propose a series of **small, incremental projects** targeting vectorization, JIT compilation, and algorithmic improvements that could yield **5-20x speedups** on large problems.

## Current Performance Baseline

Based on profiling analysis (see `OPTIMIZATION_ROADMAP.md`):

| Problem Size | Nodes | Arcs | Time (s) | Iterations | Key Bottleneck |
|-------------|-------|------|----------|------------|----------------|
| Small (5×5) | 11 | 25 | 0.047 | 26 | Setup overhead |
| Medium (20×20) | 41 | 400 | 0.594 | 42 | Pricing + Solves |
| Large (50×50) | 101 | 2,500 | 7.758 | 253 | Linear algebra (58%) |

### Hot Path Breakdown (Large Problem)

**Top 5 Bottlenecks (74% of total runtime):**

1. **Forrest-Tomlin solve** - 42.8% (3.324s, 20,729 calls)
2. **SuperLU solve** - 15.0% (1.165s, 20,729 calls)
3. **Devex pricing** - 9.1% (0.705s, 254 calls)
4. **Devex weight updates** - 4.6% (0.354s, 20,225 calls)
5. **Residual calculations** - 2.8% (0.215s, 188,583 calls)

## Optimization Roadmap: Small Incremental Projects

### Phase 1: Low-Hanging Fruit Vectorization (Weeks 1-3)

#### Project 1.1: Vectorize Residual Calculations
**Difficulty:** ⭐ Easy  
**Expected Speedup:** 2-3x on residual computations (saves ~0.15s on large problems)  
**Files:** `simplex.py` (ArcState methods)

**Current State:**
```python
# Called 188k+ times per solve
def forward_residual(self) -> float:
    if math.isinf(self.upper):
        return math.inf
    return self.upper - self.flow

def backward_residual(self) -> float:
    return self.flow - self.lower
```

**Optimization Approach:**
- Create parallel NumPy arrays: `arc_upper`, `arc_lower`, `arc_flow`
- Vectorized computation:
  ```python
  forward_res = np.minimum(arc_upper - arc_flow, np.inf)
  backward_res = arc_flow - arc_lower
  ```
- Synchronize arrays only when arc flows change (pivot operations)

**Implementation Steps:**
1. Add array fields to `NetworkSimplex.__init__`
2. Create `_sync_arc_arrays()` helper method
3. Replace individual residual calls with array indexing
4. Add feature flag: `SolverOptions.use_vectorized_residuals`
5. Benchmark and verify correctness

**Success Criteria:**
- All tests pass
- 2x faster on residual-heavy operations
- Memory increase < 10%

---

#### Project 1.2: Vectorize Arc Eligibility Checking
**Difficulty:** ⭐⭐ Medium  
**Expected Speedup:** 3-5x on pricing (saves ~0.5s on large problems)  
**Files:** `simplex_pricing.py`

**Current State:**
```python
# Iterates through arcs checking eligibility one by one
for idx in range(start, end):
    arc = arcs[idx]
    if not arc.in_tree:
        # Check residuals, compute reduced cost...
```

**Optimization Approach:**
- Create boolean mask for tree membership: `arc_in_tree` (NumPy array)
- Vectorized reduced cost computation
- Use `np.where()` and `np.argmin()` for candidate selection

**Implementation Steps:**
1. Add `arc_in_tree` boolean array
2. Create `compute_reduced_costs_vectorized()` function
3. Implement vectorized candidate scoring
4. Handle tie-breaking carefully to match existing behavior
5. Add feature flag: `SolverOptions.use_vectorized_pricing`

**Success Criteria:**
- Pricing time reduced by 50%+
- Iteration counts match original (±5% tolerance)
- All tests pass with vectorized mode enabled

---

#### Project 1.3: Batch Devex Weight Updates
**Difficulty:** ⭐⭐ Medium  
**Expected Speedup:** 2-4x on weight updates (saves ~0.25s on large problems)  
**Files:** `simplex_pricing.py` (DevexPricing class)

**Current State:**
```python
# Updates weights one arc at a time after each pivot
def _update_devex_weight(self, entering_idx, projection):
    # ...individual calculations
```

**Optimization Approach:**
- Compute projections for multiple arcs in batch
- Use NumPy broadcasting for weight formula
- Cache unchanged weights between pivots

**Implementation Steps:**
1. Create `_batch_update_weights()` method
2. Leverage existing `self.weights` NumPy array
3. Use boolean indexing for selective updates
4. Add weight staleness tracking

**Success Criteria:**
- Weight update time reduced by 50%+
- Convergence behavior unchanged
- Memory overhead < 5%

---

### Phase 2: Numba JIT Compilation (Weeks 4-6)

#### Project 2.1: JIT-Compile Pricing Kernels
**Difficulty:** ⭐⭐ Medium  
**Expected Speedup:** 2-3x additional on pricing  
**Dependencies:** Phase 1 vectorization completed  
**New Dependency:** `numba>=0.58`

**Optimization Approach:**
```python
import numba

@numba.jit(nopython=True, cache=True)
def compute_reduced_costs_jit(
    arc_costs, arc_tails, arc_heads, node_potentials, arc_in_tree
):
    n = len(arc_costs)
    reduced_costs = np.empty(n)
    for i in range(n):
        if not arc_in_tree[i]:
            reduced_costs[i] = arc_costs[i] + node_potentials[arc_tails[i]] - node_potentials[arc_heads[i]]
        else:
            reduced_costs[i] = 0.0
    return reduced_costs
```

**Implementation Steps:**
1. Add `numba` to `pyproject.toml` optional dependencies
2. Extract pure numerical kernels from pricing logic
3. Apply `@numba.jit` with `nopython=True`
4. Ensure all inputs are NumPy arrays (no Python objects)
5. Add feature flag: `SolverOptions.use_numba_jit`
6. Benchmark compiled vs interpreted

**Success Criteria:**
- First run compiles without errors
- Cached runs are 2x+ faster than vectorized-only
- All tests pass

---

#### Project 2.2: JIT-Compile Residual Kernels
**Difficulty:** ⭐ Easy  
**Expected Speedup:** 1.5-2x additional  
**Dependencies:** Project 1.1, Project 2.1

**Optimization Approach:**
```python
@numba.jit(nopython=True, cache=True)
def compute_residuals_jit(arc_upper, arc_lower, arc_flow):
    forward_res = arc_upper - arc_flow
    backward_res = arc_flow - arc_lower
    return forward_res, backward_res
```

**Implementation Steps:**
1. Extract residual calculations into pure functions
2. Apply `@numba.jit`
3. Integrate with vectorized residual system from 1.1

**Success Criteria:**
- Residual computation time reduced by 50%+
- No behavioral changes

---

#### Project 2.3: JIT-Compile Hot Loops in Pivot Operations
**Difficulty:** ⭐⭐⭐ Hard  
**Expected Speedup:** 1.2-1.5x on overall runtime  
**Dependencies:** Projects 2.1, 2.2

**Target Functions:**
- `_pivot()` - cycle flow updates
- `_update_tree_sets()` - adjacency list rebuilding
- Arc flow adjustment loops

**Optimization Approach:**
- Identify pure numerical loops within pivot logic
- Extract into `@numba.jit` functions
- Careful handling of edge cases (infinite capacities, degeneracy)

**Implementation Steps:**
1. Profile pivot operations in detail
2. Extract inner loops into separate functions
3. Convert to NumPy-compatible operations
4. Apply JIT compilation
5. Extensive testing for edge cases

**Success Criteria:**
- Pivot time reduced by 20%+
- All degeneracy handling preserved
- Edge cases (unbounded, zero-cost arcs) handled correctly

---

### Phase 3: Algorithmic Improvements (Weeks 7-10)

#### Project 3.1: Reduce Basis Solve Frequency
**Difficulty:** ⭐⭐⭐ Hard  
**Expected Speedup:** 10-20% on large problems  
**Files:** `basis.py`, `simplex.py`

**Current Issue:**
- Forrest-Tomlin solve called 20,729 times for 253 iterations (81 solves/iteration)
- Many solves are for projections and potential calculations that could be cached

**Optimization Strategies:**

**A. Cache Repeated Projections**
```python
class TreeBasis:
    def __init__(self, ...):
        self.projection_cache = {}  # col_idx -> projection vector
        
    def project_column(self, arc):
        col_hash = hash(arc.key)  # or stable identifier
        if col_hash in self.projection_cache and not self._basis_changed:
            return self.projection_cache[col_hash]
        
        projection = self._compute_projection(arc)
        self.projection_cache[col_hash] = projection
        return projection
```

**B. Batch Potential Calculations**
```python
def _batch_compute_potentials(self, arc_indices):
    """Compute potentials for multiple arcs at once."""
    # Single solve with multiple RHS vectors
    rhs_matrix = self._build_rhs_matrix(arc_indices)
    potentials = solve_lu(self.lu_factors, rhs_matrix)
    return potentials
```

**Implementation Steps:**
1. Add cache invalidation on basis changes
2. Track which arcs need projection updates
3. Implement batch solve capability
4. Benchmark cache hit rates

**Success Criteria:**
- Basis solve calls reduced by 30%+
- Cache hit rate > 60%
- Memory overhead < 20MB

---

#### Project 3.2: Smarter Devex Weight Caching
**Difficulty:** ⭐⭐ Medium  
**Expected Speedup:** 5-10%  
**Files:** `simplex_pricing.py`

**Current Issue:**
- Weights recalculated for all arcs periodically
- Many arcs haven't changed since last calculation

**Optimization Approach:**
```python
class DevexPricing:
    def __init__(self, ...):
        self.weight_last_updated = np.zeros(arc_count, dtype=int)
        self.iteration_counter = 0
        
    def select_entering_arc(self, ...):
        # Only update weights for arcs affected by last pivot
        affected_arcs = self._get_affected_arcs(last_entering, last_leaving)
        for idx in affected_arcs:
            self._update_single_weight(idx)
            self.weight_last_updated[idx] = self.iteration_counter
```

**Implementation Steps:**
1. Track arc dependencies (which arcs affected by each pivot)
2. Implement incremental weight updates
3. Add staleness threshold (e.g., force update every 10 iterations)

**Success Criteria:**
- Weight update calls reduced by 50%+
- Convergence rate unchanged
- Pricing quality maintained

---

#### Project 3.3: Early Pivot Termination
**Difficulty:** ⭐ Easy  
**Expected Speedup:** 2-5% on highly degenerate problems  
**Files:** `simplex.py`

**Optimization Approach:**
```python
def _pivot(self, arc_idx, direction):
    # ... compute cycle ...
    
    # NEW: Quick degeneracy check before expensive operations
    if self._is_degenerate_pivot(cycle):
        # Skip expensive Devex updates for zero-flow pivots
        self._quick_degenerate_pivot(arc_idx, direction)
        return
    
    # ... normal pivot logic ...
```

**Implementation Steps:**
1. Add fast degeneracy detection
2. Create lightweight pivot path for degenerate cases
3. Skip unnecessary weight updates and projections

**Success Criteria:**
- Degenerate pivot time reduced by 30%+
- No change in iteration counts
- All tests pass

---

#### Project 3.4: Parallel Block Pricing (Advanced)
**Difficulty:** ⭐⭐⭐⭐ Very Hard  
**Expected Speedup:** 1.5-2x on multi-core systems  
**Dependencies:** All Phase 1 & 2 projects  
**New Dependencies:** `multiprocessing` or `concurrent.futures`

**Optimization Approach:**
- Divide arc set into independent blocks
- Evaluate blocks in parallel using multiple CPU cores
- Collect results and select best candidate

**Challenges:**
- Maintaining deterministic behavior
- Managing shared state (basis, potentials)
- Overhead of inter-process communication

**Implementation Steps:**
1. Design block partitioning strategy
2. Implement parallel pricing with process pool
3. Add synchronization for shared state
4. Benchmark overhead vs speedup
5. Feature flag: `SolverOptions.parallel_pricing`

**Success Criteria:**
- 1.5x speedup on 4+ core systems
- Deterministic results (same solution every run)
- Graceful fallback to serial on single-core systems

---

### Phase 4: Advanced Optimizations (Weeks 11-16)

#### Project 4.1: Custom Sparse LU with Caching
**Difficulty:** ⭐⭐⭐⭐ Very Hard  
**Expected Speedup:** 5-15% on large problems  
**Files:** `basis_lu.py`

**Concept:**
- Exploit network structure for faster factorization
- Cache symbolic factorization (sparsity pattern doesn't change often)
- Use specialized basis update formulas

**Research Required:**
- Survey sparse LU libraries (UMFPACK, KLU, CHOLMOD)
- Investigate basis update techniques from literature
- Benchmark against current SuperLU approach

---

#### Project 4.2: GPU Acceleration (Exploratory)
**Difficulty:** ⭐⭐⭐⭐⭐ Extremely Hard  
**Expected Speedup:** 10-50x on very large problems (>10k arcs)  
**New Dependencies:** `cupy`, `cusparse`

**Concept:**
- Port vectorized operations to GPU
- Use GPU sparse linear algebra (cuSPARSE)
- Hybrid CPU/GPU approach (pricing on GPU, pivots on CPU)

**Feasibility Study Needed:**
- Memory transfer overhead
- Problem size threshold for GPU benefits
- Compatibility with existing code

---

## Implementation Priority Matrix

| Project | Difficulty | Impact | Priority | Estimated Time |
|---------|-----------|--------|----------|----------------|
| 1.1 Vectorize Residuals | Easy | Medium | **HIGH** | 3-5 days |
| 1.2 Vectorize Eligibility | Medium | High | **HIGH** | 5-7 days |
| 1.3 Batch Devex Updates | Medium | Medium | **MEDIUM** | 4-6 days |
| 2.1 JIT Pricing Kernels | Medium | High | **HIGH** | 5-7 days |
| 2.2 JIT Residual Kernels | Easy | Low | **MEDIUM** | 2-3 days |
| 2.3 JIT Pivot Loops | Hard | Medium | **MEDIUM** | 7-10 days |
| 3.1 Cache Basis Solves | Hard | High | **HIGH** | 10-14 days |
| 3.2 Smart Weight Caching | Medium | Medium | **MEDIUM** | 5-7 days |
| 3.3 Early Pivot Termination | Easy | Low | **LOW** | 2-3 days |
| 3.4 Parallel Pricing | Very Hard | High | **LOW** | 14-21 days |
| 4.1 Custom Sparse LU | Very Hard | Medium | **RESEARCH** | 3-4 weeks |
| 4.2 GPU Acceleration | Extreme | Very High | **RESEARCH** | 2-3 months |

## Recommended Implementation Order

### Quarter 1: Quick Wins (Weeks 1-6)
1. **Project 1.1** - Vectorize Residuals (Week 1)
2. **Project 1.2** - Vectorize Eligibility (Weeks 2-3)
3. **Project 2.1** - JIT Pricing Kernels (Weeks 4-5)
4. **Project 1.3** - Batch Devex Updates (Week 6)

**Expected Cumulative Speedup:** 3-5x on large problems

### Quarter 2: Algorithmic Depth (Weeks 7-12)
5. **Project 3.1** - Cache Basis Solves (Weeks 7-9)
6. **Project 3.2** - Smart Weight Caching (Weeks 10-11)
7. **Project 2.2** - JIT Residual Kernels (Week 12)

**Expected Cumulative Speedup:** 5-8x on large problems

### Quarter 3: Polish & Advanced (Weeks 13-18)
8. **Project 2.3** - JIT Pivot Loops (Weeks 13-15)
9. **Project 3.3** - Early Pivot Termination (Week 16)
10. **Project 4.1** - Research custom sparse LU (Weeks 17-18)

**Expected Cumulative Speedup:** 6-10x on large problems

### Quarter 4: Experimental (Weeks 19-24)
11. **Project 3.4** - Parallel Pricing (Weeks 19-22)
12. **Project 4.2** - GPU feasibility study (Weeks 23-24)

**Expected Cumulative Speedup:** 8-20x on large problems (hardware-dependent)

## Feature Flag Strategy

To enable gradual rollout and easy A/B testing:

```python
@dataclass
class SolverOptions:
    # ... existing options ...
    
    # Performance flags (all default to auto-detect)
    use_vectorized_residuals: bool = True
    use_vectorized_pricing: bool = True
    use_numba_jit: bool | None = None  # Auto-detect numba availability
    cache_basis_solves: bool = True
    cache_devex_weights: bool = True
    parallel_pricing: bool = False  # Opt-in for now
    
    def __post_init__(self):
        # Auto-detect numba
        if self.use_numba_jit is None:
            try:
                import numba
                self.use_numba_jit = True
            except ImportError:
                self.use_numba_jit = False
```

## Benchmarking & Validation Strategy

### Performance Regression Tests

Create `tests/performance/test_regression.py`:

```python
def test_performance_baseline():
    """Ensure optimizations don't regress performance."""
    problem = create_large_transportation(30, 30)
    
    start = time.time()
    result = solve_min_cost_flow(problem)
    elapsed = time.time() - start
    
    # Should solve in < 5 seconds with optimizations
    assert elapsed < 5.0, f"Performance regression: {elapsed:.2f}s"
    assert result.iterations < 300, f"Too many iterations: {result.iterations}"
```

### Correctness Validation

```python
def test_vectorized_matches_original():
    """Verify vectorized code produces identical results."""
    problem = create_test_problem()
    
    # Original implementation
    result_original = solve_min_cost_flow(
        problem,
        options=SolverOptions(use_vectorized_residuals=False)
    )
    
    # Vectorized implementation
    result_vectorized = solve_min_cost_flow(
        problem,
        options=SolverOptions(use_vectorized_residuals=True)
    )
    
    assert abs(result_original.objective - result_vectorized.objective) < 1e-6
    assert abs(result_original.iterations - result_vectorized.iterations) <= 5  # Allow small variation
```

## Risk Mitigation

### Potential Pitfalls

1. **Behavioral Changes:**
   - **Risk:** Vectorized/JIT code produces different iteration counts
   - **Mitigation:** Extensive testing, tolerance-based comparisons, feature flags

2. **Memory Overhead:**
   - **Risk:** Array duplication increases memory 2-3x
   - **Mitigation:** Lazy array creation, memory profiling, configurable limits

3. **Numba Compilation Time:**
   - **Risk:** First run slower due to JIT compilation
   - **Mitigation:** Cache compiled functions, document warmup behavior

4. **Maintenance Burden:**
   - **Risk:** Parallel code paths harder to maintain
   - **Mitigation:** Comprehensive tests, clear documentation, gradual deprecation of old paths

5. **Platform Compatibility:**
   - **Risk:** Numba/multiprocessing issues on some platforms
   - **Mitigation:** Graceful fallbacks, platform-specific tests in CI/CD

## Success Metrics

### Performance Targets

| Problem Size | Current Time | Target Time | Speedup Goal |
|-------------|--------------|-------------|--------------|
| Small (5×5) | 0.047s | 0.020s | 2x |
| Medium (20×20) | 0.594s | 0.120s | 5x |
| Large (50×50) | 7.758s | 0.800s | 10x |
| XLarge (100×100) | ~60s (est) | 5.000s | 12x |

### Quality Targets

- ✅ All existing tests pass
- ✅ Correctness within 1e-6 tolerance
- ✅ Iteration counts within ±5% of original
- ✅ Memory overhead < 2x on large problems
- ✅ No compilation errors with Numba
- ✅ Deterministic results (same seed → same solution)

## Conclusion

This optimization strategy provides a **clear, incremental path** to dramatically improving solver performance through:

1. **Vectorization** - Eliminate Python loops, leverage NumPy
2. **JIT Compilation** - Compile hot paths to machine code with Numba
3. **Algorithmic Improvements** - Reduce unnecessary computation through caching and smarter algorithms

By following the **quarter-by-quarter roadmap** and implementing **small, testable projects**, we can achieve **8-20x speedups** while maintaining code quality and correctness.

### Next Steps

1. **Immediate:** Implement Project 1.1 (Vectorize Residuals) as proof-of-concept
2. **Short-term:** Complete Phase 1 vectorization (Projects 1.1-1.3)
3. **Medium-term:** Add Numba JIT compilation (Projects 2.1-2.3)
4. **Long-term:** Algorithmic optimizations and research advanced techniques

**The key to success is incremental progress with continuous validation.**
