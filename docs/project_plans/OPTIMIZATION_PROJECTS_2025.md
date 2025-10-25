# Optimization Projects - Data-Driven Roadmap

## Overview

Based on comprehensive profiling (see `PROFILING_ANALYSIS_2025.md`), we have **4 high-impact projects** that can achieve **3-4x speedup** on large problems.

**Current Baseline:** Large network (4,267 arcs) = 65.9 seconds

## Project 1: Cache Basis Solves 🔥 HIGHEST PRIORITY

**Impact:** 50% speedup (33s saved)  
**Difficulty:** ⭐⭐⭐ Hard  
**Timeline:** 2-3 weeks  
**Priority:** CRITICAL

### Problem
- 131,331 basis solves in 65.9s (369 per iteration!)
- Takes 43.9s total (66.6% of runtime)
- Many projections are likely redundant

### Solution
Cache projection results with smart invalidation

### Implementation Plan

#### Week 1: Analysis & Design
1. **Profile projection patterns**
   ```python
   # Add instrumentation
   projection_requests = {}  # arc_key -> request_count
   ```
   
2. **Measure potential**
   - How many unique projections per solve?
   - How many repeated requests?
   - What's the working set size?

3. **Design cache strategy**
   - LRU cache with configurable size
   - Invalidation on basis changes
   - Cache key: arc identifier + basis version

#### Week 2: Implementation
```python
class TreeBasis:
    def __init__(self, ...):
        self.projection_cache = {}  # (arc_key, basis_version) -> projection
        self.basis_version = 0  # Increment on basis changes
        self.cache_hits = 0
        self.cache_misses = 0
    
    def project_column(self, arc):
        cache_key = (arc.key, self.basis_version)
        
        if cache_key in self.projection_cache:
            self.cache_hits += 1
            return self.projection_cache[cache_key]
        
        # Cache miss - compute projection
        self.cache_misses += 1
        projection = self._compute_projection_uncached(arc)
        
        # Store in cache (with size limit)
        if len(self.projection_cache) < MAX_CACHE_SIZE:
            self.projection_cache[cache_key] = projection
        
        return projection
    
    def replace_arc(self, leaving_idx, entering_idx, ...):
        # ... existing logic ...
        
        # Invalidate cache on basis change
        self.basis_version += 1
        self.projection_cache.clear()  # Or use smarter invalidation
        
        return result
```

#### Week 3: Optimization & Validation
1. **Tune cache parameters**
   - Optimal cache size
   - Smart invalidation (partial vs full clear)
   - Cache eviction policy (LRU vs FIFO)

2. **Measure results**
   - Cache hit rate (target: >60%)
   - Solve call reduction
   - Overall speedup

3. **Add monitoring**
   ```python
   def get_cache_stats(self):
       total = self.cache_hits + self.cache_misses
       hit_rate = self.cache_hits / total if total > 0 else 0
       return {
           'hits': self.cache_hits,
           'misses': self.cache_misses,
           'hit_rate': hit_rate,
           'size': len(self.projection_cache)
       }
   ```

### Success Criteria
- [ ] Cache hit rate > 60%
- [ ] Basis solve calls reduced by 50%+
- [ ] Runtime reduced by 30%+ on large problems
- [ ] All tests pass
- [ ] Memory overhead < 50MB

### Files to Modify
- `src/network_solver/basis.py` - Add caching logic
- `src/network_solver/data.py` - Add `SolverOptions.projection_cache_size`
- `tests/unit/test_basis_numeric.py` - Add cache tests

---

## Project 2: Vectorize Pricing

**Impact:** 10% speedup (7s saved)  
**Difficulty:** ⭐⭐ Medium  
**Timeline:** 1 week  
**Priority:** HIGH

### Problem
- `select_entering_arc` takes 4.4s (6.7% of runtime)
- Python loops over thousands of arcs
- Called 357 times (once per iteration)

### Solution
Replace Python loops with NumPy vectorized operations

### Implementation Plan

#### Day 1-2: Create Array Infrastructure
```python
class NetworkSimplex:
    def __init__(self, ...):
        # Parallel arrays for vectorization
        self.arc_costs = np.array([arc.cost for arc in self.arcs])
        self.arc_tails = np.array([arc.tail for arc in self.arcs])
        self.arc_heads = np.array([arc.head for arc in self.arcs])
        self.arc_in_tree = np.array([arc.in_tree for arc in self.arcs])
        self.arc_flows = np.array([arc.flow for arc in self.arcs])
        self.arc_uppers = np.array([arc.upper for arc in self.arcs])
        self.arc_lowers = np.array([arc.lower for arc in self.arcs])
```

#### Day 3-4: Vectorized Pricing Logic
```python
def _find_entering_arc_vectorized(self):
    # Compute reduced costs for all non-tree arcs
    rc = self.arc_costs + self.node_potentials[self.arc_tails] - self.node_potentials[self.arc_heads]
    
    # Forward candidates (rc < -tol and forward_res > 0)
    forward_res = self.arc_uppers - self.arc_flows
    forward_eligible = (~self.arc_in_tree) & (rc < -self.tolerance) & (forward_res > self.tolerance)
    
    # Backward candidates (rc > tol and backward_res > 0)
    backward_res = self.arc_flows - self.arc_lowers
    backward_eligible = (~self.arc_in_tree) & (rc > self.tolerance) & (backward_res > self.tolerance)
    
    # Find best candidate
    if np.any(forward_eligible):
        candidates = np.where(forward_eligible)[0]
        best_idx = candidates[np.argmin(rc[candidates])]
        return best_idx, 1
    elif np.any(backward_eligible):
        candidates = np.where(backward_eligible)[0]
        best_idx = candidates[np.argmax(rc[candidates])]
        return best_idx, -1
    
    return None
```

#### Day 5: Integration & Testing
1. Add sync function for arc arrays
2. Test against original implementation
3. Verify iteration counts match (±5%)

### Success Criteria
- [ ] Pricing time reduced by 70%+
- [ ] Iteration counts within 5% of original
- [ ] All tests pass

### Files to Modify
- `src/network_solver/simplex.py` - Add arrays, vectorized pricing
- `src/network_solver/data.py` - Add `SolverOptions.use_vectorized_pricing`

---

## Project 3: Batch Devex Weight Updates

**Impact:** 4% speedup (3s saved)  
**Difficulty:** ⭐⭐ Medium  
**Timeline:** 1 week  
**Priority:** MEDIUM

### Problem
- `_update_weight` called 127,063 times (3.9s total)
- Many weights unchanged between pivots
- Individual updates expensive

### Solution
Batch updates using NumPy, only update changed arcs

### Implementation Plan

#### Day 1-2: Track Affected Arcs
```python
class DevexPricing:
    def select_entering_arc(self, ...):
        # ... existing logic ...
        
        # After pivot, track which arcs need weight updates
        affected_arcs = self._get_affected_arcs(entering_arc, leaving_arc)
        self._batch_update_weights(affected_arcs)
```

#### Day 3-4: Vectorized Updates
```python
def _batch_update_weights(self, arc_indices):
    """Update weights for multiple arcs at once using NumPy."""
    if len(arc_indices) == 0:
        return
    
    # Get projections for all affected arcs (one basis solve with multiple RHS)
    projections = self._batch_project(arc_indices)
    
    # Vectorized weight computation
    new_weights = np.sum(projections ** 2, axis=1)
    new_weights = np.clip(new_weights, DEVEX_WEIGHT_MIN, DEVEX_WEIGHT_MAX)
    
    # Update weights array
    self.weights[arc_indices] = new_weights
```

#### Day 5: Testing
1. Verify convergence unchanged
2. Measure weight update time reduction

### Success Criteria
- [ ] Weight updates 50%+ faster
- [ ] Convergence rate unchanged
- [ ] All tests pass

### Files to Modify
- `src/network_solver/simplex_pricing.py` - Batch update logic

---

## Project 4: Vectorize Residual Calculations

**Impact:** 3% speedup (2s saved)  
**Difficulty:** ⭐ Easy  
**Timeline:** 2-3 days  
**Priority:** MEDIUM

### Problem
- `forward_residual` + `backward_residual` = 1.7s total
- Called 750,000 times each
- Trivial functions with high call overhead

### Solution
Pre-compute residuals as arrays, update on flow changes

### Implementation Plan

#### Day 1: Array-Based Residuals
```python
class NetworkSimplex:
    def __init__(self, ...):
        # Pre-compute residuals
        self._update_residuals()
    
    def _update_residuals(self):
        """Update residual arrays based on current flows."""
        self.forward_residuals = self.arc_uppers - self.arc_flows
        self.backward_residuals = self.arc_flows - self.arc_lowers
    
    def _pivot(self, arc_idx, direction):
        # ... update flows ...
        
        # Update residuals for affected arcs
        for idx, sign in cycle:
            self.arc_flows[idx] += sign * theta
        self._update_residuals()  # Or selective update
```

#### Day 2: Replace Calls
```python
# Old:
if arc.forward_residual() > 0:
    ...

# New:
if self.forward_residuals[arc_idx] > 0:
    ...
```

### Success Criteria
- [ ] No more `forward_residual()` / `backward_residual()` calls
- [ ] 750k function calls eliminated
- [ ] ~2s time saved

### Files to Modify
- `src/network_solver/simplex.py`

---

## Implementation Schedule

### Month 1
- **Week 1-3:** Project 1 (Cache Basis Solves) - 50% speedup
- **Week 4:** Project 2 (Vectorize Pricing) - Additional 10%

### Month 2
- **Week 5:** Project 3 (Batch Devex) - Additional 4%
- **Week 6:** Project 4 (Vectorize Residuals) - Additional 3%
- **Week 7-8:** Numba JIT (if needed)

### Expected Results

| Milestone | Runtime (Large) | Speedup | Cumulative |
|-----------|----------------|---------|------------|
| Baseline | 65.9s | 1.0x | - |
| After Project 1 | 32.9s | 2.0x | 2.0x |
| After Project 2 | 25.9s | 2.5x | 2.5x |
| After Project 3 | 22.9s | 2.9x | 2.9x |
| After Project 4 | 20.9s | 3.2x | 3.2x |

**Target:** **3x speedup in 2 months**

## Risk Mitigation

1. **Feature Flags:** All optimizations behind `SolverOptions` flags
2. **Regression Tests:** Track performance on each commit
3. **Validation:** Compare solutions with original implementation
4. **Incremental:** Each project independently testable

## Success Metrics

- [ ] Large network (4,267 arcs) solves in < 22s (3x speedup)
- [ ] Medium network (1,066 arcs) solves in < 2.5s (3x speedup)
- [ ] All 453 unit tests pass
- [ ] Iteration counts within 5% of baseline
- [ ] Memory overhead < 100MB

## Next Steps

1. **This week:** Start Project 1 (Cache Basis Solves)
2. **Set up benchmarking:** Add performance regression tests
3. **Document baseline:** Record current performance metrics
4. **Create feature branch:** `feature/cache-basis-solves`

**The data is clear: caching basis solves is the #1 priority.** 🎯
