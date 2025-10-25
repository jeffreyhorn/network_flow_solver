# Performance Profiling Analysis

## Executive Summary

Profiling reveals clear performance bottlenecks in the large problem scenario (50x50 transportation, 7.76s total):

**Top 5 Hot Paths (by total time):**
1. **forrest_tomlin.py:46(solve)** - 3.324s (42.8% of runtime)
2. **SuperLU.solve** - 1.165s (15.0% of runtime)
3. **_find_entering_arc_devex** - 0.705s (9.1% of runtime)
4. **_update_devex_weight** - 0.354s (4.6% of runtime)
5. **forward_residual** - 0.215s (2.8% of runtime)

**Combined: 74.3% of total runtime is in these 5 functions.**

## Detailed Analysis by Category

### 1. Linear Algebra (58% of runtime)

**Hot Paths:**
- `forrest_tomlin.py:46(solve)`: 3.324s (20,729 calls)
- `SuperLU.solve`: 1.165s (20,729 calls)
- Total: 4.489s

**Analysis:**
- Called ~20,729 times during solve (253 iterations)
- Average: 0.16ms per call for forrest_tomlin, 0.056ms for SuperLU
- This is the Forrest-Tomlin basis update system solving Bx=b

**Optimization Opportunities:**
- ✅ Already using scipy's sparse LU (SuperLU) - highly optimized C code
- ❌ Numba won't help here - already in optimized C/Fortran
- ✅ Could reduce call frequency by:
  - Caching basis inverse calculations
  - Using rank-1 update formulas more aggressively
  - Batching multiple linear solves if possible
- ✅ Consider BLAS vectorization for the Forrest-Tomlin update operations

### 2. Pricing (15% of runtime)

**Hot Paths:**
- `_find_entering_arc_devex`: 0.705s (254 calls)
- `_update_devex_weight`: 0.354s (20,225 calls)
- `project_column`: 0.069s (20,478 calls)
- Total: 1.128s

**Analysis:**
- Devex pricing called once per iteration (254 times)
- Weight updates called ~80 times per pricing operation
- Average: 2.78ms per pricing operation

**Optimization Opportunities:**
- ✅ **HIGH IMPACT**: Vectorize devex weight calculations with NumPy
- ✅ **HIGH IMPACT**: Use Numba JIT for the pricing loop
- ✅ Precompute and cache reduced costs where possible
- ✅ Use NumPy array operations instead of Python loops

### 3. Residual Calculations (5% of runtime)

**Hot Paths:**
- `forward_residual`: 0.215s (188,583 calls)
- `backward_residual`: 0.108s (188,916 calls)
- Total: 0.323s

**Analysis:**
- Very high call frequency (~188k calls each)
- Simple calculations but called in tight loops
- Average: 1.14µs per call (very fast, but high volume)

**Optimization Opportunities:**
- ✅ **HIGH IMPACT**: Vectorize with NumPy arrays
- ✅ **HIGH IMPACT**: Apply Numba @jit decorator
- ✅ Batch calculations instead of per-arc calls

### 4. Graph Operations (4% of runtime)

**Hot Paths:**
- `_update_tree_sets`: 0.148s (253 calls)
- `collect_cycle`: 0.078s (253 calls)
- Total: 0.226s

**Analysis:**
- Called once per pivot operation
- Tree structure updates and cycle detection

**Optimization Opportunities:**
- ✅ Use sets more efficiently
- ✅ Consider NumPy boolean arrays for tree membership
- ❌ Hard to vectorize graph algorithms

### 5. Generator Expressions (2% of runtime)

**Hot Path:**
- `simplex.py:798(<genexpr>)`: 0.158s (366,702 calls)

**Optimization Opportunities:**
- ✅ Replace generator expressions with list comprehensions if materializing
- ✅ Use NumPy operations if applicable

## Optimization Priority

### Priority 1: High Impact, Low Risk (Vectorization)

1. **Vectorize residual calculations** (`forward_residual`, `backward_residual`)
   - Current: Per-arc method calls
   - Target: NumPy array operations
   - Expected speedup: 5-10x
   - Lines to modify: simplex.py:40-50

2. **Vectorize devex weight updates** (`_update_devex_weight`)
   - Current: Loop with individual calculations
   - Target: NumPy vector operations
   - Expected speedup: 10-20x
   - Lines to modify: simplex.py:635-648

3. **Optimize pricing loops** (`_find_entering_arc_devex`)
   - Current: Python loops over arcs
   - Target: NumPy masked arrays / vectorized operations
   - Expected speedup: 3-5x
   - Lines to modify: simplex.py:657-700

### Priority 2: Medium Impact, Medium Risk (Numba JIT)

4. **JIT compile hot loops with Numba**
   - Candidates:
     - `_find_entering_arc_devex` (after vectorization)
     - `_update_devex_weight` (after vectorization)
     - `forward_residual` / `backward_residual`
   - Expected speedup: 2-5x additional
   - Risk: Compatibility with scipy/numpy objects

5. **JIT compile Forrest-Tomlin operations**
   - Target: `forrest_tomlin.py:solve` and `update`
   - Expected speedup: 2-3x
   - Risk: Integration with SuperLU objects

### Priority 3: Algorithmic Improvements (Higher Risk)

6. **Reduce basis solve frequency**
   - Cache repeated calculations
   - Use incremental updates where possible
   - Expected speedup: 10-20% reduction in calls

7. **Batch operations**
   - Process multiple arcs/calculations together
   - Expected speedup: Variable, 5-15%

## Implementation Plan

### Phase 1: Vectorization (Safest, Highest Impact)
- [ ] Create NumPy-based residual calculation arrays
- [ ] Vectorize devex weight updates
- [ ] Vectorize pricing arc evaluation
- [ ] Benchmark improvements
- [ ] Run full test suite

### Phase 2: Numba JIT (Medium Risk)
- [ ] Add Numba dependency
- [ ] JIT compile vectorized functions
- [ ] Test compatibility with existing code
- [ ] Benchmark improvements
- [ ] Run full test suite

### Phase 3: Advanced Optimizations
- [ ] Profile again to find new bottlenecks
- [ ] Consider caching strategies
- [ ] Explore algorithmic improvements

## Expected Overall Speedup

**Conservative estimate:**
- Vectorization: 2-3x speedup on large problems
- Numba JIT: Additional 1.5-2x speedup
- Combined: 3-6x total speedup

**Optimistic estimate:**
- Vectorization: 4-5x speedup
- Numba JIT: Additional 2-3x speedup
- Combined: 8-15x total speedup

## Baseline Metrics

- Small (5x5, 25 arcs): 0.047s
- Medium (20x20, 400 arcs): 0.594s
- Large (50x50, 2500 arcs): 7.758s
- Network (complex structure): 0.853s

**Target after optimization:**
- Large problem: < 2s (4x speedup minimum)
