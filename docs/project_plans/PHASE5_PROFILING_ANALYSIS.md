# Phase 5: JIT Compilation - Profiling Analysis

**Date**: 2025-10-26  
**Problem**: GRIDGEN 8_12a (4097 nodes, 32776 arcs)  
**Total Time**: 560.8 seconds  
**Iterations**: 8941  
**Degeneracy**: 86.2%

---

## Executive Summary

Profiling reveals **three critical bottlenecks** consuming 51% of total runtime:

1. **`basis.rebuild()`**: 117.2s (21% of total time) - 8855 calls
2. **`simplex._update_tree_sets()`**: 110.5s (20% of total time) - 8852 calls  
3. **`basis.collect_cycle()`**: 55.3s (10% of total time) - 8941 calls

These three functions alone account for **282.9s out of 560.8s** (50.4%).

**Key insight**: The bottleneck is NOT in pricing (as originally hypothesized), but in **tree operations and basis maintenance**. This makes sense given the 86.2% degeneracy rate - most pivots are degenerate, so we're doing tree updates without meaningful progress.

---

## Top Functions by Total Time

| Rank | Function | Time (s) | % Total | Calls | Time/Call (ms) | Priority |
|------|----------|----------|---------|-------|----------------|----------|
| 1 | `basis.rebuild()` | 117.2 | 20.9% | 8855 | 13.2 | **P1** |
| 2 | `simplex._update_tree_sets()` | 110.5 | 19.7% | 8852 | 12.5 | **P1** |
| 3 | `numpy.ndarray.nonzero()` | 58.8 | 10.5% | 278 | 211.6 | Native |
| 4 | `basis.collect_cycle()` | 55.3 | 9.9% | 8941 | 6.2 | **P1** |
| 5 | `list.append()` | 28.5 | 5.1% | 72.9M | 0.0004 | Native |
| 6 | `deque.append()` | 24.1 | 4.3% | 61.0M | 0.0004 | Native |
| 7 | `deque.popleft()` | 21.8 | 3.9% | 58.5M | 0.0004 | Native |
| 8 | `numpy.array()` | 17.0 | 3.0% | 20343 | 0.8 | Native |
| 9 | `set.add()` | 15.7 | 2.8% | 36.4M | 0.0004 | Native |
| 10 | `simplex.solve()` | 13.8 | 2.5% | 1 | 13826 | Top-level |
| 11 | `numpy.linalg.norm()` | 13.0 | 2.3% | 17609 | 0.7 | Native |
| 12 | `SuperLU.solve()` | 11.0 | 2.0% | 19604 | 0.6 | Native |
| 13 | `basis._build_numeric_basis()` | 10.8 | 1.9% | 139 | 77.5 | **P2** |
| 14 | `forrest_tomlin.solve()` | 8.7 | 1.6% | 19604 | 0.4 | **P2** |
| 15 | `simplex._compute_reduced_costs_vectorized()` | 8.1 | 1.4% | 8942 | 0.9 | **P2** |

**Total time in top 3 targetable functions**: 282.9s (50.4%)

---

## Detailed Analysis

### Priority 1: Tree Operations (50.4% of total time)

#### 1.1 `basis.rebuild()` - 117.2s (20.9%)

**What it does**: Rebuilds the tree structure after basis changes.

**Called**: 8855 times (almost every pivot)

**Performance**:
- 13.2ms per call
- Cumulative: 259.1s (includes subcalls)
- Total: 117.2s (excluding subcalls)

**Why it's slow**: 
- Python loops over tree nodes
- Frequent set/dict operations (not JIT-friendly as-is)
- Tree traversal with Python overhead

**JIT potential**: Medium-High
- Core loops can be extracted
- Tree traversal is algorithmic (not data structure dependent)
- May need to refactor to use NumPy arrays instead of sets/dicts

**Action**: Read `basis.py:68` to understand implementation, extract JIT-able kernels

---

#### 1.2 `simplex._update_tree_sets()` - 110.5s (19.7%)

**What it does**: Updates tree node sets (depths, parents, children) after pivot.

**Called**: 8852 times (every pivot)

**Performance**:
- 12.5ms per call
- Cumulative: 138.8s (includes subcalls)
- Total: 110.5s (excluding subcalls)

**Why it's slow**:
- Set operations (add, remove) - 36.4M `set.add()` calls visible in profile
- Deque operations (append, popleft) - 119M deque operations visible
- Python overhead for BFS/DFS traversal

**JIT potential**: High
- Tree traversal with NumPy arrays = perfect for Numba
- If we use integer node IDs and array-based tree structure

**Action**: Read `simplex.py:1055` to understand implementation, refactor to array-based

---

#### 1.3 `basis.collect_cycle()` - 55.3s (9.9%)

**What it does**: Finds the cycle when adding entering arc to tree.

**Called**: 8941 times (every pivot)

**Performance**:
- 6.2ms per call
- Cumulative: 72.1s (includes subcalls)
- Total: 55.3s (excluding subcalls)

**Why it's slow**:
- Tree path traversal (Python loops)
- List append operations (73M visible in profile)

**JIT potential**: Very High
- Classic algorithmic problem (find path in tree)
- Can be implemented with NumPy arrays
- No complex data structures needed

**Action**: Read `basis.py:113` to understand implementation, write JIT version

---

### Priority 2: Pricing and Basis Operations (11.3%)

#### 2.1 `basis._build_numeric_basis()` - 10.8s (1.9%)

**What it does**: Builds numerical basis matrix for LU factorization.

**Called**: 139 times (refactorizations)

**Performance**:
- 77.5ms per call
- Cumulative: 97.0s (includes subcalls to scipy.sparse)

**Why it's slow**:
- Building sparse matrix (COO format)
- Includes 59s in `numpy.ndarray.nonzero()` (scipy overhead)

**JIT potential**: Low-Medium
- Sparse matrix construction is scipy-dependent
- Could optimize the setup, but scipy does the heavy lifting

**Action**: Lower priority, focus on P1 first

---

#### 2.2 `simplex._compute_reduced_costs_vectorized()` - 8.1s (1.4%)

**What it does**: Computes reduced costs for all non-basic arcs.

**Called**: 8942 times (every pivot)

**Performance**:
- 0.9ms per call
- Already vectorized with NumPy

**Why it's slow**:
- Still has Python overhead for the function call
- NumPy operations could be JIT-compiled

**JIT potential**: Medium
- Simple vectorized operation
- Easy to JIT with Numba

**Action**: Quick win after P1 optimizations

---

#### 2.3 `forrest_tomlin.solve()` - 8.7s (1.6%)

**What it does**: Solves linear system using Forrest-Tomlin updates.

**Called**: 19604 times

**Performance**:
- 0.4ms per call
- Calls SuperLU.solve() (11s - native code, can't optimize)

**JIT potential**: Low
- Most time is in native SuperLU
- Wrapper overhead is minimal

**Action**: Skip, already using native code

---

## Optimization Strategy

### Phase 5A: Priority 1 Tree Operations (Week 1-2)

**Target**: 50% of total runtime (283s → ~140s with 2x speedup)

1. **`collect_cycle()`** - Start here (highest JIT potential)
   - Extract to pure NumPy/Numba implementation
   - Use parent array for tree traversal
   - Expected: 2-3x speedup → save 35-40s

2. **`_update_tree_sets()`** - Biggest time consumer
   - Refactor from sets to NumPy arrays
   - JIT-compile BFS/DFS traversal
   - Expected: 2-4x speedup → save 55-80s

3. **`rebuild()`** - Complex but high impact
   - Identify JIT-able kernels within rebuild
   - May require partial refactoring
   - Expected: 1.5-2x speedup → save 30-60s

**Total expected savings**: 120-180s (21-32% of total runtime)  
**New total time**: 380-440s (1.3-1.5x overall speedup)

---

### Phase 5B: Priority 2 Quick Wins (Week 3)

**Target**: 10% of total runtime (56s → ~35s with 1.6x speedup)

1. **`_compute_reduced_costs_vectorized()`**
   - Simple JIT compilation
   - Expected: 2x speedup → save 4s

2. **`_select_entering_arc_vectorized()`**
   - JIT-compile arc selection loop
   - Expected: 2x speedup → save 2-3s

3. **Other pricing functions**
   - Quick JIT passes
   - Expected: save 5-10s total

**Total expected savings**: 15-20s (3-4% of total runtime)  
**New total time**: 360-425s (1.3-1.6x overall speedup)

---

### Combined Expected Impact

**Conservative estimate**:
- Phase 5A: 1.3x speedup
- Phase 5B: 1.1x additional
- **Total: 1.4x overall speedup** (560s → 400s)

**Target estimate**:
- Phase 5A: 1.4x speedup
- Phase 5B: 1.2x additional
- **Total: 1.7x overall speedup** (560s → 330s)

**Optimistic estimate**:
- Phase 5A: 1.5x speedup
- Phase 5B: 1.3x additional
- **Total: 2.0x overall speedup** (560s → 280s)

---

## Key Findings vs Plan

**Original Plan Hypothesis**:
- Pricing operations: 30-40% of time
- Theta computation: 15-20% of time
- Forrest-Tomlin: 10-15% of time

**Actual Results**:
- Tree operations: **50% of time** (not in original plan!)
- Pricing operations: ~6% of time (much less than expected)
- Forrest-Tomlin: ~2% of time (native code, can't optimize)

**Why the difference**:
- High degeneracy (86%) → many pivots with minimal progress
- Each pivot requires tree update, regardless of degeneracy
- Pricing is already well-optimized (vectorized)
- Tree operations use Python data structures (sets, dicts, deques)

**Strategic pivot**:
Focus on tree operations first, not pricing. This aligns better with the actual bottlenecks.

---

## Next Steps

1. **Read and understand tree operation implementations**:
   - `basis.py:113` (`collect_cycle`)
   - `simplex.py:1055` (`_update_tree_sets`)
   - `basis.py:68` (`rebuild`)

2. **Design array-based tree structure**:
   - Replace sets/dicts with NumPy arrays
   - Use integer indices for nodes
   - Enable Numba JIT compilation

3. **Implement JIT versions**:
   - Start with `collect_cycle()` (cleanest, highest JIT potential)
   - Move to `_update_tree_sets()` (biggest impact)
   - Finally tackle `rebuild()` (most complex)

4. **Validate correctness**:
   - Unit tests for each function
   - Full benchmark suite
   - Compare with non-JIT versions

5. **Measure impact**:
   - Profile again after each optimization
   - Track cumulative speedup
   - Adjust strategy based on results

---

## Success Criteria

- **Minimum**: 1.4x overall speedup (560s → 400s)
- **Target**: 1.7x overall speedup (560s → 330s)
- **Stretch**: 2.0x overall speedup (560s → 280s)
- **Correctness**: 100% agreement with non-JIT version on all 18 benchmark problems
- **Code quality**: Maintain readability, add JIT fallback, all tests pass

---

**Status**: Analysis complete, ready for implementation  
**Next**: Read `basis.collect_cycle()` and implement JIT version
