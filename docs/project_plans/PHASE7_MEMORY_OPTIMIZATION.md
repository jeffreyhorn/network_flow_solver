# Phase 7: Memory Optimization

**Goal**: Understand and reduce memory usage, improve cache efficiency  
**Expected Impact**: 1.1-1.3x speedup (indirect, via cache efficiency)  
**Status**: In Progress  
**Date**: 2025-10-27

---

## Executive Summary

Phase 7 focuses on understanding and optimizing memory usage in the network simplex solver. Previous observations noted unexplained memory usage on medium-sized problems. This phase will profile memory usage, identify bottlenecks, and implement optimizations that improve both memory footprint and performance through better cache efficiency.

**Key Insight**: Smaller memory footprint → better cache utilization → faster execution.

---

## Background

### Motivation

From the `GETTING_TO_50X_PLAN.md`:
- Observed memory usage: 1.7 GB for 256-node problems
- Expected memory: ~100-200 MB
- Excess: ~1.5 GB unaccounted

### Why Memory Optimization Matters

1. **Cache Efficiency**: Smaller working set → better L1/L2/L3 cache hit rates
2. **Larger Problems**: Reduced memory enables solving bigger instances
3. **Reduced GC Pressure**: Fewer allocations → less garbage collection overhead
4. **Better Locality**: Related data closer together in memory

### Expected Performance Benefits

While primarily a memory optimization, we expect indirect performance improvements:
- **Minimum**: Identify source of excess memory
- **Target**: 1.1-1.3x speedup from cache effects
- **Stretch**: Enable solving larger problems that previously caused memory issues

---

## Approach

### Phase 7A: Memory Profiling (Week 1)

**Objective**: Understand current memory usage and identify bottlenecks.

**Tools**:
1. **tracemalloc** (Python built-in): Line-by-line allocation tracking
2. **memory_profiler** (optional): Detailed function-level profiling
3. **sys.getsizeof**: Size of specific objects

**Created Infrastructure**:
- `profile_memory.py`: Memory profiling script
  - Tracks memory before/after solve
  - Reports peak memory usage
  - Shows top allocations by line and file
  - Formatted output for analysis

**Profiling Targets**:
- Small problem: gridgen_8_12a.min (4K nodes, 32K arcs) - baseline
- Medium problem: gridgen_16_24a.min (16K nodes, if available)
- Large problem: gridgen_32_48a.min (65K nodes, if available)

**Expected Findings**:
- NumPy array allocations (flows, costs, potentials, etc.)
- Basis matrix and LU factorization storage
- Tree structure data (adjacency lists, etc.)
- Python object overhead
- Temporary allocations not being freed

### Phase 7B: High-Impact Optimizations (Week 2)

**Based on profiling results**, target optimizations with >10% memory reduction:

**Potential Optimizations**:

1. **Eliminate Unnecessary Array Copies**:
   - Use in-place operations where possible (`+=` instead of `+`)
   - Pre-allocate arrays and reuse buffers
   - Explicitly delete temporary arrays

2. **Optimize Data Structures**:
   - Use `__slots__` for classes to reduce Python object overhead
   - Consider using NumPy structured arrays for arc/node data
   - Use views instead of copies where safe

3. **Clear Temporary Data**:
   - Explicitly free memory after Phase 1 (feasibility)
   - Clear old eta matrices after refactorization
   - Remove debug/diagnostic data that's not needed

4. **Sparse Data Structures**:
   - Use sparse matrices for basis storage (if not already)
   - Sparse representation for tree adjacency

### Phase 7C: Validation and Measurement (Week 3)

**For each optimization**:
1. Measure memory before/after
2. Run full test suite (ensure no regressions)
3. Benchmark performance impact
4. Document findings

**Success Criteria**:
- All 576 tests pass
- No performance regression
- Measurable memory reduction
- Cache efficiency improvement (if measurable)

### Phase 7D: Documentation (Week 4)

**Deliverables**:
- Memory profiling guide for users
- Documentation of optimizations implemented
- Performance impact analysis
- Recommendations for memory-constrained environments

---

## Initial Setup

### Profiling Script Created

```bash
# Profile memory usage on a problem
python profile_memory.py benchmarks/problems/lemon/gridgen/gridgen_8_12a.min
```

**Output includes**:
- Memory before/after solve
- Peak memory usage
- Top 20 memory allocations (by line)
- Top 15 memory allocations (by file)
- Current memory snapshot

---

## Baseline Measurements

**Testing on**: `gridgen_8_12a.min` (4097 nodes, 32776 arcs)

*Results pending - profiling currently running*

Expected metrics:
```
Problem: 4097 nodes, 32776 arcs
Memory before solve: XXX MB
Memory after solve: XXX MB
Peak memory: XXX MB
Memory increase: XXX MB

Top allocations:
  [Will show line-by-line breakdown]
```

---

## Hypotheses to Test

From the `GETTING_TO_50X_PLAN.md`, we'll investigate:

### Hypothesis 1: Basis Matrix Storage
- Full NxN dense matrices for basis
- LU factors
- Eta matrices accumulation

### Hypothesis 2: Array Copies
- NumPy operations creating hidden copies
- Temporary arrays not being freed

### Hypothesis 3: Convergence Monitoring
- Full iteration history storage
- Diagnostic data accumulation

### Hypothesis 4: Python Object Overhead
- Object wrappers for NumPy arrays
- Class instances with large `__dict__`

---

## Implementation Strategy

### Priority Order

1. **Profile First** (Week 1)
   - Understand actual memory usage
   - Don't guess - measure!

2. **High Impact First** (Week 2)
   - Focus on >10% reductions
   - Low-hanging fruit first

3. **Validate Thoroughly** (Week 3)
   - Test after each change
   - Benchmark performance
   - Ensure no regressions

4. **Document Everything** (Week 4)
   - What was found
   - What was optimized
   - Impact measurements

### Risk Mitigation

- **Correctness First**: All tests must pass after each change
- **Performance Monitoring**: Benchmark before/after each optimization
- **Incremental Changes**: One optimization at a time
- **Easy Rollback**: Git commit after each successful change

---

## Success Metrics

### Memory Reduction

- **Minimum**: Identify source of excess memory
- **Target**: 25% reduction in peak memory
- **Stretch**: 50% reduction in peak memory

### Performance Impact

- **Minimum**: No performance regression
- **Target**: 1.1x speedup from cache effects
- **Stretch**: 1.3x speedup from cache effects

### Problem Size

- **Minimum**: Document current limits
- **Target**: Enable 2x larger problems
- **Stretch**: Enable 4x larger problems

---

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Memory profiling | Profiling results, bottleneck identification |
| 2 | High-impact optimizations | Implementation, memory measurements |
| 3 | Validation & benchmarking | Test results, performance analysis |
| 4 | Documentation | Complete Phase 7 documentation |

**Estimated Duration**: 2-3 weeks (can compress if findings are clear)

---

## Current Status

- [x] Phase 7 branch created
- [x] Memory profiling script created
- [x] Baseline profiling completed
- [x] Bottleneck identification completed
- [x] Detailed analysis document created (`PHASE7_PROFILING_ANALYSIS.md`)
- [ ] Phase 7.1: Sparse basis matrix implementation
- [ ] Performance validation
- [ ] Documentation

---

## Profiling Results Summary

**Problem**: `gridgen_8_12a.min` (4,097 nodes, 32,776 arcs)

### Key Metrics
- **Peak Memory**: 1.02 GB ⚠️
- **Residual Memory**: 17 MB
- **Iterations**: 8,903
- **Status**: Optimal (783,027,844.0)

### Critical Finding: Dense Basis Matrix

The **1.02 GB peak** is dominated by dense basis matrix storage:
- Each basis matrix: `(n-1)² × 8 bytes = 134 MB` for n=4,097
- Basis is 99.9% sparse (only 2 nonzeros per column)
- **Wasting 133+ MB per matrix**
- With LU factors and temporaries: **400-800 MB per rebuild**

### Top Memory Consumers (Our Code)
1. Line 1713 (`simplex.py`): 1.28 MB - flows dict
2. Line 1728 (`simplex.py`): 197 KB - duals dict  
3. Line 1037 (`simplex.py`): 144 KB - tree_arcs set
4. Line 426 (`simplex.py`): 246 KB - arc objects
5. **Total from simplex.py**: 2.3 MB (only 0.2% of peak!)

**Conclusion**: The problem is NOT in simplex.py - it's in **basis matrix storage**.

See `PHASE7_PROFILING_ANALYSIS.md` for complete analysis.

---

## Implementation Plan

Based on profiling results, we've identified three high-impact optimizations:

### Phase 7.1: Sparse Basis Matrix ⭐ HIGH PRIORITY
**Expected Impact**: 80% memory reduction (1.02 GB → 200 MB)

**Problem**: Dense storage wastes 99.9% of space
```python
# Current (basis.py:254)
matrix = np.zeros((n, n), dtype=float)  # 134 MB for n=4,097
```

**Solution**: Use scipy.sparse CSC/CSR format
```python
from scipy.sparse import csc_matrix
# Only store nonzeros: 267 KB for n=4,097
```

**Implementation**:
1. Update `basis.py` to build sparse matrices
2. Update `basis_lu.py` to use sparse LU factorization
3. Verify all basis operations work with sparse
4. Run full test suite

**Risk**: Medium (need to ensure numerical stability)

### Phase 7.2: In-Place Basis Updates
**Expected Impact**: 40% additional reduction (200 MB → 120 MB)

**Problem**: Forrest-Tomlin updates may create copies

**Solution**: Ensure all updates modify matrices in-place
- Audit basis update code for `.copy()` calls
- Use in-place operations where possible

**Risk**: Low

### Phase 7.3: Reuse Temporary Arrays
**Expected Impact**: 20% additional reduction (120 MB → 100 MB)

**Problem**: Temporary arrays allocated per pivot

**Solution**: Pre-allocate and reuse buffers
```python
class TreeBasis:
    def __init__(self):
        self._temp_vector = np.zeros(n)  # Reuse this
```

**Risk**: Very low

---

## Success Criteria

### Memory Targets (gridgen_8_12a.min)

| Phase | Peak Memory | Reduction | Status |
|-------|-------------|-----------|--------|
| Baseline | 1.02 GB | - | ✅ Measured |
| After 7.1 | ~200 MB | 80% | 🎯 Target |
| After 7.2 | ~120 MB | 88% | 🎯 Target |
| After 7.3 | ~100 MB | 90% | 🎯 Target |

### Correctness
- All 576 tests must pass
- Identical objective values (±1e-6)
- No performance regression

---

## Next Steps

1. **Implement Phase 7.1** - Sparse basis matrix
2. **Run memory profiling** - Measure improvement
3. **Validate tests** - Ensure correctness
4. **Benchmark performance** - Check for speedup
5. **Implement Phase 7.2** - In-place updates
6. **Continue iteratively**

---

## Notes

- Dense basis matrix is the smoking gun (80% of memory)
- Network simplex basis is naturally sparse (2 nonzeros per column)
- Using sparse storage is the obvious optimization
- Should also improve performance (sparse operations faster)

---

**Status**: 🟢 Ready to Implement - Analysis Complete  
**Branch**: `optimization/phase7-memory-optimization`  
**Next Milestone**: Phase 7.1 sparse basis implementation
