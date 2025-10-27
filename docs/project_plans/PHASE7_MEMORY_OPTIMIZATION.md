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
- [ ] Baseline profiling (in progress)
- [ ] Bottleneck identification
- [ ] Optimizations implementation
- [ ] Performance validation
- [ ] Documentation

---

## Next Steps

1. **Wait for baseline profiling** to complete (~3-5 minutes)
2. **Analyze results** - identify top memory consumers
3. **Create optimization plan** based on actual findings
4. **Implement optimizations** incrementally
5. **Measure impact** of each change
6. **Document findings** for Phase 7 completion

---

## Notes

- Memory optimization is inherently exploratory
- Actual optimizations will depend on profiling results
- May discover unexpected bottlenecks
- Success = understanding + targeted improvements

---

**Status**: 🟡 In Progress - Profiling Phase  
**Branch**: `optimization/phase7-memory-optimization`  
**Next Milestone**: Baseline profiling results
