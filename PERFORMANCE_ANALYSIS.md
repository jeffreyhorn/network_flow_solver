# Performance Analysis - Small Benchmark Readiness

**Date**: 2025-10-25  
**Branch**: `solver/small-benchmark-readiness`

## Executive Summary

Our solver produces **correct solutions** but has severe **performance issues** that prevent it from solving most benchmark problems within reasonable time limits.

**Current State**:
- ✅ Correctness: All capacity constraints validated correctly
- ❌ Performance: Only 39% (7/18) of LEMON benchmark problems solve within 60 seconds
- ❌ Scalability: Cannot handle problems larger than ~500 nodes

## Benchmark Results

### Success Rate by Problem Family

| Family   | Size Range   | Success Rate | Notes |
|----------|--------------|--------------|-------|
| goto     | 256 nodes    | 2/4 (50%)    | Only _08 variants succeed |
| gridgen  | 257-507 nodes| 3/6 (50%)    | Better than goto/netgen |
| netgen   | 256 nodes    | 2/8 (25%)    | Worst performance |

### Performance Metrics

**Successful Solves** (7 problems):
- Solve time: 6.5 - 27.0 seconds
- Problem size: 256-507 nodes, 2048-4056 arcs
- Iterations: 652 - 1960
- Memory: 1.7 GB average (very high!)

**Key Metric: Milliseconds per Iteration**:
- goto: 6.9-7.6 ms/iter (best)
- gridgen: 10.0-21.7 ms/iter (moderate, degrades with size)
- netgen: 31.3-32.3 ms/iter (worst - 4-5x slower than goto!)

### Critical Observations

1. **Iteration Cost Varies by Problem Type**:
   - NETGEN problems take 4-5x longer per iteration than GOTO
   - This suggests structural differences (density, degeneracy) affect performance

2. **Problem Size Scaling is Poor**:
   - 256 nodes: 6.5-27 seconds (borderline acceptable)
   - 512 nodes: Most timeout
   - 4096 nodes: All timeout

3. **One Success at 507 Nodes**:
   - `gridgen_8_09a` (507 nodes, 1204 iters, 26 seconds) barely succeeded
   - Similar-sized `goto_8_09a` and `netgen_8_09a` (512 nodes) timed out
   - Suggests we're right at the edge of feasibility for this size

4. **Memory Usage is Excessive**:
   - 1.7 GB for 256-node problems is extremely high
   - Indicates inefficient data structures or memory leaks

## Root Cause Analysis

### Primary Issue: Too Many Iterations

The solver requires **too many iterations** to reach optimality:
- 256 nodes: 650-1960 iterations
- 507 nodes: 1204 iterations (and climbing)

**Expected** for well-tuned network simplex:
- Small problems (256 nodes): 50-200 iterations
- Medium problems (512 nodes): 100-400 iterations

We're taking **5-10x more iterations** than expected!

### Secondary Issue: Slow Iterations

Each iteration takes too long:
- Expected: 0.1-1 ms per iteration
- Actual: 7-32 ms per iteration (10-300x slower!)

### Tertiary Issue: Memory Usage

Memory consumption is way too high:
- Expected: 10-50 MB for 256-node problems
- Actual: 1700 MB (34x-170x higher!)

## Likely Causes

### Iteration Count Issues (Primary)

1. **Degeneracy Handling**:
   - Many degenerate pivots (zero flow change)
   - Anti-cycling mechanisms may be insufficient
   - Cost perturbation might not be effective

2. **Poor Initial Basis**:
   - Artificial arc Phase 1 may be slow
   - Initial basis quality affects Phase 2 convergence

3. **Pricing Strategy**:
   - Devex pricing might be selecting poor entering arcs
   - Block search or candidate list might help

### Iteration Speed Issues (Secondary)

1. **Basis Updates**:
   - Tree factorization updates may be slow
   - Refactorization frequency might be too low/high

2. **Pricing Overhead**:
   - Computing reduced costs for all arcs every iteration
   - Even with vectorization, this is expensive

3. **Python Overhead**:
   - Network simplex is iteration-intensive
   - Python's interpreted nature adds overhead

### Memory Issues (Tertiary)

1. **Data Structure Bloat**:
   - Multiple copies of arc data (ArcState, numpy arrays, etc.)
   - Excessive caching

2. **Memory Leaks**:
   - Growing data structures during pivots
   - Not releasing memory after refactorization

## Comparison to State-of-the-Art

Modern network simplex solvers (LEMON, CS2, etc.) can solve:
- 256-node problems: < 0.1 seconds
- 512-node problems: < 0.5 seconds  
- 4096-node problems: 1-5 seconds

We're **100-1000x slower** than state-of-the-art!

## Recommendations

### Immediate Actions (Must Fix)

1. **Profile the Code**:
   - Identify which parts of each iteration are slow
   - Use Python profiler on a small problem
   - Focus on the hot path

2. **Reduce Iteration Count**:
   - Investigate degeneracy: count degenerate pivots
   - Review cost perturbation effectiveness
   - Consider better Phase 1 initialization

3. **Optimize Critical Path**:
   - Focus on whatever profiling shows is slowest
   - Likely candidates: basis updates, pricing, reduced cost computation

### Medium-Term Improvements

1. **Better Pricing**:
   - Implement candidate list pricing
   - Add partial pricing (don't scan all arcs)
   - Consider block pricing

2. **Improved Initialization**:
   - Better artificial basis construction
   - Crash basis heuristics
   - Warm-start support

3. **Memory Optimization**:
   - Reduce data structure duplication
   - Use more compact representations
   - Profile memory usage

### Long-Term Considerations

1. **JIT Compilation**:
   - Use Numba for hot loops
   - Compile critical functions

2. **Algorithm Variants**:
   - Try different pivot rules
   - Experiment with primal vs dual simplex
   - Consider cost-scaling

3. **Benchmark Suite Expansion**:
   - Once we can solve small problems efficiently
   - Add medium (1K-10K nodes) and large (>10K) benchmarks

## Success Criteria

To be ready for small benchmark problems, we need:

**Minimum (MVP)**:
- ✅ Correctness: All solutions pass validation
- ❌ Small problems (256-512 nodes): < 5 seconds (currently 6-60+ seconds)
- ❌ Success rate: > 90% on _08 and _09 variants (currently 39% overall)

**Target**:
- Small problems (256-512 nodes): < 1 second (currently 6-60+ seconds)
- Medium problems (1K-2K nodes): < 10 seconds (currently timeout)
- Memory: < 100 MB for 512-node problems (currently 1.7 GB)

## Next Steps

1. **Profile a failing case**: Run gridgen_8_09a (the 26-second case) through Python profiler
2. **Count degenerate pivots**: Add instrumentation to measure degeneracy rate
3. **Focus on iteration count first**: Faster iterations won't help if we need 10x too many
4. **Document findings**: Create performance improvement roadmap based on profiling

---

**Status**: Analysis complete, ready for profiling and optimization work
