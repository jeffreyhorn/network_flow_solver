# Known Issues and Limitations

This document tracks known issues and limitations with the benchmark suite and solver performance.

## Solver Performance Issues

### GOTO (Grid-on-Torus) Instances

**Status**: Known limitation - Does not converge in reasonable time

**Affected Instances**:
- `goto_8_08a.min` (256 nodes, 2048 arcs)
- `goto_8_08b.min` (256 nodes, 2048 arcs)
- `goto_8_09a.min` (512 nodes, 4096 arcs)

**Symptoms**:
- Instances timeout after 300+ seconds
- Solver does not find feasible solution within iteration limit
- Even with increased iteration limit (20x arcs = ~40,960 iterations), solver does not converge

**Root Cause**:
Grid-on-torus networks have unique topological properties (wrapped grid with no boundaries) that appear to be challenging for the current network simplex implementation. Possible contributing factors:
- Poor initial basis selection for toroidal topology
- Cycling or slow convergence in Phase 1 (finding initial feasible solution)
- May benefit from specialized pivot selection strategies

**Workaround**:
- Skip GOTO instances when benchmarking: Use size filters in benchmark runner
- These instances are not representative of most real-world network flow problems
- Focus on NETGEN and GRIDGEN instances which converge successfully

**Future Work** (Phase 6+):
- Investigate initial basis generation for toroidal networks
- Consider anti-cycling strategies (perturbation methods, Bland's rule)
- Profile solver execution to identify bottleneck
- May require algorithmic improvements beyond parameter tuning

## Instance-Specific Issues

### netgen_8_11a.min

**Status**: Timeout with current settings

**Details**:
- Instance: 512 nodes, 4096 arcs
- Timeout: >300 seconds
- Larger NETGEN instance that requires more iterations than smaller variants
- May benefit from longer timeout or continued solver optimizations

**Workaround**:
- Increase timeout: `--timeout 600` or higher
- Focus benchmarking on instances 08-10 which complete successfully

## Benchmark Results Summary

### Working Instances (7/11 small instances)

**GRIDGEN instances** - All successful:
- `gridgen_8_08a.min`: ✓ ~26s, 654 iterations
- `gridgen_8_08b.min`: ✓ ~27s, 652 iterations  
- `gridgen_8_09a.min`: ✓ ~90s, 1204 iterations

**NETGEN instances** - Mostly successful:
- `netgen_8_08a.min`: ✓ ~15s, 777 iterations
- `netgen_8_08b.min`: ✓ ~15s, 836 iterations
- `netgen_8_09a.min`: ✓ ~88s, 1403 iterations
- `netgen_8_10a.min`: ✓ ~179s, 2657 iterations (hits iteration limit but finds optimal)
- `netgen_8_11a.min`: ⏱ Timeout (>300s)

### Problematic Instances (4/11 small instances)

**GOTO instances** - All timeout:
- `goto_8_08a.min`: ⏱ Timeout
- `goto_8_08b.min`: ⏱ Timeout
- `goto_8_09a.min`: ⏱ Timeout

**Large NETGEN**:
- `netgen_8_11a.min`: ⏱ Timeout

## Recommendations

### For Benchmarking

1. **Focus on working instances**: Use GRIDGEN and NETGEN instances (08-10 range)
2. **Skip GOTO instances**: These don't converge and skew results
3. **Increase timeout for larger instances**: Consider 600s default for comprehensive testing

### For Development

1. **Investigate GOTO topology**: Understand why toroidal networks cause issues
2. **Profile slow instances**: Identify computational bottlenecks
3. **Consider algorithmic improvements**:
   - Better initial basis heuristics
   - Improved pivot selection strategies
   - Anti-cycling mechanisms

### For Users

- When using the solver on real problems with toroidal/wrapped topologies, be aware of potential convergence issues
- Consider alternative formulations or preprocessing steps for wrapped grid networks
- Monitor iteration counts and time limits

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 5 - Medium Instances & Benchmark Runner  
**Benchmark Results**: See `benchmarks/results/after-iteration-fix.json`
