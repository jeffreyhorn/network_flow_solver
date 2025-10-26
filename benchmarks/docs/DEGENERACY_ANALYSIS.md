# Degeneracy Analysis - Critical Performance Bottleneck Identified

**Date**: 2025-10-25  
**Branch**: `performance/reduce-iteration-count`  
**Finding**: Extreme degeneracy (67-72%) is the #1 limiting factor for solver performance

## Executive Summary

Instrumented the solver to measure degeneracy (pivots where theta ≈ 0, making no progress). Results show **extreme degeneracy** on most benchmark problems:

- **GRIDGEN problems**: 67-72% degeneracy ❗
- **NETGEN problems**: 67-72% degeneracy ❗  
- **GOTO problems**: 20-28% degeneracy

**Impact**: On gridgen_8_09a, **867 out of 1204 pivots (72%) make zero progress**. Eliminating degeneracy could provide **3-4x speedup** on these problems.

**Conclusion**: Anti-degeneracy strategies should be the **top priority** for Phase 4.

---

## What is Degeneracy?

In the network simplex algorithm:
- **Normal pivot**: theta > 0, flows change, we make progress toward optimality
- **Degenerate pivot**: theta ≈ 0, basis changes but flows don't, no progress made

**Why it happens**: Multiple arcs at their bounds simultaneously, creating "ties" where we change basis without changing the solution.

**Why it's bad**: We're doing expensive pivot operations (cycle collection, basis updates, tree traversal) but making zero progress toward the optimal solution.

---

## Measurement Results

### Detailed Findings

| Problem | Nodes | Arcs | Iterations | Degenerate | Rate | Productive |
|---------|-------|------|------------|------------|------|------------|
| **gridgen_8_08a** | 256 | 2048 | 654 | 438 | **67.0%** | 216 |
| **gridgen_8_09a** | 512 | 4056 | 1204 | 867 | **72.0%** | 337 |
| **goto_8_08a** | 256 | 2043 | 1869 | 370 | 19.8% | 1499 |
| **goto_8_09a** | 512 | 4083 | 5513 | 1569 | 28.5% | 3944 |
| **netgen_8_08a** | 256 | 2040 | 777 | 526 | **67.7%** | 251 |

### Key Patterns

1. **Problem family matters**:
   - GRIDGEN: Extremely degenerate (67-72%)
   - NETGEN: Extremely degenerate (67-72%)
   - GOTO: Moderately degenerate (20-28%)

2. **Pricing strategy correlation**:
   - Devex pricing (GRIDGEN, NETGEN): Very high degeneracy
   - Dantzig pricing (GOTO, auto-detected): Lower degeneracy
   - Suggests pricing strategy affects degeneracy rate

3. **Scale of waste**:
   - gridgen_8_09a: Only 337 productive pivots out of 1204 (72% waste!)
   - netgen_8_08a: Only 251 productive pivots out of 777 (67% waste!)
   - This explains why iteration counts are so high

---

## Impact Analysis

### Speedup Potential

If we could eliminate degeneracy, theoretical speedup:

**gridgen_8_09a example**:
- Current: 1204 iterations
- Productive pivots: 337
- **Potential speedup if zero degeneracy**: 3.57x fewer iterations

**Realistic expectations**:
- Eliminating 100% of degeneracy is impossible
- **Reducing degeneracy by 50%** (72% → 36%):
  - gridgen_8_09a: 1204 → ~770 iterations (~1.56x faster)
  
- **Reducing degeneracy by 75%** (72% → 18%):
  - gridgen_8_09a: 1204 → ~550 iterations (~2.19x faster)

### Problems We Could Solve

With 2x iteration reduction:
- Many timeout problems take too long due to high iteration count
- goto_8_10a, goto_8_10b: Currently timeout, might solve with fewer iterations
- gridgen_8_10a: Might complete within 60s timeout

---

## Root Causes

### Why GRIDGEN/NETGEN are So Degenerate

1. **Grid structure**: Regular patterns create many ties
2. **Multiple optimal bases**: Many ways to achieve same flow
3. **Capacity patterns**: Many arcs at upper/lower bounds simultaneously

### Why GOTO is Less Degenerate

1. **Auto-detected as grid-on-torus**: Uses Dantzig pricing instead of Devex
2. **Different arc cost structure**: Fewer simultaneous boundary conditions
3. **Dantzig pricing**: Might inherently handle degeneracy better

---

## Anti-Degeneracy Strategies

### Option 1: Perturbation (Recommended, Low Effort)

**Idea**: Add tiny random perturbations to costs or bounds to break ties

**Approaches**:
- **Cost perturbation**: Add ε × random() to each arc cost
- **Lexicographic perturbation**: Use arc index as tiebreaker (more principled)
- **Bound perturbation**: Slightly adjust capacity bounds

**Pros**:
- Simple to implement
- Mathematically sound (solution converges as ε → 0)
- Proven effective in practice

**Cons**:
- Need to be careful with epsilon size
- May affect numerical stability

**Expected impact**: 30-50% degeneracy reduction

---

### Option 2: Better Tie-Breaking Rules

**Current**: When multiple arcs have same residual, we pick by index

**Improvements**:
- **Steepest edge tie-breaker**: Among tied arcs, pick the one that changes flow most
- **Dynamic tie-breaking**: Use problem-specific heuristics
- **Remember recent bases**: Avoid cycling back to previous bases

**Pros**:
- No numerical perturbation needed
- Can be very effective

**Cons**:
- More complex to implement
- May add per-pivot overhead

**Expected impact**: 20-40% degeneracy reduction

---

### Option 3: Degeneracy-Aware Pricing

**Idea**: Modify pricing strategy to avoid entering degenerate pivots

**Approaches**:
- **Look-ahead**: Check if entering arc will lead to degenerate pivot, skip it
- **Prefer non-degenerate**: When selecting entering arc, favor those unlikely to be degenerate
- **Adaptive block sizing**: Reduce block size in degenerate regions

**Pros**:
- Addresses root cause
- No perturbation needed

**Cons**:
- Complex to implement
- May miss optimal entering arcs

**Expected impact**: 40-60% degeneracy reduction (if done well)

---

### Option 4: Switch to Steepest-Edge Pricing

**Idea**: Use steepest-edge instead of Devex

**Rationale**:
- Steepest-edge makes better arc selections
- Might naturally avoid degenerate pivots
- Known to reduce iteration count in some problem classes

**Pros**:
- We might implement this anyway for better pricing
- Could reduce both degeneracy AND improve productive pivot quality

**Cons**:
- More expensive per iteration (need to compute actual edge weights)
- Might not address degeneracy directly
- Need to implement and test

**Expected impact**: Unknown, need to test

---

## Recommended Approach for Phase 4

### Stage 1: Quick Win - Cost Perturbation (1-2 days)

1. **Implement lexicographic perturbation**:
   ```python
   # Add small cost perturbation based on arc index
   perturbed_cost[i] = original_cost[i] + epsilon * (i / num_arcs)
   ```

2. **Test on GRIDGEN problems**:
   - Measure degeneracy reduction
   - Verify solution quality maintained
   - Benchmark performance improvement

3. **Expected outcome**: 30-50% degeneracy reduction, 1.2-1.4x speedup

---

### Stage 2: Better Tie-Breaking (2-3 days)

1. **Implement smarter leaving arc selection**:
   - When multiple arcs tied at minimum residual
   - Pick the one that maximizes expected progress
   - Consider arc costs, flow magnitudes

2. **Test across all problem families**

3. **Expected outcome**: Additional 10-20% degeneracy reduction

---

### Stage 3: Evaluate Steepest-Edge (3-5 days, optional)

1. **Implement steepest-edge pricing**
2. **Compare to Devex on degenerate problems**
3. **Measure trade-off**: iterations vs time-per-iteration

4. **Decision point**: Keep if overall speedup, otherwise discard

---

## Success Metrics

**Target for Phase 4**:
- Reduce average degeneracy from 67% to <40% on GRIDGEN/NETGEN
- Reduce iteration count by 30-50%
- Overall speedup: 1.3-1.5x on currently-solving problems
- New problems solved: 2-3 additional problems within timeout

**Validation**:
- Solution quality maintained (same objectives)
- No numerical instability
- All current problems still solve correctly

---

## Comparison to Other Solvers

**Note**: Commercial solvers like CPLEX, Gurobi handle degeneracy extensively:
- Multiple anti-degeneracy techniques
- Sophisticated perturbation schemes
- Degenerate pivots are a well-known issue

Our finding of 67-72% degeneracy is **not unusual** for network flow problems, but highlights the importance of handling it properly.

---

## Next Steps

1. ✅ **Measure degeneracy** - DONE
2. **Implement cost perturbation** - High priority, quick win
3. **Test on full benchmark suite** - Validate approach
4. **Iterate on tie-breaking** - If needed for further improvement
5. **Document results** - Show before/after degeneracy rates

---

## Conclusion

The degeneracy analysis reveals that **67-72% of pivots on GRIDGEN/NETGEN problems make zero progress**. This is the single biggest opportunity for performance improvement.

**Key findings**:
- Extreme degeneracy is the #1 bottleneck
- Potential for 1.5-3x iteration reduction
- Simple strategies (perturbation) could provide quick wins
- This explains why iteration counts are so high

**Recommendation**: Prioritize anti-degeneracy strategies for Phase 4. Even modest improvements (50% degeneracy reduction) would provide significant speedup and enable solving more benchmark problems.

This is a **higher-impact opportunity** than JIT compilation or other low-level optimizations, because it directly reduces the amount of work the solver needs to do.
