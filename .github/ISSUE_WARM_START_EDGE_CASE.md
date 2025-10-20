# Warm-Start Fails on Identical Problem Re-solve

## Description

When rebuilding the **exact same problem** (identical structure and parameters) and applying warm-start with its own basis, the solver becomes infeasible with objective value = 0.

## Expected Behavior

When warm-starting a problem with a basis extracted from that exact same problem's optimal solution, it should:
- Accept the basis as valid
- Skip or quickly complete Phase 1 (already feasible)
- Return the same optimal solution

## Actual Behavior

- Solver returns `status = "infeasible"`
- Objective value = 0
- Falls back to cold start, which then finds the correct solution

## Steps to Reproduce

```python
from network_solver import solve_min_cost_flow, Node, Arc, NetworkProblem

# Create a simple problem
problem1 = NetworkProblem(
    nodes=[
        Node("A", supply=10),
        Node("B", demand=10),
    ],
    arcs=[
        Arc("A", "B", cost=5, capacity=100),
    ]
)

# Solve it
result1 = solve_min_cost_flow(problem1)
print(f"First solve: {result1.status}, objective={result1.objective}")

# Rebuild the EXACT SAME problem
problem2 = NetworkProblem(
    nodes=[
        Node("A", supply=10),
        Node("B", demand=10),
    ],
    arcs=[
        Arc("A", "B", cost=5, capacity=100),
    ]
)

# Try to warm-start with basis from first solve
result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
print(f"Second solve: {result2.status}, objective={result2.objective}")
# Expected: optimal, 50.0
# Actual: infeasible, 0.0 (then falls back to cold start)
```

## Impact

- **Low severity** - Edge case that rarely occurs in practice
- Warm-start is designed for solving *similar* problems with parameter changes
- Re-solving the identical problem without changes is an uncommon use case
- Solver correctly falls back to cold start, so no incorrect results are produced

## Workarounds

1. **Use cold start** (default behavior) when re-solving identical problems
2. **Make a small parameter change** (e.g., slightly adjust a cost or supply) to differentiate the problems
3. **Warm-start works correctly** for similar problems with moderate changes to:
   - Supply/demand values
   - Arc costs
   - Arc capacities

## Technical Context

This issue was discovered during implementation of warm-start support. Extensive debugging was performed, including:

- Investigated flow recomputation logic in `_recompute_tree_flows()`
- Tested various approaches to handle artificial arc flows
- Attempted different basis validation strategies

The issue could not be resolved without breaking the 16 existing warm-start tests that correctly handle similar (but not identical) problems.

## Code References

- `src/network_solver/simplex.py:278` - `_apply_warm_start_basis()` method
- `src/network_solver/simplex.py:424` - `_recompute_tree_flows()` method
- `tests/unit/test_warm_start.py` - 16 passing tests for warm-start functionality

## Related Documentation

- TODO comment in `simplex.py:_apply_warm_start_basis()` 
- Known Limitations section in CHANGELOG.md
- Warm-start docstring mentions this edge case

## Priority

**Low** - This is a known limitation that:
- Has clear workarounds
- Doesn't affect the primary use case (solving similar problems)
- Doesn't produce incorrect results (falls back gracefully)
- All 16 warm-start tests pass for the intended use cases

## Suggested Investigation

Future work could investigate:
1. Why flow recomputation fails for identical problems
2. Whether the issue is in basis validation, tree construction, or flow initialization
3. If a minimal reproducing case can isolate the root cause
4. Whether there's a way to detect this case and skip warm-start automatically
