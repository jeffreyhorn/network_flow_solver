# Phase 1 Terminates with Invalid Solution (Flow Conservation Violated)

## Summary
Phase 1 of the network simplex algorithm can terminate prematurely, returning a solution that violates flow conservation constraints despite having zero artificial arc flow. The solver incorrectly reports the solution as "optimal" when it is actually infeasible or incomplete.

## Issue Type
- [x] Bug
- [ ] Enhancement
- [ ] Question

## Severity
- [x] Critical - Solver returns incorrect results
- [ ] Major
- [ ] Minor

## Environment
- **Branch:** `fix/warm-start-infeasibility-detection` (but issue exists on `main` as well)
- **Python Version:** 3.12
- **Affected Component:** `src/network_solver/simplex.py` - Phase 1 iteration logic

## Description

### Problem
The network simplex solver's Phase 1 can complete with all artificial arcs having zero flow (which should indicate feasibility), but the solution violates flow conservation at multiple nodes. The solver then proceeds to Phase 2 and returns status="optimal" with an invalid solution.

### Expected Behavior
Phase 1 should either:
1. Find a feasible solution satisfying all flow conservation constraints, OR
2. Detect infeasibility and return status="infeasible"

### Actual Behavior
Phase 1 terminates with:
- All artificial arcs have flow = 0
- Flow conservation violated at multiple nodes
- Solver returns status="optimal" with invalid flows

## Steps to Reproduce

```python
from network_solver.data import build_problem
from network_solver.solver import solve_min_cost_flow

# Create a simple 4-node problem with parallel paths
problem = build_problem(
    nodes=[
        {"id": "a", "supply": 15.0},
        {"id": "b", "supply": 0.0},
        {"id": "c", "supply": 0.0},
        {"id": "d", "supply": -15.0},
    ],
    arcs=[
        {"tail": "a", "head": "b", "capacity": 15.0, "cost": 1.0},
        {"tail": "b", "head": "c", "capacity": 15.0, "cost": 1.0},
        {"tail": "c", "head": "d", "capacity": 15.0, "cost": 1.0},
        {"tail": "a", "head": "d", "capacity": 10.0, "cost": 4.0},  # Parallel path
    ],
    directed=True,
    tolerance=1e-6,
)

result = solve_min_cost_flow(problem)

print(f"Status: {result.status}")
print(f"Objective: {result.objective}")
print(f"Flows: {result.flows}")
print()

# Manual verification of flow conservation
for node_id, supply in [("a", 15.0), ("b", 0.0), ("c", 0.0), ("d", -15.0)]:
    inflow = sum(flow for (tail, head), flow in result.flows.items() if head == node_id)
    outflow = sum(flow for (tail, head), flow in result.flows.items() if tail == node_id)
    balance = supply + inflow - outflow
    status = "✓" if abs(balance) < 1e-6 else "✗ VIOLATED"
    print(f"Node {node_id}: supply={supply}, in={inflow}, out={outflow}, balance={balance} {status}")
```

### Actual Output
```
Status: optimal
Objective: 40.0
Flows: {('a', 'd'): 10.0}

Node a: supply=15.0, in=0.0, out=10.0, balance=5.0 ✗ VIOLATED
Node b: supply=0.0, in=0.0, out=0.0, balance=0.0 ✓
Node c: supply=0.0, in=0.0, out=0.0, balance=0.0 ✓
Node d: supply=-15.0, in=10.0, out=0.0, balance=-5.0 ✗ VIOLATED
```

### Expected Output
```
Status: optimal
Objective: 45.0
Flows: {('a', 'b'): 15.0, ('b', 'c'): 15.0, ('c', 'd'): 15.0}

Node a: supply=15.0, in=0.0, out=15.0, balance=0.0 ✓
Node b: supply=0.0, in=15.0, out=15.0, balance=0.0 ✓
Node c: supply=0.0, in=15.0, out=15.0, balance=0.0 ✓
Node d: supply=-15.0, in=15.0, out=0.0, balance=0.0 ✓
```

## Analysis

### Why This Happens
1. Phase 1 starts with an initial tree (possibly containing artificial arcs)
2. The simplex iterations run but terminate based on some stopping criterion
3. All artificial arcs end up with zero flow (which passes the old infeasibility check)
4. However, flow conservation is NOT satisfied at some nodes
5. The solver incorrectly concludes the problem is feasible and proceeds to Phase 2

### Investigation Needed
1. **Stopping Criteria:** Why does `_run_simplex_iterations()` terminate when conservation is violated?
   - Is it finding no improving pivots even though the solution is infeasible?
   - Is there an issue with the optimality check in Phase 1?

2. **Pivot Selection:** Are the pivot rules failing to find necessary entering arcs?
   - Could this be related to degeneracy handling?
   - Are reduced costs being computed correctly for Phase 1 costs?

3. **Initial Basis:** Does the problem occur with specific initial tree structures?
   - The issue was discovered with a minimal warm-start basis
   - But it also happens with cold-start on the same problem

4. **Network Topology:** Is this specific to certain network structures?
   - Networks with parallel paths of different capacities?
   - Networks with specific cost/capacity ratios?

### Code Locations to Investigate

**Primary suspect - Phase 1 iteration logic:**
- `src/network_solver/simplex.py:1183-1217` - Phase 1 execution
- `src/network_solver/simplex.py:_run_simplex_iterations()` - Main iteration loop
- `src/network_solver/simplex.py:_find_entering_arc()` - Pivot selection

**Infeasibility check (currently insufficient):**
```python
# Line 1222 - Old check only looks at artificial flow
infeasible = any(arc.artificial and arc.flow > self.tolerance for arc in self.arcs)
```

This should also verify flow conservation, not just artificial flow!

## Related Code

### Failing Test
- **File:** `tests/unit/test_warm_start.py`
- **Test:** `TestWarmStartMultiComponent::test_warm_start_single_arc_basis_creates_components`
- **Line:** 652
- **Status:** Currently marked as `xfail` with reason: "Minimal warm-start basis can lead to incomplete Phase 1 - needs investigation"

### Recent Changes
This issue was exposed by the fix in PR #XXX (warm-start infeasibility detection) which added flow conservation validation:

```python
# New validation added after Phase 1 (for warm-start cases)
for node_idx in range(1, self.node_count):
    net_flow = self.node_supply[node_idx]
    for arc in self.arcs:
        if arc.tail == node_idx:
            net_flow -= arc.flow
        elif arc.head == node_idx:
            net_flow += arc.flow
    
    if abs(net_flow) > self.tolerance:
        # Flow conservation violated!
        flow_conservation_violated = True
```

This validation caught the bug, but the issue is pre-existing.

## Proposed Solutions

### Option 1: Add Flow Conservation Check to Phase 1 Completion
Always verify flow conservation after Phase 1, not just for warm-start:

```python
# After Phase 1 completes, verify flow conservation
flow_conservation_violated = False
for node_idx in range(1, self.node_count):
    if node_idx == self.root:
        continue
    
    net_flow = self.node_supply[node_idx]
    for arc in self.arcs:
        if arc.tail == node_idx:
            net_flow -= arc.flow
        elif arc.head == node_idx:
            net_flow += arc.flow
    
    if abs(net_flow) > self.tolerance:
        self.logger.error(f"Flow conservation violated at node {self.node_ids[node_idx]}")
        flow_conservation_violated = True
        break

# If conservation is violated but artificial flow is zero, this is a bug
if flow_conservation_violated and not has_artificial_flow:
    raise RuntimeError("Phase 1 completed with violated flow conservation - algorithm bug")
```

### Option 2: Fix the Root Cause in Phase 1 Iteration
Investigate why Phase 1 terminates early. Possible fixes:
- Ensure Phase 1 doesn't terminate until all artificial arcs are zero AND flow is conserved
- Add additional pivot cycles if conservation is violated
- Improve the stopping criteria in `_run_simplex_iterations()`

### Option 3: Improve Initial Basis Construction
Better initial tree construction might avoid states where Phase 1 can get stuck:
- Use a better heuristic for the initial spanning tree
- Ensure the initial basis is "closer" to feasibility

## Workarounds

### For Users
Avoid using warm-start with minimal bases. Prefer warm-starting with complete or near-complete bases from similar problems.

### For Developers
The flow conservation check added for warm-start cases can be extended to all cases to catch this bug, but it will cause test failures and requires the root cause fix.

## References

- **Original xfail test:** `test_warm_start_capacity_decrease_infeasible_flow` (now fixed)
- **New xfail test:** `test_warm_start_single_arc_basis_creates_components` (this issue)
- **Related PR:** #XXX - Fix warm-start infeasibility detection
- **Related commit:** 3a4cb37 - "Fix warm-start infeasibility detection"

## Impact

**Severity: Critical**
- Solver returns invalid results (flow conservation violated)
- Users get "optimal" status but solution is actually infeasible
- Can lead to incorrect decisions in applications using the solver

**Frequency:**
- Rare in typical usage (most problems don't trigger this)
- More likely with:
  - Warm-start with minimal bases
  - Networks with parallel paths
  - Specific topology patterns (to be determined)

## Checklist for Fix

- [ ] Identify exact stopping condition causing early Phase 1 termination
- [ ] Determine if this is a pivot selection issue or iteration control issue
- [ ] Add comprehensive flow conservation checks after Phase 1 (not just for warm-start)
- [ ] Add unit tests for various network topologies that might trigger this
- [ ] Remove xfail from `test_warm_start_single_arc_basis_creates_components`
- [ ] Verify fix doesn't introduce performance regressions
- [ ] Update documentation if any API changes needed

## Additional Notes

This is a **serious algorithmic bug** that undermines the correctness guarantees of the network simplex algorithm. While it doesn't appear to affect most typical use cases, it can produce silent failures where the solver claims optimality but returns an invalid solution.

The bug has likely existed since the beginning but was masked by the insufficient infeasibility check that only looked at artificial arc flow.

---

**Note to maintainer:** This issue was discovered and documented during work on warm-start infeasibility detection. The failing test has been temporarily marked as xfail to not block that PR, but this issue should be addressed in a follow-up PR focused specifically on Phase 1 correctness.
