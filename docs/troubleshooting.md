# Troubleshooting Guide

This guide helps diagnose and resolve common issues when using the network flow solver.

## Table of Contents

- [Numeric Issues](#numeric-issues)
- [Convergence Problems](#convergence-problems)
- [Performance Issues](#performance-issues)
- [Problem Formulation](#problem-formulation)
- [Error Messages](#error-messages)

## Numeric Issues

### Detecting Numeric Problems

The solver provides built-in numeric validation to detect potential issues:

```python
from network_solver import build_problem, analyze_numeric_properties

problem = build_problem(
    nodes=[...],
    arcs=[...],
    directed=True,
    tolerance=1e-6
)

# Analyze numeric properties
analysis = analyze_numeric_properties(problem)

if not analysis.is_well_conditioned:
    print("Problem has numeric issues:")
    for warning in analysis.warnings:
        print(f"  {warning.severity}: {warning.message}")
        print(f"    → {warning.recommendation}")
```

### Common Numeric Issues

#### 1. Extreme Value Ranges

**Symptom:** Cost or capacity values span many orders of magnitude (e.g., 1e-8 to 1e12).

**Problem:** Wide ranges can cause numerical instability and precision loss in floating-point arithmetic.

**Solutions:**
- **Scale your problem:** Normalize costs and capacities to similar ranges (e.g., [0.01, 1000])
- **Use consistent units:** Convert all values to consistent units before solving
- **Increase tolerance:** Use a larger tolerance value for very wide ranges

```python
from network_solver import SolverOptions, solve_min_cost_flow

# For wide-range problems, increase tolerance
options = SolverOptions(tolerance=1e-4)
result = solve_min_cost_flow(problem, options=options)
```

#### 2. Very Large or Very Small Values

**Symptom:** Individual costs, capacities, or supplies are extremely large (>1e10) or small (<1e-10).

**Problem:** Can cause overflow, underflow, or loss of precision.

**Solutions:**
- **Scale the problem:** Divide all costs by a common factor
- **Example:** If costs are in range [1e8, 1e12], divide by 1e8

```python
# Scale costs down
scaled_arcs = [
    {
        "tail": arc["tail"],
        "head": arc["head"],
        "capacity": arc["capacity"],
        "cost": arc["cost"] / 1e8  # Scale down
    }
    for arc in original_arcs
]

# Solve with scaled problem
result = solve_min_cost_flow(scaled_problem)

# Scale objective back up
true_objective = result.objective * 1e8
```

#### 3. Coefficient Range Warnings

**Symptom:** `analyze_numeric_properties()` reports wide coefficient ranges.

**Problem:** Ratio of max/min values is too large (>1e6).

**Solutions:**

```python
# Check recommended tolerance
analysis = analyze_numeric_properties(problem)
print(f"Recommended tolerance: {analysis.recommended_tolerance}")

# Use recommended settings
options = SolverOptions(tolerance=analysis.recommended_tolerance)
result = solve_min_cost_flow(problem, options=options)
```

### Strict Validation Mode

Enable strict mode to prevent solving ill-conditioned problems:

```python
from network_solver import validate_numeric_properties

try:
    validate_numeric_properties(problem, strict=True)
    result = solve_min_cost_flow(problem)
except ValueError as e:
    print(f"Problem validation failed: {e}")
    # Apply scaling or reformulation
```

## Convergence Problems

### Detecting Convergence Issues

Monitor convergence during solving:

```python
from network_solver import solve_min_cost_flow, ConvergenceMonitor

monitor = ConvergenceMonitor(window_size=50, stall_threshold=1e-8)

def track_progress(info):
    # Called every 100 iterations
    monitor.record_iteration(
        objective=info.objective_estimate,
        is_degenerate=(info.phase == 2),  # Approximate
        iteration=info.iteration
    )

    if monitor.is_stalled():
        print(f"Warning: Stalling detected at iteration {info.iteration}")
        diagnostics = monitor.get_diagnostic_summary()
        print(f"  Degeneracy ratio: {diagnostics['degeneracy_ratio']:.2%}")

result = solve_min_cost_flow(
    problem,
    progress_callback=track_progress,
    progress_interval=100
)
```

### Common Convergence Issues

#### 1. Stalling (Slow Progress)

**Symptom:** Objective value barely changes over many iterations.

**Causes:**
- High degeneracy (many zero pivots)
- Ill-conditioned problem
- Tight tolerance relative to coefficient ranges

**Solutions:**

```python
# Increase tolerance
options = SolverOptions(tolerance=1e-4)  # Looser than default 1e-6

# Or use smaller block size for better arc selection
options = SolverOptions(block_size=20)

# Or switch pricing strategy
options = SolverOptions(pricing_strategy="dantzig")

result = solve_min_cost_flow(problem, options=options)
```

#### 2. High Degeneracy

**Symptom:** Many pivots don't change the objective value.

**Causes:**
- Problem structure (especially transportation/assignment problems)
- Cost perturbation not sufficient

**Solutions:**
- Use auto-tuning (enabled by default): `SolverOptions(block_size="auto")`
- Accept that degeneracy is common - it's not always a problem
- Check if problem has alternative optimal solutions

#### 3. Cycling Detection

**Symptom:** Same basis visited multiple times.

**Detection:**

```python
from network_solver import BasisHistory

history = BasisHistory(max_history=100)

# Track during custom solving (advanced use)
if history.is_cycling(min_revisits=3):
    print("Cycling detected!")
    cycle_length = history.get_cycle_length()
    if cycle_length:
        print(f"Estimated cycle length: {cycle_length}")
```

**Solutions:**
- Use cost perturbation (solver does this automatically)
- Increase tolerance
- Try different pricing strategy

## Performance Issues

### Slow Solving

#### Diagnose Performance

```python
import time
from network_solver import solve_min_cost_flow

start = time.time()
result = solve_min_cost_flow(problem)
elapsed = time.time() - start

print(f"Solved in {elapsed:.2f} seconds")
print(f"Iterations: {result.iterations}")
print(f"Iterations per second: {result.iterations / elapsed:.0f}")
```

#### Common Causes and Solutions

**1. Too many iterations:**
- **Check:** `result.iterations` is very high relative to problem size
- **Solution:** Use better pricing strategy or tune block size

```python
# Try Devex pricing with auto-tuning (default, but explicit)
options = SolverOptions(
    pricing_strategy="devex",
    block_size="auto"
)
```

**2. Poor initial basis:**
- **Solution:** Use warm-starting for sequential solves

```python
# First solve
result1 = solve_min_cost_flow(problem1)
basis = result1.basis

# Modified problem - reuse basis
result2 = solve_min_cost_flow(problem2, warm_start_basis=basis)
# Often 50-90% fewer iterations!
```

**3. Frequent basis refactorization:**
- **Check:** If debugging shows frequent rebuilds
- **Solution:** Adjust `ft_update_limit`

```python
# Fewer rebuilds (faster but less stable)
options = SolverOptions(ft_update_limit=128)

# More rebuilds (slower but more stable)
options = SolverOptions(ft_update_limit=32)
```

### Memory Issues

For very large problems (>100k arcs):

1. **Monitor memory usage**
2. **Consider problem reduction** (remove redundant arcs)
3. **Use sparse data structures** (solver does this when scipy available)

## Problem Formulation

### Unbalanced Supply/Demand

**Error:** `InvalidProblemError: Problem is unbalanced`

**Cause:** Sum of node supplies doesn't equal zero.

**Solution:**

```python
# Check balance
total_supply = sum(node["supply"] for node in nodes)
print(f"Total supply: {total_supply}")

# Add balancing node if needed
if abs(total_supply) > 1e-6:
    nodes.append({
        "id": "slack",
        "supply": -total_supply
    })
    # Add arcs to/from slack node as needed
```

### Unbounded Problem

**Error:** `UnboundedProblemError: entering arc can increase indefinitely`

**Cause:** Negative-cost cycle with infinite capacity.

**Solutions:**
1. **Check for negative-cost cycles** without capacity restrictions
2. **Add capacity bounds** to all arcs
3. **Verify cost signs** (negative costs mean revenue, not cost)

```python
from network_solver import UnboundedProblemError

try:
    result = solve_min_cost_flow(problem)
except UnboundedProblemError as e:
    print(f"Unbounded at arc: {e.entering_arc}")
    print(f"Reduced cost: {e.reduced_cost}")
    # Review problem formulation
```

### Infeasible Problem

**Status:** `result.status == "infeasible"`

**Cause:** No feasible flow exists satisfying all constraints.

**Common Reasons:**
1. **Insufficient capacity:** Arc capacities can't satisfy demands
2. **Disconnected network:** No path from supplies to demands
3. **Lower bound violations:** Lower bounds conflict with capacities

**Debugging:**

```python
from network_solver import InfeasibleProblemError

try:
    result = solve_min_cost_flow(problem)
    if result.status == "infeasible":
        print(f"No feasible solution found after {result.iterations} iterations")
        # Check for disconnected components
        # Verify arc capacities are sufficient
except InfeasibleProblemError as e:
    print(f"Infeasible: {e}")
    print(f"Iterations attempted: {e.iterations}")
```

## Error Messages

### Common Error Messages and Solutions

#### `InvalidProblemError: Arc tail 'X' not found in node set`

**Cause:** Arc references a node that doesn't exist.

**Solution:** Ensure all arc endpoints reference valid node IDs.

```python
# Verify all nodes referenced
arc_nodes = set()
for arc in arcs:
    arc_nodes.add(arc["tail"])
    arc_nodes.add(arc["head"])

node_ids = set(node["id"] for node in nodes)
missing = arc_nodes - node_ids

if missing:
    print(f"Missing nodes: {missing}")
```

#### `InvalidProblemError: Tolerance must be positive`

**Cause:** Invalid tolerance value in `SolverOptions`.

**Solution:** Use positive tolerance value.

```python
# Wrong
options = SolverOptions(tolerance=0)

# Correct
options = SolverOptions(tolerance=1e-6)
```

#### `RuntimeError: Tree reconstruction failed: disconnected spanning tree`

**Cause:** Internal solver error - basis tree is disconnected (rare).

**Solution:**
1. Check for numerical issues
2. Try different solver options
3. Validate problem formulation
4. Report bug with minimal example

## Getting Help

If issues persist:

1. **Validate your problem:**
   ```python
   from network_solver import validate_numeric_properties
   validate_numeric_properties(problem, warn=True)
   ```

2. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Create minimal reproducible example**
4. **Check GitHub issues:** https://github.com/jeffreyhorn/network_flow_solver/issues
5. **Open new issue with:**
   - Problem description
   - Minimal code to reproduce
   - Solver output/error messages
   - System information

## Best Practices

### Problem Formulation

✅ **Do:**
- Keep coefficient values in reasonable ranges ([1e-6, 1e6])
- Use consistent units across all values
- Balance supply and demand
- Add finite capacities to most arcs
- Validate problems before solving

❌ **Don't:**
- Mix very large and very small values
- Leave problems unbalanced
- Use infinite capacities for undirected edges
- Ignore numeric warnings

### Performance

✅ **Do:**
- Use warm-starting for sequential solves
- Enable auto-tuning (default)
- Monitor convergence for long-running solves
- Consider problem specializations

❌ **Don't:**
- Use unnecessarily tight tolerance
- Rebuild basis too frequently
- Ignore stalling warnings
- Solve same problem repeatedly without warm-starting

### Reliability

✅ **Do:**
- Validate numeric properties
- Monitor convergence
- Handle exceptions appropriately
- Test on small instances first

❌ **Don't:**
- Skip validation on production problems
- Ignore warnings about ill-conditioning
- Assume all problems will solve quickly
- Use untested parameter combinations
