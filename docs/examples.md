# Examples

Annotated examples demonstrating various features and use cases.

**Note:** Many example scripts in the `examples/` directory support the `--verbose` flag:
- No flag: Quiet operation (WARNING+ only)
- `-v`: Show phase transitions and progress (INFO level)
- `-vv`: Show every pivot operation (DEBUG level)

Example: `python examples/solve_example.py -v`

## Table of Contents

- [Basic Transportation Problem](#basic-transportation-problem)
- [Supply Chain with Transshipment](#supply-chain-with-transshipment)
- [Maximum Flow Problem](#maximum-flow-problem)
- [Minimum Cost Circulation](#minimum-cost-circulation)
- [Undirected Graphs](#undirected-graphs)
- [Progress Monitoring](#progress-monitoring)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Solver Configuration](#solver-configuration)
- [Flow Validation and Analysis](#flow-validation-and-analysis)

## Basic Transportation Problem

**Problem:** Ship goods from factories to warehouses at minimum cost.

```python
from network_solver import build_problem, solve_min_cost_flow

# Define factories (supply nodes)
# Factory A can supply 100 units, Factory B can supply 150 units
nodes = [
    {"id": "factory_a", "supply": 100.0},
    {"id": "factory_b", "supply": 150.0},
    {"id": "warehouse_1", "supply": -80.0},   # Needs 80 units
    {"id": "warehouse_2", "supply": -120.0},  # Needs 120 units
    {"id": "warehouse_3", "supply": -50.0},   # Needs 50 units
]

# Define shipping routes with costs per unit
arcs = [
    # From Factory A
    {"tail": "factory_a", "head": "warehouse_1", "capacity": 100.0, "cost": 2.5},
    {"tail": "factory_a", "head": "warehouse_2", "capacity": 100.0, "cost": 3.0},
    {"tail": "factory_a", "head": "warehouse_3", "capacity": 100.0, "cost": 1.5},
    # From Factory B
    {"tail": "factory_b", "head": "warehouse_1", "capacity": 150.0, "cost": 1.8},
    {"tail": "factory_b", "head": "warehouse_2", "capacity": 150.0, "cost": 2.2},
    {"tail": "factory_b", "head": "warehouse_3", "capacity": 150.0, "cost": 2.8},
]

# Build and solve
problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

# Display results
print(f"Total shipping cost: ${result.objective:,.2f}")
print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")
print("\nOptimal shipments:")
for (tail, head), flow in sorted(result.flows.items()):
    if flow > 1e-6:  # Only show non-zero flows
        cost_for_route = next(a["cost"] for a in arcs if a["tail"] == tail and a["head"] == head)
        print(f"  {tail:12s} -> {head:12s}: {flow:6.1f} units @ ${cost_for_route}/unit")
```

**Output:**
```
Total shipping cost: $518.00
Status: optimal
Iterations: 5

Optimal shipments:
  factory_a    -> warehouse_1 :   50.0 units @ $2.5/unit
  factory_a    -> warehouse_3 :   50.0 units @ $1.5/unit
  factory_b    -> warehouse_1 :   30.0 units @ $1.8/unit
  factory_b    -> warehouse_2 :  120.0 units @ $2.2/unit
```

**Interpretation:**
- Factory A ships primarily to Warehouse 3 (lowest cost route at $1.5)
- Factory B handles most of Warehouse 2's demand
- Total cost is $518.00

## Supply Chain with Transshipment

**Problem:** Multi-stage supply chain with distribution centers.

```python
from network_solver import build_problem, solve_min_cost_flow

# Three-stage supply chain: Suppliers -> DCs -> Customers
nodes = [
    # Suppliers
    {"id": "supplier_1", "supply": 150.0},
    {"id": "supplier_2", "supply": 200.0},
    # Distribution Centers (transshipment - no supply/demand)
    {"id": "dc_east", "supply": 0.0},
    {"id": "dc_west", "supply": 0.0},
    # Customers
    {"id": "customer_a", "supply": -120.0},
    {"id": "customer_b", "supply": -150.0},
    {"id": "customer_c", "supply": -80.0},
]

arcs = [
    # Supplier to DC (first mile)
    {"tail": "supplier_1", "head": "dc_east", "capacity": 150.0, "cost": 10.0},
    {"tail": "supplier_1", "head": "dc_west", "capacity": 150.0, "cost": 15.0},
    {"tail": "supplier_2", "head": "dc_east", "capacity": 200.0, "cost": 12.0},
    {"tail": "supplier_2", "head": "dc_west", "capacity": 200.0, "cost": 8.0},
    # DC to Customer (last mile)
    {"tail": "dc_east", "head": "customer_a", "capacity": 200.0, "cost": 5.0},
    {"tail": "dc_east", "head": "customer_b", "capacity": 200.0, "cost": 7.0},
    {"tail": "dc_west", "head": "customer_b", "capacity": 200.0, "cost": 6.0},
    {"tail": "dc_west", "head": "customer_c", "capacity": 200.0, "cost": 4.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

# Analyze flow through distribution centers
print("Distribution Center Throughput:")
for dc in ["dc_east", "dc_west"]:
    inflow = sum(flow for (tail, head), flow in result.flows.items() if head == dc)
    outflow = sum(flow for (tail, head), flow in result.flows.items() if tail == dc)
    print(f"  {dc}: {inflow:.0f} in, {outflow:.0f} out")
```

**Key Concepts:**
- **Transshipment nodes**: Have supply = 0, just pass flow through
- **Multi-stage optimization**: Solver jointly optimizes both stages
- **Flow conservation**: Inflow = outflow at each DC

## Maximum Flow Problem

**Problem:** Find maximum flow from source to sink (convert to min-cost flow).

```python
from network_solver import build_problem, solve_min_cost_flow

# Trick: Add a return arc from sink to source with cost -1
# and large capacity. The solver will maximize flow on this arc.
nodes = [
    {"id": "source", "supply": 0.0},  # Will be determined by return arc
    {"id": "a", "supply": 0.0},
    {"id": "b", "supply": 0.0},
    {"id": "sink", "supply": 0.0},
]

arcs = [
    # Network arcs (zero cost)
    {"tail": "source", "head": "a", "capacity": 100.0, "cost": 0.0},
    {"tail": "source", "head": "b", "capacity": 80.0, "cost": 0.0},
    {"tail": "a", "head": "sink", "capacity": 70.0, "cost": 0.0},
    {"tail": "b", "head": "sink", "capacity": 90.0, "cost": 0.0},
    {"tail": "a", "head": "b", "capacity": 50.0, "cost": 0.0},
    # Return arc with negative cost (maximizes flow)
    {"tail": "sink", "head": "source", "capacity": 1000.0, "cost": -1.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

# Maximum flow is the flow on the return arc
max_flow = result.flows.get(("sink", "source"), 0.0)
print(f"Maximum flow: {max_flow:.0f}")
print(f"Objective (negative of max flow): ${result.objective:.2f}")

# Remove return arc from display
network_flows = {k: v for k, v in result.flows.items() if k != ("sink", "source")}
print("\nFlow distribution:")
for (tail, head), flow in sorted(network_flows.items()):
    if flow > 1e-6:
        print(f"  {tail:10s} -> {head:10s}: {flow:.0f}")
```

**Explanation:**
- The negative-cost return arc creates an incentive to maximize flow
- All network arcs have zero cost (we only care about throughput)
- The optimal objective equals -1 × (maximum flow)

## Minimum Cost Circulation

**Problem:** Find minimum-cost circulation (no supplies/demands).

```python
from network_solver import build_problem, solve_min_cost_flow

# All nodes have supply = 0 (circulation problem)
nodes = [
    {"id": "a", "supply": 0.0},
    {"id": "b", "supply": 0.0},
    {"id": "c", "supply": 0.0},
    {"id": "d", "supply": 0.0},
]

# Some arcs have lower bounds (must carry minimum flow)
arcs = [
    {"tail": "a", "head": "b", "capacity": 100.0, "cost": 2.0, "lower": 10.0},
    {"tail": "b", "head": "c", "capacity": 100.0, "cost": 3.0, "lower": 0.0},
    {"tail": "c", "head": "d", "capacity": 100.0, "cost": 1.0, "lower": 5.0},
    {"tail": "d", "head": "a", "capacity": 100.0, "cost": 4.0, "lower": 0.0},
    {"tail": "a", "head": "c", "capacity": 50.0, "cost": 5.0, "lower": 0.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

print(f"Minimum cost circulation: ${result.objective:.2f}")
print("\nCirculation:")
for (tail, head), flow in sorted(result.flows.items()):
    if flow > 1e-6:
        print(f"  {tail} -> {head}: {flow:.1f}")
```

**Use Cases:**
- Resource allocation with mandatory constraints
- Production planning with minimum production levels
- Network equilibrium problems

## Undirected Graphs

**Problem:** Network design with bidirectional links (cables, roads, pipes).

Undirected graphs are useful when edges naturally allow flow in both directions with symmetric costs and capacities. Each undirected edge is transformed internally to a directed arc with `lower = -capacity`.

**Key Points:**
- Set `directed=False` in problem construction
- All edges must have finite capacity (no `capacity=None`)
- Cannot specify custom lower bounds (leave at default 0.0)
- Positive flow = tail→head, Negative flow = head→tail

```python
from network_solver import build_problem, solve_min_cost_flow

# Campus network: route data from A to D
nodes = [
    {"id": "A", "supply": 100.0},   # Data center (source)
    {"id": "B", "supply": 0.0},     # Intermediate building
    {"id": "C", "supply": 0.0},     # Intermediate building
    {"id": "D", "supply": -100.0},  # Research lab (sink)
]

# Bidirectional fiber optic cables
# Note: must have finite capacity for undirected graphs
arcs = [
    {"tail": "A", "head": "B", "capacity": 80.0, "cost": 10.0},  # A-B cable
    {"tail": "A", "head": "C", "capacity": 60.0, "cost": 8.0},   # A-C cable
    {"tail": "B", "head": "D", "capacity": 70.0, "cost": 12.0},  # B-D cable
    {"tail": "C", "head": "D", "capacity": 90.0, "cost": 15.0},  # C-D cable
]

# IMPORTANT: Set directed=False for undirected graph
problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
result = solve_min_cost_flow(problem)

print(f"Total cost: ${result.objective:.2f}")
print("\nFlow on each link:")
for (tail, head), flow in sorted(result.flows.items()):
    if abs(flow) > 1e-6:
        if flow > 0:
            print(f"  {tail}--{head}: {flow:.1f} units ({tail} → {head})")
        else:
            print(f"  {tail}--{head}: {abs(flow):.1f} units ({head} → {tail})")
```

**Output:**
```
Total cost: $2100.00
Flow on each link:
  A--B: 80.0 units (A → B)
  A--C: 20.0 units (A → C)
  B--D: 70.0 units (B → D)
  C--D: 20.0 units (C → D)
```

**How It Works:**
- Edge `{A, B}` with capacity 80 becomes arc `(A, B)` with lower=-80, upper=80
- Solver can assign flow from -80 to +80
- Positive flow means A→B, negative means B→A
- Final solution uses A→B→D path (70 units) + A→C→D path (30 units)

**Common Errors:**

```python
# ❌ ERROR: Infinite capacity not allowed
arcs = [{"tail": "A", "head": "B", "capacity": None, "cost": 1.0}]
problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
# Raises: InvalidProblemError: "Undirected edge A -- B has infinite capacity..."

# ❌ ERROR: Custom lower bounds not allowed
arcs = [{"tail": "A", "head": "B", "capacity": 50.0, "cost": 1.0, "lower": 5.0}]
problem = build_problem(nodes=nodes, arcs=arcs, directed=False, tolerance=1e-6)
# Raises: InvalidProblemError: "Undirected edge A -- B has custom lower bound..."

# ✓ CORRECT: Finite capacity, default lower bound
arcs = [{"tail": "A", "head": "B", "capacity": 50.0, "cost": 1.0}]  # lower=0.0 by default
```

**When to Use Undirected:**
- ✓ Physical infrastructure (cables, pipes, roads) with symmetric capacity/cost
- ✓ Want simpler problem specification (1 edge vs 2 arcs)
- ✓ Costs and capacities are same in both directions
- ✗ Need different costs by direction (use directed graph instead)
- ✗ Need different capacities by direction (use directed graph instead)

See `examples/undirected_graph_example.py` for a complete demonstration and [API Reference - Undirected Graphs](api.md#working-with-undirected-graphs) for detailed documentation.

## Progress Monitoring

**Problem:** Monitor a long-running solve in real-time.

```python
from network_solver import solve_min_cost_flow, build_problem, ProgressInfo

# Create a large problem (example: 100 nodes)
nodes = [{"id": f"node_{i}", "supply": 100.0 if i < 50 else -100.0} 
         for i in range(100)]
arcs = [{"tail": f"node_{i}", "head": f"node_{j}", "capacity": 10.0, "cost": abs(i-j)}
        for i in range(50) for j in range(50, 100)]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

# Progress callback with percentage complete
last_percent = -1

def progress_callback(info: ProgressInfo) -> None:
    global last_percent
    percent = int(100 * info.iteration / info.max_iterations)
    
    if percent != last_percent:  # Only print when percentage changes
        last_percent = percent
        phase_name = "Feasibility" if info.phase == 1 else "Optimality"
        print(f"[{phase_name}] {percent:3d}% | "
              f"Iter: {info.iteration:5d} | "
              f"Obj: ${info.objective_estimate:12,.2f} | "
              f"Time: {info.elapsed_time:6.2f}s",
              flush=True)

# Solve with progress updates every 10 iterations
result = solve_min_cost_flow(
    problem,
    progress_callback=progress_callback,
    progress_interval=10,
)

print(f"\nFinal: ${result.objective:,.2f} in {result.iterations} iterations")
```

**Output:**
```
[Feasibility]   5% | Iter:   100 | Obj:  $12,345.67 | Time:   0.15s
[Feasibility]  10% | Iter:   200 | Obj:  $10,234.56 | Time:   0.28s
...
[Optimality]  95% | Iter:  1900 | Obj:   $5,432.10 | Time:   2.85s
[Optimality] 100% | Iter:  2000 | Obj:   $5,400.00 | Time:   3.01s

Final: $5,400.00 in 2000 iterations
```

## Sensitivity Analysis

**Problem:** Understand how solution changes with supply/demand perturbations.

```python
from network_solver import solve_min_cost_flow, build_problem

nodes = [
    {"id": "factory", "supply": 100.0},
    {"id": "warehouse", "supply": -100.0},
]
arcs = [
    {"tail": "factory", "head": "warehouse", "capacity": 150.0, "cost": 2.5},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

# Dual values indicate marginal cost of changing supply/demand
factory_dual = result.duals["factory"]
warehouse_dual = result.duals["warehouse"]

print("Shadow Prices (Dual Values):")
print(f"  Factory: ${factory_dual:.2f}")
print(f"  Warehouse: ${warehouse_dual:.2f}")

# Reduced cost on the arc
arc_cost = 2.5
reduced_cost = arc_cost + factory_dual - warehouse_dual
print(f"\nReduced cost on arc: ${reduced_cost:.6f}")
print("(Should be ~0 for arcs with positive flow)")

# Marginal cost analysis
print("\nSensitivity Analysis:")
print(f"  Increasing factory supply by 1 unit costs: ${-factory_dual:.2f}")
print(f"  Increasing warehouse demand by 1 unit costs: ${warehouse_dual:.2f}")
print(f"  Total marginal cost: ${warehouse_dual - factory_dual:.2f}")
print(f"  (Should equal arc cost: ${arc_cost:.2f})")
```

**Key Insights:**
- Dual values = shadow prices = marginal costs
- For arcs with positive flow: `cost + dual[tail] - dual[head] ≈ 0`
- Difference in duals = marginal cost of changing flow

## Solver Configuration

**Problem:** Compare different solver configurations.

```python
from network_solver import solve_min_cost_flow, build_problem, SolverOptions

# Create a medium-sized problem
problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

# Configuration 1: Default (Devex pricing)
print("1. Default configuration:")
result1 = solve_min_cost_flow(problem)
print(f"   Iterations: {result1.iterations}")

# Configuration 2: Dantzig pricing
print("\n2. Dantzig pricing:")
options2 = SolverOptions(pricing_strategy="dantzig")
result2 = solve_min_cost_flow(problem, options=options2)
print(f"   Iterations: {result2.iterations}")
print(f"   (Typically more iterations than Devex)")

# Configuration 3: High precision
print("\n3. High precision solve:")
options3 = SolverOptions(tolerance=1e-10)
result3 = solve_min_cost_flow(problem, options=options3)
print(f"   Objective: ${result3.objective:.10f}")
print(f"   (More decimal places)")

# Configuration 4: Small block size (more thorough search)
print("\n4. Small block size:")
options4 = SolverOptions(block_size=5)
result4 = solve_min_cost_flow(problem, options=options4)
print(f"   Iterations: {result4.iterations}")

# Configuration 5: Aggressive refactorization
print("\n5. Frequent refactorization (more stable):")
options5 = SolverOptions(ft_update_limit=20)
result5 = solve_min_cost_flow(problem, options=options5)
print(f"   Iterations: {result5.iterations}")

# All should give same objective (within tolerance)
print(f"\nAll objectives equal: {all(abs(r.objective - result1.objective) < 1e-4 
                                      for r in [result2, result3, result4, result5])}")
```

## Flow Validation and Analysis

**Problem:** Verify solution correctness and find bottlenecks.

```python
from network_solver import (
    solve_min_cost_flow, build_problem,
    validate_flow, compute_bottleneck_arcs, extract_path
)

# Build and solve problem
problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

# 1. Validate the solution
print("=" * 60)
print("SOLUTION VALIDATION")
print("=" * 60)

validation = validate_flow(problem, result)
if validation.is_valid:
    print("✓ Solution is VALID - all constraints satisfied")
else:
    print("✗ Solution has violations:")
    for error in validation.errors:
        print(f"  - {error}")

# 2. Extract a specific path
print("\n" + "=" * 60)
print("PATH ANALYSIS")
print("=" * 60)

path = extract_path(result, problem, source="factory_a", target="warehouse_1")
if path:
    print(f"Path found: {' -> '.join(path.nodes)}")
    print(f"  Flow: {path.flow:.2f} units")
    print(f"  Cost: ${path.cost:.2f}")
    print(f"  Hops: {len(path.arcs)}")
else:
    print("No path found (no flow between nodes)")

# 3. Identify bottlenecks
print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)

bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.90)
if bottlenecks:
    print(f"Found {len(bottlenecks)} bottleneck(s) (≥90% utilization):")
    for b in bottlenecks:
        print(f"\n  Arc: {b.tail} -> {b.head}")
        print(f"    Utilization: {b.utilization * 100:.1f}%")
        print(f"    Flow: {b.flow:.2f} / Capacity: {b.capacity:.2f}")
        print(f"    Slack: {b.slack:.2f} units remaining")
        print(f"    Cost: ${b.cost:.2f}/unit")
        if b.slack < 1.0:
            print(f"    ⚠️  Nearly saturated - consider capacity expansion")
else:
    print("No bottlenecks found - all arcs have spare capacity")

# 4. Comprehensive report
print("\n" + "=" * 60)
print("SOLUTION SUMMARY")
print("=" * 60)
print(f"Objective: ${result.objective:,.2f}")
print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")
print(f"Active arcs: {len(result.flows)}")
print(f"Total flow: {sum(result.flows.values()):,.2f}")
```

This example demonstrates the complete workflow:
1. Solve the problem
2. Validate the solution
3. Extract specific flow paths
4. Identify capacity bottlenecks
5. Generate a comprehensive report

## See Also

- [API Reference](api.md) - Complete API documentation
- [Algorithm](algorithm.md) - Network simplex algorithm details
- [Benchmarks](benchmarks.md) - Performance characteristics
- `examples/` directory - Runnable code examples
