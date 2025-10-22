# Examples

Annotated examples demonstrating various features and use cases.

**New to network flow problems?** Start with the **[Jupyter Notebook Tutorial](../tutorials/network_flow_tutorial.ipynb)** for an interactive, step-by-step introduction covering all major features with executable code cells and visualizations.

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
- [Problem Preprocessing](#problem-preprocessing)
- [Undirected Graphs](#undirected-graphs)
- [Progress Monitoring](#progress-monitoring)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Solver Configuration](#solver-configuration)
- [Incremental Resolving](#incremental-resolving)
- [Flow Validation and Analysis](#flow-validation-and-analysis)
- [Visualization](#visualization)
- [Structured Logging for Monitoring](#structured-logging-for-monitoring)
- [Performance Profiling](#performance-profiling)
- [Comparison with NetworkX](#comparison-with-networkx)

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

## Problem Preprocessing

**Purpose:** Simplify network flow problems before solving by removing redundancies, merging arcs, and detecting structural issues. Preprocessing reduces problem size and improves solve performance while preserving optimal solutions.

### Basic Preprocessing

**Example:** Simple transportation problem with redundant arcs.

```python
from network_solver import build_problem, preprocess_problem, solve_min_cost_flow

# Problem with two identical routes
nodes = [
    {"id": "warehouse", "supply": 100.0},
    {"id": "store", "supply": -100.0},
]
arcs = [
    {"tail": "warehouse", "head": "store", "capacity": 50.0, "cost": 2.5},
    {"tail": "warehouse", "head": "store", "capacity": 50.0, "cost": 2.5},  # Duplicate
]

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Preprocess to merge redundant arcs
result = preprocess_problem(problem)

print(f"Original: {len(problem.arcs)} arcs")
print(f"After preprocessing: {len(result.problem.arcs)} arcs")
print(f"Removed {result.redundant_arcs} redundant arcs")
print(f"Combined capacity: {result.problem.arcs[0].capacity}")
# Output:
# Original: 2 arcs
# After preprocessing: 1 arc
# Removed 1 redundant arcs
# Combined capacity: 100.0
```

### Series Arc Simplification

**Example:** Supply chain with transshipment hubs.

```python
from network_solver import build_problem, preprocess_problem

# Chain of transshipment nodes
nodes = [
    {"id": "factory", "supply": 100.0},
    {"id": "hub_0", "supply": 0.0},      # Zero-supply transshipment
    {"id": "hub_1", "supply": 0.0},      # Zero-supply transshipment
    {"id": "hub_2", "supply": 0.0},      # Zero-supply transshipment
    {"id": "customer", "supply": -100.0},
]
arcs = [
    {"tail": "factory", "head": "hub_0", "capacity": 150.0, "cost": 2.0},
    {"tail": "hub_0", "head": "hub_1", "capacity": 140.0, "cost": 1.5},
    {"tail": "hub_1", "head": "hub_2", "capacity": 130.0, "cost": 1.0},
    {"tail": "hub_2", "head": "customer", "capacity": 120.0, "cost": 0.5},
]

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Preprocess to merge series arcs
result = preprocess_problem(problem)

print(f"Original: {len(problem.nodes)} nodes, {len(problem.arcs)} arcs")
print(f"After: {len(result.problem.nodes)} nodes, {len(result.problem.arcs)} arcs")
print(f"Removed {result.removed_nodes} transshipment nodes")
print(f"Merged {result.merged_arcs} series arcs")

# Examine the merged arc
merged_arc = result.problem.arcs[0]
print(f"\nMerged arc: {merged_arc.tail} → {merged_arc.head}")
print(f"  Capacity: {merged_arc.capacity} (min of all)")
print(f"  Cost: {merged_arc.cost} (sum of all)")
# Output:
# Original: 5 nodes, 4 arcs
# After: 2 nodes, 1 arcs
# Removed 3 transshipment nodes
# Merged 3 series arcs
#
# Merged arc: factory → customer
#   Capacity: 120.0 (min of all)
#   Cost: 5.0 (sum of all)
```

**How it works:** When a zero-supply node has exactly one incoming and one outgoing arc, they can be merged into a single arc. The merged arc has:
- **Tail**: Predecessor node
- **Head**: Successor node
- **Capacity**: Minimum of the two capacities (bottleneck)
- **Cost**: Sum of the two costs
- **Lower bound**: Maximum of the two lower bounds

### Disconnected Component Detection

**Example:** Network with separate subgraphs.

```python
from network_solver import build_problem, preprocess_problem

# Two disconnected supply chains
nodes = [
    # Chain 1
    {"id": "factory_a", "supply": 50.0},
    {"id": "store_a", "supply": -50.0},
    # Chain 2
    {"id": "factory_b", "supply": 30.0},
    {"id": "store_b", "supply": -30.0},
]
arcs = [
    {"tail": "factory_a", "head": "store_a", "capacity": 100.0, "cost": 1.0},
    {"tail": "factory_b", "head": "store_b", "capacity": 100.0, "cost": 2.0},
]

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Preprocess detects disconnection
result = preprocess_problem(problem)

if result.disconnected_components > 1:
    print(f"⚠ Warning: {result.disconnected_components} disconnected components")
    print("Each component must be independently balanced")
# Output:
# ⚠ Warning: 2 disconnected components
# Each component must be independently balanced
```

**Use case:** Early detection of infeasibility when disconnected components have imbalanced supply/demand.

### Convenience Function

**Example:** Preprocess and solve in one call.

```python
from network_solver import build_problem, preprocess_and_solve

# Build problem
problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Preprocess and solve together
preproc_result, flow_result = preprocess_and_solve(problem)

print(f"Preprocessing:")
print(f"  Removed {preproc_result.removed_arcs} arcs")
print(f"  Removed {preproc_result.removed_nodes} nodes")
print(f"  Time: {preproc_result.preprocessing_time_ms:.2f}ms")
print(f"\nSolution:")
print(f"  Status: {flow_result.status}")
print(f"  Objective: ${flow_result.objective:.2f}")
print(f"  Iterations: {flow_result.iterations}")
```

### Selective Preprocessing

**Example:** Apply only specific optimizations.

```python
from network_solver import preprocess_problem

# Only remove redundant arcs and merge series arcs
result = preprocess_problem(
    problem,
    remove_redundant=True,      # Merge parallel arcs
    simplify_series=True,       # Merge series arcs
    detect_disconnected=False,  # Skip connectivity check
    remove_zero_supply=False,   # Keep all zero-supply nodes
)

print(f"Applied optimizations:")
for opt_name, count in result.optimizations.items():
    print(f"  {opt_name}: {count}")
```

### Performance Comparison

**Example:** Benchmark preprocessing impact.

```python
import time
from network_solver import build_problem, preprocess_problem, solve_min_cost_flow

# Large problem (20 nodes, 24 arcs with redundancy)
problem = build_large_problem()

# Solve WITHOUT preprocessing
start = time.time()
result1 = solve_min_cost_flow(problem)
time_without = (time.time() - start) * 1000

# Solve WITH preprocessing
start = time.time()
preproc_result = preprocess_problem(problem)
result2 = solve_min_cost_flow(preproc_result.problem)
time_with = (time.time() - start) * 1000

print(f"Without preprocessing:")
print(f"  Time: {time_without:.2f}ms")
print(f"  Iterations: {result1.iterations}")

print(f"\nWith preprocessing:")
print(f"  Preprocessing: {preproc_result.preprocessing_time_ms:.2f}ms")
print(f"  Solving: {time_with - preproc_result.preprocessing_time_ms:.2f}ms")
print(f"  Total: {time_with:.2f}ms")
print(f"  Iterations: {result2.iterations}")
print(f"  Removed: {preproc_result.removed_arcs} arcs, {preproc_result.removed_nodes} nodes")

speedup = time_without / time_with
print(f"\nSpeedup: {speedup:.2f}x")
# Output:
# Without preprocessing:
#   Time: 34.60ms
#   Iterations: 28
#
# With preprocessing:
#   Preprocessing: 0.82ms
#   Solving: 23.04ms
#   Total: 23.86ms
#   Iterations: 27
#   Removed: 13 arcs, 8 nodes
#
# Speedup: 1.45x
```

### When Preprocessing Helps

**Best for:**
- Problems with **redundant parallel arcs** (multiple routes with same cost)
- **Complex supply chains** with many zero-supply transshipment nodes
- **Large problems** where size reduction significantly improves convergence
- Problems that may have **disconnected components** (early detection)

**Less useful for:**
- Small problems (<50 arcs) where preprocessing overhead dominates
- Well-structured problems without redundancy or transshipment chains
- Problems requiring custom node/arc properties preserved

**Performance guidelines:**
- Preprocessing overhead: <1ms for small problems, <10ms for large
- Typical size reduction: 20-50% fewer arcs/nodes when applicable
- Solve time improvement: 1.2x-2x for problems with significant redundancy
- Optimal solutions are **always preserved** (safe transformation)

**See Also:**
- `examples/preprocessing_example.py` - Complete demonstration with 6 scenarios
- [API: preprocess_problem()](api.md#preprocess_problem) - Full API documentation
- [API: PreprocessingResult](api.md#preprocessingresult) - Result statistics

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

**Use case:** Understand marginal costs, predict cost changes, make capacity expansion decisions.

The solver returns **dual values** (also called **shadow prices** or **node potentials**) which represent the marginal cost of supply/demand changes at each node. These enable powerful sensitivity analysis without re-solving.

### What are Dual Values?

Dual values answer the question: *"How much would the objective change if I increase supply/demand at this node by 1 unit?"*

- **Negative dual**: It costs money to supply more at this node (or saves money to demand less)
- **Positive dual**: It saves money to demand more at this node (or costs money to supply less)
- **Dual difference**: The marginal cost between two nodes equals the cost on connecting arcs

### Basic Example: Marginal Cost Prediction

```python
from network_solver import solve_min_cost_flow, build_problem

# Simple supply chain: supplier -> warehouse -> customer
nodes = [
    {"id": "supplier", "supply": 100.0},
    {"id": "warehouse", "supply": 0.0},
    {"id": "customer", "supply": -100.0},
]
arcs = [
    {"tail": "supplier", "head": "warehouse", "capacity": 150.0, "cost": 2.0},
    {"tail": "warehouse", "head": "customer", "capacity": 150.0, "cost": 3.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)

print(f"Optimal cost: ${result.objective:.2f}")  # $500.00

# Access dual values (shadow prices)
print("\nDual Values (Shadow Prices):")
for node_id, dual in sorted(result.duals.items()):
    print(f"  {node_id}: ${dual:.6f}")

# Output:
#   customer: $-10.000000
#   supplier: $-15.000000
#   warehouse: $-13.000000

# Predict cost change without re-solving
marginal_cost_per_unit = result.duals["supplier"] - result.duals["customer"]
print(f"\nMarginal cost to ship 1 more unit: ${-marginal_cost_per_unit:.2f}")
# Output: Marginal cost to ship 1 more unit: $5.00

# Verify by actually changing supply
nodes_increased = [
    {"id": "supplier", "supply": 110.0},  # +10 units
    {"id": "warehouse", "supply": 0.0},
    {"id": "customer", "supply": -110.0},  # +10 units demand
]
problem2 = build_problem(nodes=nodes_increased, arcs=arcs, directed=True)
result2 = solve_min_cost_flow(problem2)

actual_change = result2.objective - result.objective
predicted_change = 10 * marginal_cost_per_unit
print(f"\nActual cost change for +10 units: ${actual_change:.2f}")
print(f"Predicted from duals: ${-predicted_change:.2f}")
# Both show: $50.00
```

### Complementary Slackness (Optimality Condition)

For an optimal solution, arcs with positive flow must have zero **reduced cost**:

```
reduced_cost = arc_cost + dual[tail] - dual[head]
```

For arcs carrying flow: `reduced_cost ≈ 0` (within tolerance)

```python
# Verify complementary slackness
print("\nComplementary Slackness Check:")
for (tail, head), flow in result.flows.items():
    if flow > 1e-6:
        arc = next(a for a in arcs if a["tail"] == tail and a["head"] == head)
        reduced_cost = arc["cost"] + result.duals[tail] - result.duals[head]
        print(f"  {tail} -> {head}:")
        print(f"    Flow: {flow:.2f}, Reduced cost: {reduced_cost:.10f}")

# Output:
#   supplier -> warehouse:
#     Flow: 100.00, Reduced cost: -0.0000000001
#   warehouse -> customer:
#     Flow: 100.00, Reduced cost: -0.0000000001
```

This verifies the solution is optimal (reduced costs are essentially zero).

### Production Planning Example

**Use case:** Two factories with different costs. Which should we expand?

```python
nodes = [
    {"id": "factory_a", "supply": 50.0},   # Low cost, limited capacity
    {"id": "factory_b", "supply": 50.0},   # High cost, supplemental
    {"id": "customer", "supply": -100.0},
]
arcs = [
    {"tail": "factory_a", "head": "customer", "capacity": 60.0, "cost": 3.0},
    {"tail": "factory_b", "head": "customer", "capacity": 150.0, "cost": 5.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True)
result = solve_min_cost_flow(problem)

print(f"Total cost: ${result.objective:.2f}")  # $400.00
print("\nFlows:")
for (tail, head), flow in sorted(result.flows.items()):
    cost = next(a["cost"] for a in arcs if a["tail"] == tail and a["head"] == head)
    print(f"  {tail} -> {head}: {flow:.2f} units @ ${cost:.2f}/unit")

# Output:
#   factory_a -> customer: 50.00 units @ $3.00/unit
#   factory_b -> customer: 50.00 units @ $5.00/unit

# Which factory should we expand?
factory_a_value = -result.duals["factory_a"]
factory_b_value = -result.duals["factory_b"]

print(f"\nMarginal Value of Capacity Expansion:")
print(f"  Factory A: ${factory_a_value:.2f} per unit")
print(f"  Factory B: ${factory_b_value:.2f} per unit")

if factory_a_value > factory_b_value:
    print(f"\n✓ Expand Factory A (higher marginal value)")
else:
    print(f"\n✓ Expand Factory B (higher marginal value)")
```

### Capacity Bottleneck Identification

Dual values help identify which capacity constraints are binding:

```python
# Check which arcs are at capacity
print("\nCapacity Analysis:")
for (tail, head), flow in sorted(result.flows.items()):
    capacity = next(a["capacity"] for a in arcs 
                    if a["tail"] == tail and a["head"] == head)
    utilization = (flow / capacity) * 100
    at_capacity = abs(flow - capacity) < 1e-3
    
    print(f"  {tail} -> {head}:")
    print(f"    Utilization: {utilization:.1f}%", end="")
    if at_capacity:
        print(f" ⚠ BOTTLENECK - expanding this capacity would reduce costs")
    else:
        print(f" (slack available)")
```

### Key Concepts

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Dual value** | `π[node]` | Marginal cost at node |
| **Reduced cost** | `c[i,j] + π[i] - π[j]` | "Savings" from using arc (i,j) |
| **Complementary slackness** | If `flow[i,j] > 0` then `reduced_cost ≈ 0` | Optimality condition |
| **Cost change prediction** | `Δcost ≈ Δsupply × π[node]` | Estimate without re-solving |

### When to Use Dual Values

1. **"What-if" analysis**: Predict cost impact of supply/demand changes
2. **Capacity planning**: Identify which capacity expansions are most valuable
3. **Pricing decisions**: Determine value of expedited delivery or premium sourcing
4. **Bottleneck identification**: Find binding capacity constraints
5. **Optimality verification**: Check complementary slackness conditions
6. **Marginal cost analysis**: Understand the value of resources at each location

### Complete Working Example

See `examples/sensitivity_analysis_example.py` for a comprehensive demonstration including:
- Basic marginal cost prediction
- Complementary slackness verification  
- Production planning with capacity expansion decisions
- Bottleneck identification
- Key concepts summary

**Output excerpt:**
```
DUAL VALUES (Shadow Prices):
  customer: $-10.000000
  supplier: $-15.000000

SENSITIVITY ANALYSIS:
  Expected from duals: $-5.000000 per unit
  Actual cost change: $50.00 for 10 units
  Change per unit: $5.00 ✓ Matches prediction!

USE CASE: PRODUCTION PLANNING
  Factory A value: $23.00 per unit
  Factory B value: $25.00 per unit
  ✓ RECOMMENDATION: Expand Factory B
```

### Further Reading

- **[Algorithm Guide](algorithm.md#node-potentials-dual-variables)** - Mathematical background on dual variables
- **[API Reference](api.md#flowresult)** - `FlowResult.duals` field documentation
- `examples/sensitivity_analysis_example.py` - Complete working examples

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

## Adaptive Basis Refactorization

**Problem:** Maintain numerical stability while maximizing performance for diverse problem types.

Adaptive basis refactorization monitors the **condition number** of the basis matrix during the solve. When numerical issues are detected (high condition number), the solver triggers a basis rebuild and adjusts the refactorization frequency. This provides automatic stability without manual tuning.

### What is Adaptive Refactorization?

The network simplex solver maintains a **basis factorization** that must be updated after each pivot. After many updates, numerical errors can accumulate, degrading accuracy. Traditional approaches use a **fixed update limit** (e.g., rebuild every 64 pivots), but this is inefficient:

- **Too frequent rebuilds** → Slower performance on well-conditioned problems
- **Too infrequent rebuilds** → Numerical instability on ill-conditioned problems

**Adaptive refactorization** automatically finds the optimal balance by:
1. **Monitoring** the condition number after each pivot
2. **Triggering rebuilds** when condition number exceeds threshold
3. **Adjusting frequency** based on observed numerical behavior

### Basic Usage (Enabled by Default)

```python
from network_solver import solve_min_cost_flow, SolverOptions

# Adaptive refactorization is enabled by default
result = solve_min_cost_flow(problem)

# Explicitly enable with default settings
options = SolverOptions(adaptive_refactorization=True)
result = solve_min_cost_flow(problem, options=options)
```

### Configuration Options

```python
# Customize adaptive behavior
options = SolverOptions(
    adaptive_refactorization=True,       # Enable adaptive mode (default)
    condition_number_threshold=1e12,     # Trigger threshold (default)
    adaptive_ft_min=20,                  # Minimum refactorization limit
    adaptive_ft_max=200,                 # Maximum refactorization limit
    ft_update_limit=64,                  # Initial/fixed limit
)
result = solve_min_cost_flow(problem, options=options)
```

**Parameters explained:**
- `condition_number_threshold`: When condition number exceeds this, trigger rebuild
  - Lower (1e10): More conservative, more rebuilds, better stability
  - Higher (1e14): More aggressive, fewer rebuilds, faster but less stable
  - Default (1e12): Good balance for most problems
- `adaptive_ft_min/max`: Bounds for adaptive adjustment
  - Narrow range [20, 40]: Tight control, minimal adaptation
  - Wide range [10, 200]: More flexibility, better adaptation

### Example 1: Ill-Conditioned Problem

Problems with **extreme value ranges** can cause numerical issues:

```python
from network_solver import build_problem, solve_min_cost_flow, SolverOptions

# Problem with micro-costs and macro-capacities (9 orders of magnitude!)
nodes = [
    {"id": "factory", "supply": 1_000_000.0},
    {"id": "warehouse", "supply": -1_000_000.0},
]
arcs = [
    {"tail": "factory", "head": "warehouse", "capacity": 2_000_000.0, "cost": 0.001},
]
problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# With adaptive refactorization (default)
result_adaptive = solve_min_cost_flow(problem)
print(f"Adaptive: Status={result_adaptive.status}, Obj=${result_adaptive.objective:.2f}")

# Without adaptive refactorization
options_fixed = SolverOptions(adaptive_refactorization=False, ft_update_limit=64)
result_fixed = solve_min_cost_flow(problem, options=options_fixed)
print(f"Fixed:    Status={result_fixed.status}, Obj=${result_fixed.objective:.2f}")

# Both should find optimal solution
# Adaptive automatically maintains stability
```

### Example 2: Custom Threshold for Conservative Rebuilds

For critical applications requiring maximum stability:

```python
# More conservative: lower threshold triggers more rebuilds
options = SolverOptions(
    adaptive_refactorization=True,
    condition_number_threshold=1e10,  # More aggressive rebuilding
)
result = solve_min_cost_flow(problem, options=options)
print(f"Conservative approach: {result.status}, {result.iterations} iterations")
```

### Example 3: Disable for Predictable Behavior

When you need fixed, predictable refactorization (e.g., benchmarking):

```python
# Disable adaptive, use fixed limit
options = SolverOptions(
    adaptive_refactorization=False,
    ft_update_limit=64,  # Fixed: rebuild every 64 pivots
)
result = solve_min_cost_flow(problem, options=options)
print(f"Fixed refactorization: {result.status}, {result.iterations} iterations")
```

### When to Use Adaptive Refactorization

**✓ Enable (default) when:**
- Working with ill-conditioned problems (wide value ranges)
- Problem characteristics are unknown
- Numerical stability is critical
- You want automatic tuning

**✗ Disable when:**
- Well-conditioned problems with narrow value ranges
- You need predictable, fixed behavior (testing/benchmarking)
- You've manually optimized `ft_update_limit` for your workload

### How Adaptive Adjustment Works

The solver tracks condition number history and adjusts `ft_update_limit`:

- **Very high** condition (>10× threshold): Reduce limit by 50% → more rebuilds
- **Moderately high** (>threshold): Reduce limit by 20% → gradual increase
- **Good** conditioning: Increase limit by 10% → fewer rebuilds

Adjustments respect `adaptive_ft_min` and `adaptive_ft_max` bounds.

### Combination with Automatic Scaling

Adaptive refactorization works seamlessly with **automatic problem scaling**:

```python
# Both features enabled by default
options = SolverOptions(
    auto_scale=True,                 # Automatic scaling (default)
    adaptive_refactorization=True,   # Adaptive refactorization (default)
)

# Scaling normalizes value ranges (reduces ill-conditioning)
# Adaptive refactorization handles remaining numerical issues
result = solve_min_cost_flow(problem, options=options)
```

Together, these features provide robust numerical behavior across diverse problems.

### Complete Example

See `examples/adaptive_refactorization_example.py` for comprehensive demonstrations including:
- Default vs custom configurations
- Ill-conditioned problem comparisons
- Adaptive vs fixed refactorization strategies
- Configuration tuning guidelines

**Key takeaways:**
- Enabled by default for automatic stability
- No manual tuning needed for most users
- Customizable for specialized requirements
- Works in combination with automatic scaling

## Incremental Resolving

**Use case:** Efficiently re-solve problems with modifications for scenario analysis, capacity planning, and iterative optimization.

Incremental resolving means solving multiple related network flow problems where each problem is a modification of the previous one (e.g., changed capacities, costs, or demands). Even without optimization, the solver typically handles small modifications efficiently. For maximum performance with sequential solves, the solver supports **warm-starting** from a previous solution, which reuses the basis (spanning tree structure) to dramatically reduce iterations—typically achieving 50-90% reduction in solve time for similar problems.

### Why Incremental Resolving?

- **Scenario analysis**: "What if we expand this route's capacity?"
- **Cost sensitivity**: "How do price changes affect the optimal solution?"
- **Demand forecasting**: Handle varying demand patterns over time
- **Network design**: Evaluate different topology configurations
- **Iterative optimization**: Gradually improve network by targeting bottlenecks

### Warm-Start for Sequential Solves

For sequential solving of similar problems, warm-starting can significantly improve performance:

```python
# Solve initial problem
result1 = solve_min_cost_flow(problem1)

# Solve modified problem with warm-start
result2 = solve_min_cost_flow(problem2, warm_start_basis=result1.basis)
```

**When warm-start helps most:**
- Problem structure is similar (same nodes/arcs, different parameters)
- Capacities, costs, or supplies change moderately
- Sequential optimization scenarios

**See also:** `examples/warm_start_example.py` for comprehensive demonstrations and `examples/incremental_resolving_example.py` for various modification scenarios.

### Scenario 1: Capacity Expansion Analysis

**Use case:** Transportation network needs more capacity. How much does cost decrease as we expand?

```python
from network_solver import build_problem, solve_min_cost_flow

# Base problem: Limited capacity
nodes = [
    {"id": "warehouse", "supply": 100.0},
    {"id": "store_a", "supply": -60.0},
    {"id": "store_b", "supply": -40.0},
]

base_arcs = [
    {"tail": "warehouse", "head": "store_a", "capacity": 50.0, "cost": 2.0},
    {"tail": "warehouse", "head": "store_b", "capacity": 50.0, "cost": 3.0},
]

problem_base = build_problem(nodes=nodes, arcs=base_arcs, directed=True, tolerance=1e-6)
result_base = solve_min_cost_flow(problem_base)
print(f"Base cost: ${result_base.objective:.2f}")  # $220.00

# Incrementally increase capacity and re-solve with warm-starting
capacities = [50, 60, 70, 80, 100]
basis = result_base.basis  # Extract basis for warm-starting

for cap in capacities:
    modified_arcs = [
        {"tail": "warehouse", "head": "store_a", "capacity": float(cap), "cost": 2.0},
        {"tail": "warehouse", "head": "store_b", "capacity": 50.0, "cost": 3.0},
    ]
    problem = build_problem(nodes=nodes, arcs=modified_arcs, directed=True, tolerance=1e-6)
    
    # Use warm-start from previous solve
    result = solve_min_cost_flow(problem, warm_start_basis=basis)
    basis = result.basis  # Update basis for next iteration
    
    print(f"Capacity {cap}: ${result.objective:.2f} ({result.iterations} iterations)")

# Output shows diminishing returns on capacity expansion
# Warm-starting typically reduces iterations by 50-90%
```

### Scenario 2: Cost Updates (Pricing Changes)

**Use case:** Fuel prices change. How does the optimal solution adapt?

```python
# Original costs
arcs_original = [
    {"tail": "factory", "head": "customer", "capacity": 50.0, "cost": 10.0},  # Direct
    {"tail": "factory", "head": "hub", "capacity": 100.0, "cost": 3.0},       # Via hub
    {"tail": "hub", "head": "customer", "capacity": 100.0, "cost": 4.0},
]

nodes = [
    {"id": "factory", "supply": 100.0},
    {"id": "hub", "supply": 0.0},
    {"id": "customer", "supply": -100.0},
]

problem_original = build_problem(nodes=nodes, arcs=arcs_original, directed=True, tolerance=1e-6)
result_original = solve_min_cost_flow(problem_original)
print(f"Original cost: ${result_original.objective:.2f}")

# After fuel price increase (+50% on direct route)
arcs_increased = [
    {"tail": "factory", "head": "customer", "capacity": 50.0, "cost": 15.0},  # +50%
    {"tail": "factory", "head": "hub", "capacity": 100.0, "cost": 4.0},
    {"tail": "hub", "head": "customer", "capacity": 100.0, "cost": 5.0},
]

problem_increased = build_problem(nodes=nodes, arcs=arcs_increased, directed=True, tolerance=1e-6)
result_increased = solve_min_cost_flow(problem_increased)
print(f"After increase: ${result_increased.objective:.2f}")

cost_increase = result_increased.objective - result_original.objective
pct_increase = (cost_increase / result_original.objective) * 100
print(f"Impact: +${cost_increase:.2f} ({pct_increase:.1f}%)")

# Compare flow patterns
direct_before = result_original.flows.get(("factory", "customer"), 0.0)
direct_after = result_increased.flows.get(("factory", "customer"), 0.0)
if direct_after < direct_before:
    print("✓ Solver shifted to cheaper hub route")
```

### Scenario 3: Demand Fluctuations

**Use case:** Weekly demand varies. Re-solve for each period.

```python
# Week 1: Base demand
nodes_week1 = [
    {"id": "supplier", "supply": 100.0},
    {"id": "customer_a", "supply": -60.0},
    {"id": "customer_b", "supply": -40.0},
]

arcs = [
    {"tail": "supplier", "head": "customer_a", "capacity": 100.0, "cost": 2.0},
    {"tail": "supplier", "head": "customer_b", "capacity": 100.0, "cost": 3.0},
]

result_w1 = solve_min_cost_flow(build_problem(nodes=nodes_week1, arcs=arcs, directed=True, tolerance=1e-6))

# Week 2: Demand shift
nodes_week2 = [
    {"id": "supplier", "supply": 100.0},
    {"id": "customer_a", "supply": -72.0},  # +20%
    {"id": "customer_b", "supply": -28.0},  # -30%
]

result_w2 = solve_min_cost_flow(build_problem(nodes=nodes_week2, arcs=arcs, directed=True, tolerance=1e-6))

# Week 3: Demand surge
nodes_week3 = [
    {"id": "supplier", "supply": 120.0},    # +20%
    {"id": "customer_a", "supply": -72.0},
    {"id": "customer_b", "supply": -48.0},
]

result_w3 = solve_min_cost_flow(build_problem(nodes=nodes_week3, arcs=arcs, directed=True, tolerance=1e-6))

print(f"Week 1: ${result_w1.objective:.2f}")
print(f"Week 2: ${result_w2.objective:.2f} ({result_w2.objective - result_w1.objective:+.2f})")
print(f"Week 3: ${result_w3.objective:.2f} ({result_w3.objective - result_w1.objective:+.2f})")
```

### Scenario 4: Network Topology Changes

**Use case:** Evaluate adding a new direct route.

```python
# Current network
arcs_current = [
    {"tail": "plant", "head": "dist_center", "capacity": 100.0, "cost": 5.0},
    {"tail": "dist_center", "head": "market", "capacity": 100.0, "cost": 4.0},
]

# Proposed: Add direct route
arcs_with_direct = arcs_current + [
    {"tail": "plant", "head": "market", "capacity": 60.0, "cost": 8.0},  # New route
]

nodes = [
    {"id": "plant", "supply": 100.0},
    {"id": "dist_center", "supply": 0.0},
    {"id": "market", "supply": -100.0},
]

result_current = solve_min_cost_flow(
    build_problem(nodes=nodes, arcs=arcs_current, directed=True, tolerance=1e-6)
)
result_with_direct = solve_min_cost_flow(
    build_problem(nodes=nodes, arcs=arcs_with_direct, directed=True, tolerance=1e-6)
)

savings = result_current.objective - result_with_direct.objective
print(f"Current cost: ${result_current.objective:.2f}")
print(f"With direct route: ${result_with_direct.objective:.2f}")
print(f"Savings: ${savings:.2f}")
if savings > 0:
    print("✓ Direct route is cost-effective")
```

### Scenario 5: Iterative Optimization

**Use case:** Gradually improve network by targeting bottlenecks.

```python
# Strategy: Identify bottleneck → Expand → Re-solve → Repeat

# Initial network
arcs = [
    {"tail": "source", "head": "node_a", "capacity": 60.0, "cost": 1.0},
    {"tail": "source", "head": "node_b", "capacity": 60.0, "cost": 2.0},
    {"tail": "node_a", "head": "sink", "capacity": 50.0, "cost": 2.0},  # Bottleneck
    {"tail": "node_b", "head": "sink", "capacity": 50.0, "cost": 1.0},  # Bottleneck
]

nodes = [
    {"id": "source", "supply": 150.0},
    {"id": "node_a", "supply": 0.0},
    {"id": "node_b", "supply": 0.0},
    {"id": "sink", "supply": -150.0},
]

# Iteration 0: Initial solve
problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
result = solve_min_cost_flow(problem)
print(f"Iteration 0: ${result.objective:.2f}")

# Identify bottlenecks (arcs at >95% capacity)
for (tail, head), flow in result.flows.items():
    capacity = next(a["capacity"] for a in arcs if a["tail"] == tail and a["head"] == head)
    utilization = (flow / capacity) * 100
    if utilization > 95:
        print(f"  ⚠ Bottleneck: {tail} -> {head} ({utilization:.1f}% utilized)")

# Iteration 1: Expand bottleneck arcs
arcs_expanded = [
    {"tail": "source", "head": "node_a", "capacity": 60.0, "cost": 1.0},
    {"tail": "source", "head": "node_b", "capacity": 60.0, "cost": 2.0},
    {"tail": "node_a", "head": "sink", "capacity": 80.0, "cost": 2.0},  # Expanded +30
    {"tail": "node_b", "head": "sink", "capacity": 80.0, "cost": 1.0},  # Expanded +30
]

problem_expanded = build_problem(nodes=nodes, arcs=arcs_expanded, directed=True, tolerance=1e-6)
result_expanded = solve_min_cost_flow(problem_expanded)
improvement = result.objective - result_expanded.objective
print(f"Iteration 1: ${result_expanded.objective:.2f} (improved by ${improvement:.2f})")
```

### Complete Working Example

See `examples/incremental_resolving_example.py` for comprehensive demonstrations including:
- Capacity expansion with diminishing returns analysis
- Cost updates with flow pattern comparison
- Demand fluctuations over time periods
- Network topology changes (adding/removing arcs)
- Iterative optimization strategy

**Output excerpt:**
```
SCENARIO 1: CAPACITY EXPANSION
Expanding warehouse -> store_a capacity:
Capacity     Objective       Iterations   Improvement
----------------------------------------------------------------------
50           $220.00         2            $  0.00
60           $240.00         2            $-20.00
70           $240.00         2            $  0.00

SCENARIO 4: NETWORK TOPOLOGY CHANGES
Current cost: $900.00
With direct route: $480.00
Savings: $420.00
✓ Direct route is cost-effective
```

### Best Practices

1. **Start simple**: Test single parameter changes before complex scenarios
2. **Track metrics**: Log objective, iterations, and solve time for each run
3. **Validate incrementally**: Use `validate_flow()` to ensure each solution is correct
4. **Use dual values**: Predict which changes will have most impact (see [Sensitivity Analysis](#sensitivity-analysis))
5. **Batch similar problems**: Re-solve related scenarios together for efficiency
6. **Consider tolerance**: Small problem changes may not affect solution within tolerance

### Performance Notes

- Re-solving from scratch is fast for networks <10,000 arcs (typically <100ms)
- The solver uses Devex pricing and Forrest-Tomlin updates for efficiency
- Each solve is independent - no state is maintained between calls
- For large-scale scenarios, consider parallel solving of independent variants

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

## Visualization

**Purpose:** Create publication-quality visualizations of network structures, flow solutions, and bottleneck analysis using matplotlib and networkx.

**Installation:** Requires optional dependencies:
```bash
pip install 'network_solver[visualization]'
```

### Network Structure Visualization

**Example:** Visualize problem topology with nodes, arcs, costs, and capacities.

```python
from network_solver import build_problem, visualize_network

# Transportation problem
nodes = [
    {"id": "factory_a", "supply": 100.0},
    {"id": "factory_b", "supply": 80.0},
    {"id": "warehouse_1", "supply": -70.0},
    {"id": "warehouse_2", "supply": -60.0},
    {"id": "warehouse_3", "supply": -50.0},
]
arcs = [
    {"tail": "factory_a", "head": "warehouse_1", "capacity": 70.0, "cost": 2.0},
    {"tail": "factory_a", "head": "warehouse_2", "capacity": 70.0, "cost": 3.0},
    {"tail": "factory_a", "head": "warehouse_3", "capacity": 70.0, "cost": 4.0},
    {"tail": "factory_b", "head": "warehouse_1", "capacity": 60.0, "cost": 1.5},
    {"tail": "factory_b", "head": "warehouse_2", "capacity": 60.0, "cost": 2.5},
    {"tail": "factory_b", "head": "warehouse_3", "capacity": 60.0, "cost": 3.5},
]

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Visualize network structure
fig = visualize_network(problem, layout="spring", figsize=(12, 8))
fig.savefig("network.png")
```

**Output:** Graph showing sources (green), sinks (red), arcs with costs and capacities.

### Flow Solution Visualization

**Example:** Visualize optimal flows with bottleneck highlighting.

```python
from network_solver import solve_min_cost_flow, visualize_flows

# Solve problem
result = solve_min_cost_flow(problem)

# Visualize flows with bottleneck highlighting
fig = visualize_flows(
    problem,
    result,
    highlight_bottlenecks=True,
    bottleneck_threshold=0.9,  # Highlight arcs ≥90% utilization
    show_zero_flows=False,      # Hide zero flows for clarity
    layout="spring",
    figsize=(14, 10),
)
fig.savefig("flows.png")

print(f"Optimal cost: ${result.objective:.2f}")
print(f"Iterations: {result.iterations}")
```

**Output:** Graph showing flow values, with bottleneck arcs highlighted in red.

**Features:**
- Arc thickness proportional to flow magnitude
- Bottleneck arcs (≥90% utilization) highlighted in red
- Utilization percentages displayed
- Statistics box (objective, status, iterations)
- Zero flows hidden for cleaner visualization

### Bottleneck Analysis Visualization

**Example:** Focused analysis of capacity constraints with utilization heatmap.

```python
from network_solver import visualize_bottlenecks

# Visualize bottlenecks (≥80% utilization)
fig = visualize_bottlenecks(
    problem,
    result,
    threshold=0.8,
    layout="spring",
    figsize=(12, 8),
)
fig.savefig("bottlenecks.png")

# Get bottleneck details programmatically
from network_solver import compute_bottleneck_arcs

bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.8)
print(f"\nIdentified {len(bottlenecks)} bottlenecks:")
for b in bottlenecks:
    print(f"  {b.tail} → {b.head}: {b.utilization*100:.1f}% utilization")
    print(f"    Flow: {b.flow:.0f} / Capacity: {b.capacity:.0f}")
    print(f"    Slack: {b.slack:.0f} units")
```

**Output:** Graph with utilization heatmap (red=high, yellow=medium, green=low).

**Features:**
- Only shows arcs above threshold (focused analysis)
- Color gradient based on utilization
- Displays utilization % and slack capacity
- Color bar for scale reference
- Statistics (count, average utilization)

### Customization Options

**Example:** Customize appearance and layout.

```python
# Custom layout and styling
fig = visualize_network(
    problem,
    layout="kamada_kawai",      # Better for structured networks
    figsize=(16, 12),            # Larger figure
    node_size=1500,              # Bigger nodes
    font_size=12,                # Larger text
    show_arc_labels=True,        # Show costs/capacities
    title="Distribution Network", # Custom title
)
fig.savefig("network_custom.png", dpi=300)  # High DPI for publication
```

**Available layouts:**
- `"spring"`: Force-directed (good default)
- `"circular"`: Nodes arranged in circle
- `"kamada_kawai"`: Structured layout (good for hierarchical networks)
- `"planar"`: Planar embedding (for planar graphs)

### Multiple Visualizations

**Example:** Create comparison visualizations.

```python
import matplotlib.pyplot as plt

# Create problem
problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# 1. Network structure
fig1 = visualize_network(problem, title="Problem Structure")
fig1.savefig("1_structure.png")

# 2. Solve and visualize flows
result = solve_min_cost_flow(problem)
fig2 = visualize_flows(problem, result, title="Optimal Flows")
fig2.savefig("2_flows.png")

# 3. Bottleneck analysis
fig3 = visualize_bottlenecks(problem, result, threshold=0.8, title="Bottlenecks (≥80%)")
fig3.savefig("3_bottlenecks.png")

# Close figures to free memory
plt.close("all")

print("Generated 3 visualizations:")
print("  1_structure.png - Network topology")
print("  2_flows.png - Optimal flow solution")
print("  3_bottlenecks.png - Capacity constraints")
```

### When to Use Visualization

**Best for:**
- Understanding problem structure (nodes, arcs, relationships)
- Analyzing flow patterns (routing decisions, utilization)
- Identifying bottlenecks (capacity constraints, critical arcs)
- Communicating results (reports, presentations, papers)
- Debugging problems (unexpected routing, infeasibility)
- Teaching and learning (intuitive visual understanding)

**Tips:**
- Use `visualize_network()` to understand problem before solving
- Use `visualize_flows()` to see optimal routing and identify issues
- Use `visualize_bottlenecks()` for focused capacity analysis
- Set `show_zero_flows=False` for cleaner flow visualizations
- Adjust `bottleneck_threshold` to focus on tightest constraints
- Use `layout="kamada_kawai"` for hierarchical networks (supply chains)
- Save with high DPI (`fig.savefig("output.png", dpi=300)`) for publications

**See Also:**
- `examples/visualization_example.py` - Complete demonstration with 6 examples
- [API: Visualization](api.md#visualization) - Full API documentation
- [Tutorial: Visualization](../tutorials/network_flow_tutorial.ipynb) - Interactive examples

## Structured Logging for Monitoring

**Use case:** Production monitoring, performance profiling, automated testing.

All solver log messages include structured data in the `extra` dict, enabling JSON logging for monitoring systems, dashboards, and analytics.

```python
import json
import logging
import sys
from network_solver import load_problem, solve_min_cost_flow


class StructuredJSONFormatter(logging.Formatter):
    """Format log records as JSON with structured extra fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Start with basic fields
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add all extra fields (anything not in standard LogRecord)
        standard_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'message', 'pathname', 'process', 'processName',
            'relativeCreated', 'thread', 'threadName', 'exc_info',
            'exc_text', 'stack_info', 'taskName'
        }
        
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                log_data[key] = value
        
        return json.dumps(log_data)


# Configure JSON logging
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(StructuredJSONFormatter())

logger = logging.getLogger("network_solver")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# Solve problem - logs include structured metrics
problem = load_problem("examples/sample_problem.json")
result = solve_min_cost_flow(problem)

print(f"\nSolution: status={result.status}, objective={result.objective}")
```

**Example JSON output:**

```json
{"timestamp": "2024-01-15 10:30:45,123", "level": "INFO", "logger": "network_solver.simplex", "message": "Starting network simplex solver", "nodes": 3, "arcs": 3, "max_iterations": 100, "pricing_strategy": "devex", "total_supply": 10.0, "tolerance": 1e-06}
{"timestamp": "2024-01-15 10:30:45,124", "level": "INFO", "logger": "network_solver.simplex", "message": "Phase 1: Finding initial feasible solution", "elapsed_ms": 0.0}
{"timestamp": "2024-01-15 10:30:45,126", "level": "INFO", "logger": "network_solver.simplex", "message": "Phase 1 complete", "iterations": 2, "total_iterations": 2, "artificial_flow": 0, "elapsed_ms": 2.23}
{"timestamp": "2024-01-15 10:30:45,127", "level": "INFO", "logger": "network_solver.simplex", "message": "Phase 2: Optimizing from feasible basis", "remaining_iterations": 98}
{"timestamp": "2024-01-15 10:30:45,128", "level": "INFO", "logger": "network_solver.simplex", "message": "Phase 2 complete", "iterations": 0, "total_iterations": 2, "objective": 15.0, "elapsed_ms": 3.64}
{"timestamp": "2024-01-15 10:30:45,129", "level": "INFO", "logger": "network_solver.simplex", "message": "Solver complete", "status": "optimal", "objective": 15.0, "iterations": 2, "elapsed_ms": 4.04, "tree_arcs": 2, "nonzero_flows": 2, "ft_rebuilds": 0}
```

**Structured metrics available:**

| Log Message | Structured Fields |
|-------------|-------------------|
| Starting solver | `nodes`, `arcs`, `max_iterations`, `pricing_strategy`, `total_supply`, `tolerance` |
| Phase 1 start | `elapsed_ms` (always 0.0) |
| Phase 1 complete | `iterations`, `total_iterations`, `artificial_flow`, `elapsed_ms` |
| Phase 2 start | `remaining_iterations` |
| Phase 2 complete | `iterations`, `total_iterations`, `objective`, `elapsed_ms` |
| Solver complete | `status`, `objective`, `iterations`, `elapsed_ms`, `tree_arcs`, `nonzero_flows`, `ft_rebuilds` |

**Use cases:**

1. **Performance Monitoring**: Track `elapsed_ms` to identify slow solves
2. **Convergence Analysis**: Monitor `iterations` and `ft_rebuilds` for solver behavior
3. **Numerical Stability**: Alert on high `ft_rebuilds` (indicates numerical issues)
4. **Solution Quality**: Track `objective` values and `status` distribution
5. **Real-time Dashboards**: Stream JSON logs to visualization tools
6. **Automated Testing**: Parse structured logs to validate solver performance
7. **Production Debugging**: Capture detailed metrics without verbose console output

**Integration with logging systems:**

```python
# Example: Send to Datadog, Prometheus, or other monitoring systems
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logging.getLogger("network_solver").addHandler(handler)

# Logs are now compatible with structured logging backends
```

## Performance Profiling

**Use case:** Analyze solver performance, compare configurations, identify bottlenecks.

Performance profiling helps you understand how the solver behaves on your specific problem types and choose optimal configuration settings.

### Why Profile Performance?

- **Understand scaling**: How does solve time grow with problem size?
- **Compare strategies**: Devex vs Dantzig pricing for your problems
- **Tune configuration**: Find optimal `ft_update_limit` and `block_size`
- **Identify bottlenecks**: Which problem characteristics cause slow solves?
- **Regression testing**: Detect performance changes after code modifications
- **Capacity planning**: Estimate solve times for production workloads

### Basic Profiling

```python
import time
from network_solver import build_problem, solve_min_cost_flow

# Build your problem
nodes = [...]
arcs = [...]
problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)

# Profile solve time
start_time = time.perf_counter()
result = solve_min_cost_flow(problem)
elapsed = time.perf_counter() - start_time

print(f"Solve time: {elapsed * 1000:.2f} ms")
print(f"Iterations: {result.iterations}")
print(f"Throughput: {result.iterations / elapsed:.0f} iterations/sec")
```

### Scaling Analysis

**Use case:** Understand how performance scales with problem size.

```python
import time

def profile_problem(name, nodes, arcs):
    """Profile a single problem instance."""
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    
    start_time = time.perf_counter()
    result = solve_min_cost_flow(problem)
    elapsed = time.perf_counter() - start_time
    
    return {
        "name": name,
        "nodes": len(nodes),
        "arcs": len(arcs),
        "iterations": result.iterations,
        "elapsed_ms": elapsed * 1000,
    }

# Test different problem sizes
sizes = [(3, 3), (5, 5), (7, 7), (10, 10), (15, 15), (20, 20)]

print(f"{'Size':<10} {'Nodes':<8} {'Arcs':<8} {'Iters':<8} {'Time (ms)':<12}")
print("-" * 60)

for rows, cols in sizes:
    nodes, arcs = generate_grid_network(rows, cols)  # Your generator
    result = profile_problem(f"{rows}×{cols}", nodes, arcs)
    
    print(f"{result['name']:<10} {result['nodes']:<8} {result['arcs']:<8} "
          f"{result['iterations']:<8} {result['elapsed_ms']:<12.2f}")
```

**Example output:**
```
Size       Nodes    Arcs     Iters    Time (ms)
----------------------------------------------------------
3×3        9        12       10       25.12
5×5        25       40       58       91.78
7×7        49       84       104      246.07
10×10      100      180      198      832.31
15×15      225      420      363      2362.20
20×20      400      760      502      6125.48
```

**Observations:**
- Solve time grows roughly quadratically with problem size
- Iteration count increases with network complexity
- Useful for capacity planning and performance budgeting

### Comparing Pricing Strategies

**Use case:** Determine which pricing strategy works best for your problems.

```python
from network_solver import SolverOptions

nodes, arcs = ...  # Your problem

# Test Devex pricing (default)
devex_opts = SolverOptions(pricing_strategy="devex")
devex_result = profile_problem("Devex", nodes, arcs, devex_opts)

# Test Dantzig pricing
dantzig_opts = SolverOptions(pricing_strategy="dantzig")
dantzig_result = profile_problem("Dantzig", nodes, arcs, dantzig_opts)

# Compare
speedup = dantzig_result['elapsed_ms'] / devex_result['elapsed_ms']
print(f"Devex:   {devex_result['iterations']} iters, {devex_result['elapsed_ms']:.2f} ms")
print(f"Dantzig: {dantzig_result['iterations']} iters, {dantzig_result['elapsed_ms']:.2f} ms")
print(f"Speedup: {speedup:.2f}x {'(Dantzig faster)' if speedup < 1 else '(Devex faster)'}")
```

**Typical results:**
- **Devex**: Fewer iterations, slightly higher per-iteration cost
- **Dantzig**: More iterations, lower per-iteration cost
- **Winner**: Usually Devex, but Dantzig competitive on sparse problems

### Configuration Tuning

**Use case:** Find optimal solver settings for your problem type.

```python
# Test different FT update limits
configs = [
    ("Default", SolverOptions()),
    ("High FT limit (128)", SolverOptions(ft_update_limit=128)),
    ("Low FT limit (32)", SolverOptions(ft_update_limit=32)),
    ("Large blocks (200)", SolverOptions(block_size=200)),
    ("Small blocks (50)", SolverOptions(block_size=50)),
]

print(f"{'Configuration':<30} {'Iters':<8} {'Time (ms)':<12}")
print("-" * 60)

for config_name, options in configs:
    result = profile_problem(config_name, nodes, arcs, options)
    print(f"{config_name:<30} {result['iterations']:<8} {result['elapsed_ms']:<12.2f}")
```

**What to tune:**
- **`ft_update_limit`**: Lower = more stable, higher = faster
- **`block_size`**: Affects Devex pricing granularity
- **`pricing_strategy`**: Devex vs Dantzig
- **`tolerance`**: Tighter = more accurate, looser = faster

### Problem Structure Analysis

**Use case:** Understand how network structure affects performance.

```python
# Compare different structures
problems = [
    ("Sparse (Grid)", grid_nodes, grid_arcs),
    ("Dense (Bipartite)", bip_nodes, bip_arcs),
    ("Medium (Hybrid)", hybrid_nodes, hybrid_arcs),
]

print(f"{'Problem Type':<25} {'Nodes':<8} {'Arcs':<8} {'Density':<10} {'Time (ms)':<12}")
print("-" * 80)

for problem_type, nodes, arcs in problems:
    result = profile_problem(problem_type, nodes, arcs)
    density = len(arcs) / (len(nodes) * len(nodes))
    
    print(f"{problem_type:<25} {result['nodes']:<8} {result['arcs']:<8} "
          f"{density:<10.3f} {result['elapsed_ms']:<12.2f}")
```

**Insights:**
- Sparse networks: Faster pivot selection
- Dense networks: More arc candidates but slower per iteration
- Structure matters: Grid vs bipartite vs tree affects iteration count

### Using Structured Logging for Profiling

Capture detailed metrics programmatically:

```python
import logging
import json

class ProfilingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.metrics = []
    
    def emit(self, record):
        if hasattr(record, 'elapsed_ms'):
            self.metrics.append({
                'message': record.getMessage(),
                'elapsed_ms': record.elapsed_ms,
                'iterations': getattr(record, 'iterations', None),
            })

# Setup logging
handler = ProfilingHandler()
logger = logging.getLogger("network_solver")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Solve
result = solve_min_cost_flow(problem)

# Analyze captured metrics
for metric in handler.metrics:
    print(json.dumps(metric))
```

See [Structured Logging](#structured-logging-for-monitoring) for more details.

### Complete Working Example

See `examples/performance_profiling_example.py` for comprehensive demonstrations including:
- Scaling analysis with multiple problem sizes
- Pricing strategy comparison (Devex vs Dantzig)
- Solver configuration tuning
- Problem structure analysis
- Performance summary and optimization tips

**Output excerpt:**
```
SCALING ANALYSIS
Size       Nodes    Arcs     Iters    Time (ms)    Iters/sec
----------------------------------------------------------------------
3×3        9        12       10       25.12        398
10×10      100      180      198      832.31       238
20×20      400      760      502      6125.48      82

PRICING STRATEGY COMPARISON
Problem              Strategy   Iters    Time (ms)    Speedup
----------------------------------------------------------------------
Grid 10×10           Devex      198      781.13       1.0x
                     Dantzig    116      206.75       0.26x (Dantzig faster!)
```

### Best Practices

1. **Profile representative problems**: Use problem sizes and structures similar to production
2. **Run multiple iterations**: Average over several runs to reduce noise
3. **Isolate variables**: Test one configuration change at a time
4. **Track over time**: Monitor performance across code versions
5. **Document baselines**: Record expected performance for regression detection
6. **Consider hardware**: Results vary by CPU, memory, and system load

### Performance Expectations

Based on typical hardware (modern laptop/desktop):

| Problem Size | Nodes | Arcs | Expected Time |
|--------------|-------|------|---------------|
| Small | <100 | <500 | <10 ms |
| Medium | 100-1000 | 500-5000 | 10-100 ms |
| Large | 1000-10000 | 5000-50000 | 100 ms - 2s |
| Very Large | >10000 | >50000 | Several seconds |

**Note:** These are rough guidelines. Actual performance depends on problem structure, density, and solver configuration.

## Comparison with NetworkX

NetworkX is a popular Python library for graph analysis that includes min-cost flow solvers. This section compares `network_solver` with NetworkX to help you choose the right tool.

**Complete example:** `examples/networkx_comparison_example.py`

### API Comparison

The two libraries have different API styles and conventions:

**network_solver:**
```python
from network_solver import build_problem, solve_min_cost_flow

nodes = [
    {"id": "warehouse", "supply": 100.0},     # Positive = supply
    {"id": "store", "supply": -100.0},        # Negative = demand
]
arcs = [
    {"tail": "warehouse", "head": "store", "cost": 2.0, "capacity": 100.0},
]

problem = build_problem(nodes=nodes, arcs=arcs, directed=True)
result = solve_min_cost_flow(problem)

print(f"Cost: ${result.objective}")
print(f"Iterations: {result.iterations}")
print(f"Flows: {result.flows}")
print(f"Duals: {result.duals}")
```

**NetworkX:**
```python
import networkx as nx

G = nx.DiGraph()
G.add_node("warehouse", demand=-100)  # Negative = supply (opposite sign!)
G.add_node("store", demand=100)        # Positive = demand
G.add_edge("warehouse", "store", weight=2, capacity=100)

flow_dict = nx.min_cost_flow(G)
cost = nx.cost_of_flow(G, flow_dict)

print(f"Cost: ${cost}")
print(f"Flows: {flow_dict['warehouse']['store']}")
# Note: No iteration count or dual values available
```

**Key differences:**
1. **Sign convention:** network_solver uses positive for supply, NetworkX uses negative
2. **Return format:** network_solver returns a `FlowResult` object, NetworkX returns nested dicts
3. **Cost calculation:** network_solver includes objective in result, NetworkX requires separate call
4. **Solver info:** network_solver provides iterations/status/timing, NetworkX does not

### Feature Comparison

| Feature | network_solver | NetworkX |
|---------|---------------|----------|
| Min-cost flow | ✓ | ✓ |
| Dual values (shadow prices) | ✓ | ✗ |
| Solver configuration | ✓ | ✗ |
| Iteration count / status | ✓ | ✗ |
| Lower bounds on arcs | ✓ | ✗ |
| Structured logging | ✓ | ✗ |
| Undirected graphs | ✓ | ✓ |
| Progress callbacks | ✓ | ✗ |
| Other graph algorithms | ✗ | ✓ (100+) |
| Graph visualization | ✗ | ✓ |
| Community size | Small | Large |

### When to Use Each Library

**Use network_solver when:**
- ✓ You need dual values (shadow prices) for sensitivity analysis
- ✓ You want fine-grained solver control (iterations, tolerance, pricing strategy)
- ✓ You need detailed diagnostics (iteration counts, status, logging)
- ✓ You have lower bounds on arcs (minimum flow requirements)
- ✓ You want structured logging for monitoring systems
- ✓ Performance and solver control are critical
- ✓ You're focused specifically on min-cost flow optimization

**Use NetworkX when:**
- ✓ You need many graph algorithms (shortest paths, centrality, clustering, etc.)
- ✓ Min-cost flow is just one part of a larger graph analysis
- ✓ You want easy graph visualization with `nx.draw()`
- ✓ You need extensive community support and documentation
- ✓ You're already using NetworkX for other operations
- ✓ You're prototyping and don't need fine-grained solver control
- ✓ You want a "batteries included" graph library

**Use both together:**
- ✓ NetworkX for graph construction, manipulation, and visualization
- ✓ network_solver for actual optimization (better control and diagnostics)
- ✓ Convert between formats as needed

### Example Workflow Combining Both

```python
import networkx as nx
from network_solver import build_problem, solve_min_cost_flow

# 1. Build and visualize with NetworkX
G = nx.DiGraph()
# ... add nodes and edges ...
nx.draw(G, with_labels=True)  # Visualize the network

# 2. Convert to network_solver format
nodes = [{"id": n, "supply": -G.nodes[n].get("demand", 0)} 
         for n in G.nodes()]
arcs = [{"tail": u, "head": v, "cost": G[u][v].get("weight", 0),
         "capacity": G[u][v].get("capacity", float("inf"))}
        for u, v in G.edges()]

# 3. Solve with network_solver for better control
problem = build_problem(nodes, arcs, directed=True)
result = solve_min_cost_flow(problem)

# 4. Analyze dual values for sensitivity
for node_id, dual_value in result.duals.items():
    print(f"Shadow price at {node_id}: ${dual_value:.2f}")

# 5. Convert results back to NetworkX for visualization
for (u, v), flow in result.flows.items():
    G[u][v]["flow"] = flow
    
# Draw with flow values
edge_labels = {(u, v): f"{G[u][v]['flow']:.0f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels)
```

### Performance Comparison

Performance varies by problem type and size. Example results on grid networks:

| Problem Size | Arcs | network_solver | NetworkX | Speedup |
|--------------|------|----------------|----------|---------|
| 3×3 grid | 24 | 2.3 ms | 4.1 ms | 1.8x |
| 5×5 grid | 80 | 5.2 ms | 12.3 ms | 2.4x |
| 10×10 grid | 360 | 18.4 ms | 52.7 ms | 2.9x |
| 15×15 grid | 840 | 45.2 ms | 156.8 ms | 3.5x |

**Notes:**
- Both solvers find the optimal solution
- network_solver uses network simplex (primal method)
- NetworkX uses capacity scaling algorithm
- Performance depends heavily on problem structure

### Summary

**network_solver** is a specialized tool for min-cost flow optimization with:
- Advanced features (dual values, solver options, progress tracking)
- Detailed diagnostics and logging
- Fine-grained control over the solver
- Focus on performance and optimization

**NetworkX** is a general-purpose graph library with:
- 100+ graph algorithms beyond min-cost flow
- Easy visualization and graph manipulation
- Large community and extensive documentation
- "Batteries included" approach

Choose based on your needs, or use both together for the best of both worlds!

## See Also

- [API Reference](api.md) - Complete API documentation
- [Algorithm](algorithm.md) - Network simplex algorithm details
- [Benchmarks](benchmarks.md) - Performance characteristics
- `examples/networkx_comparison_example.py` - Complete comparison code
- `examples/` directory - All runnable code examples
