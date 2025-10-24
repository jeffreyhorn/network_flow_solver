# API Reference

Complete API documentation for the network flow solver library.

## Table of Contents

- [Main Functions](#main-functions)
- [Problem Preprocessing](#problem-preprocessing)
- [Problem Definition](#problem-definition)
- [Solver Configuration](#solver-configuration)
- [Results and Analysis](#results-and-analysis)
- [Utility Functions](#utility-functions)
- [Visualization](#visualization)
- [Network Specializations](#network-specializations)
- [Progress Tracking](#progress-tracking)
- [Exceptions](#exceptions)
- [Input/Output](#inputoutput)

## Main Functions

### solve_min_cost_flow

```python
def solve_min_cost_flow(
    problem: NetworkProblem,
    options: SolverOptions | None = None,
    max_iterations: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 100,
) -> FlowResult
```

Solve a minimum-cost flow problem using the network simplex algorithm.

**Parameters:**

- `problem` (NetworkProblem): The network flow problem to solve
- `options` (SolverOptions, optional): Solver configuration options
- `max_iterations` (int, optional): Maximum iterations. Overrides `options.max_iterations` if provided
- `progress_callback` (ProgressCallback, optional): Callback for progress updates
- `progress_interval` (int): Iterations between progress callbacks (default: 100)

**Returns:**

- `FlowResult`: Solution containing flows, objective value, dual values, and status

**Raises:**

- `InvalidProblemError`: Problem definition is malformed
- `UnboundedProblemError`: Problem has unbounded objective
- Various solver errors (see [Exceptions](#exceptions))

**Example:**

```python
from network_solver import solve_min_cost_flow, build_problem, SolverOptions

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
options = SolverOptions(tolerance=1e-8, pricing_strategy="devex")
result = solve_min_cost_flow(problem, options=options)
```

### build_problem

```python
def build_problem(
    nodes: Iterable[dict[str, float]],
    arcs: Iterable[dict[str, float]],
    directed: bool,
    tolerance: float,
) -> NetworkProblem
```

Construct a NetworkProblem from node and arc dictionaries.

**Parameters:**

- `nodes` (Iterable[dict]): Node definitions with `id` and optional `supply`
  ```python
  {"id": "node1", "supply": 10.0}  # Supply node
  {"id": "node2", "supply": -5.0}  # Demand node
  ```
- `arcs` (Iterable[dict]): Arc definitions with `tail`, `head`, optional `capacity`, `cost`, `lower`
  ```python
  {"tail": "node1", "head": "node2", "capacity": 100.0, "cost": 2.5, "lower": 0.0}
  ```
- `directed` (bool): Whether the graph is directed
- `tolerance` (float): Numerical tolerance for feasibility checks

**Returns:**

- `NetworkProblem`: Validated problem instance

**Raises:**

- `InvalidProblemError`: If problem is malformed (unbalanced, invalid arcs, etc.)

**Example:**

```python
nodes = [
    {"id": "source", "supply": 100.0},
    {"id": "sink", "supply": -100.0},
]
arcs = [
    {"tail": "source", "head": "sink", "capacity": 150.0, "cost": 1.0},
]
problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
```

## Problem Preprocessing

Functions for simplifying network flow problems before solving by removing redundancies and merging arcs.

### preprocess_problem

```python
def preprocess_problem(
    problem: NetworkProblem,
    remove_redundant: bool = True,
    detect_disconnected: bool = True,
    simplify_series: bool = True,
    remove_zero_supply: bool = True,
) -> PreprocessingResult
```

Preprocess a network flow problem to reduce size and improve solving performance.

**Parameters:**

- `problem` (NetworkProblem): The network flow problem to preprocess
- `remove_redundant` (bool): Remove redundant parallel arcs (default: True)
- `detect_disconnected` (bool): Detect disconnected components (default: True)
- `simplify_series` (bool): Simplify series arcs (default: True)
- `remove_zero_supply` (bool): Remove zero-supply single-arc nodes (default: True)

**Returns:**

- `PreprocessingResult`: Contains preprocessed problem and statistics

**Optimization Techniques:**

1. **Remove redundant arcs**: Parallel arcs with identical costs are merged (capacities combined)
2. **Detect disconnected components**: BFS-based connectivity analysis warns of potential infeasibility
3. **Simplify series arcs**: Merge consecutive arcs through zero-supply transshipment nodes
4. **Remove zero-supply nodes**: Eliminate transshipment nodes with single incident arc

**Example:**

```python
from network_solver import preprocess_problem, solve_min_cost_flow

# Preprocess problem
result = preprocess_problem(problem)
print(f"Removed {result.removed_arcs} arcs, {result.removed_nodes} nodes")
print(f"Preprocessing time: {result.preprocessing_time_ms:.2f}ms")

# Solve preprocessed problem
flow_result = solve_min_cost_flow(result.problem)
```

**Selective Preprocessing:**

```python
# Only apply specific optimizations
result = preprocess_problem(
    problem,
    remove_redundant=True,
    simplify_series=True,
    detect_disconnected=False,  # Skip connectivity check
    remove_zero_supply=False,    # Keep all nodes
)
```

**See Also:**
- `preprocess_and_solve()` - Convenience function
- `PreprocessingResult` - Result dataclass
- [Examples: Problem Preprocessing](examples.md#problem-preprocessing)

### preprocess_and_solve

```python
def preprocess_and_solve(
    problem: NetworkProblem,
    **solve_kwargs: Any
) -> tuple[PreprocessingResult, FlowResult]
```

Convenience function to preprocess, solve, and **automatically translate the solution back** to the original problem structure.

**Parameters:**

- `problem` (NetworkProblem): Network flow problem to solve
- `**solve_kwargs`: Additional arguments passed to `solve_min_cost_flow()`

**Returns:**

- `tuple[PreprocessingResult, FlowResult]`: Preprocessing statistics and flow solution
  - The `FlowResult` contains flows and duals for the **original problem** (not the preprocessed one)
  - All original arcs have flow values (including removed/merged arcs)
  - All original nodes have dual values (including removed nodes)

**Example:**

```python
from network_solver import preprocess_and_solve

# Preprocess, solve, and automatically translate solution back
preproc_result, flow_result = preprocess_and_solve(problem)

print(f"Removed {preproc_result.removed_arcs} arcs")
print(f"Optimal cost: ${flow_result.objective:.2f}")

# Access flows for original arcs (even if they were removed/merged)
print(f"Flow on original arc: {flow_result.flows[('factory', 'hub1')]}")

# Access duals for original nodes (even if they were removed)
print(f"Dual for removed node: {flow_result.duals['hub1']}")
```

**Result Translation:**

Solutions are automatically translated back to the original problem:

- **Removed arcs** → assigned zero flow
- **Redundant arcs** (merged) → flows distributed proportionally by capacity
- **Series arcs** (merged) → all arcs in series carry the same flow
- **Removed nodes** → duals computed from adjacent preserved arcs
- **Preserved arcs/nodes** → flows/duals copied directly

This ensures the solution always corresponds to your original problem structure.

**With Solver Options:**

```python
from network_solver import preprocess_and_solve, SolverOptions

options = SolverOptions(tolerance=1e-8, pricing_strategy="devex")
preproc_result, flow_result = preprocess_and_solve(problem, options=options)
```

### PreprocessingResult

```python
@dataclass
class PreprocessingResult:
    problem: NetworkProblem
    removed_arcs: int = 0
    removed_nodes: int = 0
    merged_arcs: int = 0
    redundant_arcs: int = 0
    disconnected_components: int = 0
    preprocessing_time_ms: float = 0.0
    optimizations: dict[str, int] = field(default_factory=dict)
```

Result of preprocessing a network flow problem.

**Attributes:**

- `problem` (NetworkProblem): The preprocessed problem instance
- `removed_arcs` (int): Number of arcs removed during preprocessing
- `removed_nodes` (int): Number of nodes removed during preprocessing
- `merged_arcs` (int): Number of arc series that were merged
- `redundant_arcs` (int): Number of redundant parallel arcs removed
- `disconnected_components` (int): Number of disconnected components detected
- `preprocessing_time_ms` (float): Time spent preprocessing in milliseconds
- `optimizations` (dict[str, int]): Detailed breakdown by optimization type

**Example:**

```python
result = preprocess_problem(problem)

print(f"Statistics:")
print(f"  Removed {result.removed_arcs} arcs")
print(f"  Removed {result.removed_nodes} nodes")
print(f"  Merged {result.merged_arcs} series arcs")
print(f"  Found {result.redundant_arcs} redundant arcs")
print(f"  Detected {result.disconnected_components} components")
print(f"  Time: {result.preprocessing_time_ms:.2f}ms")

print(f"\nDetailed optimizations:")
for opt_name, count in result.optimizations.items():
    print(f"  {opt_name}: {count}")
```

**Optimizations Dictionary Keys:**

- `"redundant_arcs_removed"`: Count of parallel arcs merged
- `"disconnected_components"`: Number of separate components
- `"series_arcs_merged"`: Count of series arcs merged
- `"series_nodes_removed"`: Count of transshipment nodes removed
- `"zero_supply_nodes_removed"`: Count of single-arc nodes removed

## Problem Definition

### NetworkProblem

```python
@dataclass
class NetworkProblem:
    directed: bool
    nodes: dict[str, Node]
    arcs: list[Arc]
    tolerance: float = 1e-3
```

Represents a minimum-cost flow problem.

**Attributes:**

- `directed` (bool): Whether arcs are directed
- `nodes` (dict[str, Node]): Mapping from node ID to Node object
- `arcs` (list[Arc]): List of arc definitions
- `tolerance` (float): Numerical tolerance

**Methods:**

- `validate()`: Check problem validity (balance, arc endpoints)
- `undirected_expansion()`: Convert undirected edges to directed arcs

### Node

```python
@dataclass(frozen=True)
class Node:
    id: str
    supply: float = 0.0
```

Represents a network node.

**Attributes:**

- `id` (str): Unique node identifier
- `supply` (float): Supply (positive) or demand (negative). Zero for transshipment nodes.

### Arc

```python
@dataclass(frozen=True)
class Arc:
    tail: str
    head: str
    capacity: float | None
    cost: float
    lower: float = 0.0
```

Represents a directed arc.

**Attributes:**

- `tail` (str): Source node ID
- `head` (str): Destination node ID  
- `capacity` (float | None): Upper bound on flow. None for infinite capacity.
- `cost` (float): Cost per unit of flow
- `lower` (float): Lower bound on flow (default: 0)

**Constraints:**

- `tail != head` (no self-loops)
- `capacity >= lower` if capacity is not None

## Working with Undirected Graphs

The solver internally operates on directed networks, but provides first-class support for undirected graphs through automatic transformation.

### How Undirected Graphs Work

When you create a `NetworkProblem` with `directed=False`, each undirected edge is automatically transformed into a directed arc that allows bidirectional flow:

**Transformation:**
- Undirected edge: `{u, v}` with capacity `C` and cost `c`
- Becomes directed arc: `(u, v)` with:
  - `capacity = C` (upper bound)
  - `lower = -C` (lower bound, enabling reverse flow)
  - `cost = c` (symmetric cost)

**Flow Interpretation:**
- **Positive flow** `f > 0`: Flow goes from `tail → head` with magnitude `f`
- **Negative flow** `f < 0`: Flow goes from `head → tail` with magnitude `|f|`
- **Zero flow** `f = 0`: No flow on the edge

### Requirements for Undirected Edges

1. **Finite Capacity**: All edges must have finite capacity (no `capacity=None`)
   - Required because lower bound is set to `-capacity`
   - Infinite capacity would create unbounded lower bound

2. **No Custom Lower Bounds**: Leave `lower=0.0` (default)
   - Lower bound is automatically set to `-capacity` during transformation
   - Custom lower bounds will raise `InvalidProblemError`

3. **Symmetric Costs**: Cost is the same in both directions
   - If you need asymmetric costs, use directed graph instead

### Example: Creating an Undirected Problem

```python
from network_solver import Node, Arc, NetworkProblem, solve_min_cost_flow

# Undirected graph: A -- B -- C
nodes = {
    "A": Node(id="A", supply=10.0),
    "B": Node(id="B", supply=0.0),   # Transshipment node
    "C": Node(id="C", supply=-10.0),
}

# Undirected edges (note: must have finite capacity)
arcs = [
    Arc(tail="A", head="B", capacity=15.0, cost=2.0),  # A-B edge
    Arc(tail="B", head="C", capacity=15.0, cost=3.0),  # B-C edge
]

# Set directed=False for undirected graph
problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

result = solve_min_cost_flow(problem)

# Interpret results:
# result.flows[("A", "B")] = 10.0  → Flow goes A → B (positive)
# result.flows[("B", "C")] = -5.0  → Flow goes C → B (negative)
```

### Interpreting Results from Undirected Graphs

When you solve an undirected problem, the result contains flow values that may be positive or negative:

```python
result = solve_min_cost_flow(problem)

for arc in problem.arcs:
    key = (arc.tail, arc.head)
    flow = result.flows.get(key, 0.0)
    
    if flow > 0:
        print(f"Edge {arc.tail}--{arc.head}: {flow:.1f} units going {arc.tail} → {arc.head}")
    elif flow < 0:
        print(f"Edge {arc.tail}--{arc.head}: {abs(flow):.1f} units going {arc.head} → {arc.tail}")
    else:
        print(f"Edge {arc.tail}--{arc.head}: no flow")
```

### Common Errors with Undirected Graphs

**Error: Infinite capacity**

```python
# ❌ This will raise InvalidProblemError
arc = Arc(tail="A", head="B", capacity=None, cost=1.0)  # Infinite capacity
problem = NetworkProblem(directed=False, nodes=nodes, arcs=[arc])
problem.undirected_expansion()  # Error!

# ✓ Use finite capacity instead
arc = Arc(tail="A", head="B", capacity=100.0, cost=1.0)
```

**Error: Custom lower bound**

```python
# ❌ This will raise InvalidProblemError
arc = Arc(tail="A", head="B", capacity=50.0, cost=1.0, lower=5.0)
problem = NetworkProblem(directed=False, nodes=nodes, arcs=[arc])
problem.undirected_expansion()  # Error!

# ✓ Leave lower bound at default (0.0)
arc = Arc(tail="A", head="B", capacity=50.0, cost=1.0)
```

### When to Use Undirected vs Directed

**Use Undirected When:**
- Physical infrastructure is naturally bidirectional (pipes, cables, roads)
- Costs are the same in both directions
- You want simpler problem specification (one edge vs two arcs)
- Capacities are symmetric

**Use Directed When:**
- Flow has inherent direction (assembly lines, dependencies)
- Costs differ by direction (uphill vs downhill transport)
- Capacities differ by direction
- You need asymmetric lower bounds

**Directed Alternative:**

Instead of one undirected edge, you can manually create two directed arcs:

```python
# Undirected: 1 edge
Arc(tail="A", head="B", capacity=50.0, cost=2.0)  # In undirected graph

# Equivalent directed: 2 arcs
Arc(tail="A", head="B", capacity=50.0, cost=2.0)  # A → B
Arc(tail="B", head="A", capacity=50.0, cost=2.0)  # B → A
```

The undirected approach is more compact and ensures symmetric behavior automatically.

### Advanced: Internal Transformation Details

The `undirected_expansion()` method is called internally during problem setup:

```python
problem = NetworkProblem(directed=False, nodes=nodes, arcs=arcs)

# Manually inspect transformation
expanded_arcs = problem.undirected_expansion()

for orig, expanded in zip(arcs, expanded_arcs):
    print(f"Original:  {orig.tail}--{orig.head}, cap={orig.capacity}, lower={orig.lower}")
    print(f"Expanded:  {expanded.tail}→{expanded.head}, cap={expanded.capacity}, lower={expanded.lower}")
    # Output:
    # Original:  A--B, cap=50.0, lower=0.0
    # Expanded:  A→B, cap=50.0, lower=-50.0
```

This transformation preserves problem semantics while allowing the network simplex algorithm to work on a standard directed network.

## Solver Configuration

### SolverOptions

```python
@dataclass
class SolverOptions:
    max_iterations: int | None = None
    tolerance: float = 1e-6
    pricing_strategy: str = "devex"
    block_size: int | str | None = None
    ft_update_limit: int = 64
    projection_cache_size: int = 100
    auto_scale: bool = True
    adaptive_refactorization: bool = True
    condition_number_threshold: float = 1e12
    adaptive_ft_min: int = 20
    adaptive_ft_max: int = 200
```

Configuration options for the solver.

**Attributes:**

- `max_iterations` (int, optional): Maximum simplex iterations. Default: `max(100, 5*num_arcs)`
- `tolerance` (float): Numerical tolerance for feasibility/optimality (default: 1e-6)
- `pricing_strategy` (str): Arc selection strategy
  - `"devex"` (default): Devex normalized pricing with **vectorization enabled by default** (1.8-3.1x speedup on problems with 200+ arcs)
  - `"dantzig"`: Most negative reduced cost (simpler, no vectorization)
- `use_vectorized_pricing` (bool): Enable vectorized pricing operations (default: True)
  - `True` (default): Use NumPy vectorized operations for significant speedup (recommended)
    - Small problems (300 arcs): 162% speedup (2.6x faster)
    - Medium problems (600 arcs): 92% speedup (1.9x faster)
    - Average improvement: 127% speedup (2.3x faster)
    - Includes deferred weight updates (only selected arc's weight updated per iteration)
  - `False`: Use loop-based pricing with deferred weight updates
    - Optimized to update only the selected entering arc's weight (not all examined candidates)
    - 97.5% reduction in weight update calls vs. old implementation
    - 37% faster than previous loop-based implementation
    - Useful for debugging or comparing against vectorized version
  - Only applies to Devex pricing strategy
  - **Note**: Both vectorized and loop-based modes use deferred weight updates; this option only controls vectorization
- `block_size` (int | str, optional): Arcs per pricing block
  - `None` or `"auto"` (default): Auto-tune based on problem size with runtime adaptation
  - int: Fixed block size (no adaptation)
- `ft_update_limit` (int): Forrest-Tomlin updates before refactorization (default: 64)
  - Initial limit when adaptive_refactorization=True
  - Fixed limit when adaptive_refactorization=False
- `projection_cache_size` (int): Cache size for basis projections (default: 100)
  - Optimized cache provides 10-14% speedup on medium/large problems (70+ nodes)
  - Set to 0 to disable for very small problems
  - Cache stores projection results and is cleared when basis changes
  - Memory usage: ~800 bytes per cached projection
- `auto_scale` (bool): Enable automatic problem scaling (default: True)
  - Automatically detects and scales problems with wide value ranges
  - See [Automatic Problem Scaling](../README.md#automatic-problem-scaling) for details
- `adaptive_refactorization` (bool): Enable adaptive basis refactorization (default: True)
  - Monitors condition number and adjusts refactorization frequency automatically
  - Improves numerical stability for ill-conditioned problems
- `condition_number_threshold` (float): Condition number limit for triggering rebuild (default: 1e12)
  - Lower values (1e10): More conservative, more rebuilds, better stability
  - Higher values (1e14): More aggressive, fewer rebuilds, faster but less stable
  - Only used when adaptive_refactorization=True
- `adaptive_ft_min` (int): Minimum adaptive ft_update_limit (default: 20)
  - Prevents limit from becoming too small
  - Only used when adaptive_refactorization=True
- `adaptive_ft_max` (int): Maximum adaptive ft_update_limit (default: 200)
  - Prevents limit from becoming too large
  - Only used when adaptive_refactorization=True
- `use_dense_inverse` (bool | None): Compute and maintain dense basis inverse (default: None = auto)
  - `None` (default): Auto-detect based on scipy availability
    - If scipy installed: `False` (use sparse LU for better scalability)
    - If scipy not installed: `True` (fall back to dense inverse)
  - `False`: Force sparse LU only (requires scipy, raises error if unavailable)
  - `True`: Always compute dense inverse with np.linalg.inv (O(n³) time, O(n²) memory)
  - Dense inverse enables Sherman-Morrison rank-1 updates but requires O(n²) memory
  - For problems with >1000 nodes, sparse LU (scipy) is strongly recommended
  - Dense mode is useful for testing, very dense networks, or small problems without scipy

**Validation:**

- `tolerance > 0`
- `pricing_strategy` in `{"devex", "dantzig"}`
- `block_size > 0` if provided (or `"auto"`)
- `ft_update_limit > 0`
- `condition_number_threshold > 1`
- `adaptive_ft_min > 0` and `adaptive_ft_min <= adaptive_ft_max`

**Examples:**

```python
# High-precision solve
options = SolverOptions(tolerance=1e-10)

# Dantzig pricing with fixed block size
options = SolverOptions(pricing_strategy="dantzig", block_size=10)

# Conservative refactorization for maximum stability
options = SolverOptions(
    adaptive_refactorization=True,
    condition_number_threshold=1e10,  # More aggressive rebuilding
    adaptive_ft_min=10,
    adaptive_ft_max=50,
)

# Enable dense inverse for small problems or testing
# (Not recommended for large problems due to O(n²) memory usage)
options = SolverOptions(use_dense_inverse=True)

# Disable adaptive features for predictable behavior
options = SolverOptions(
    auto_scale=False,
    adaptive_refactorization=False,
    ft_update_limit=64,  # Fixed refactorization
)

# Default settings (recommended for most users)
options = SolverOptions()  # All adaptive features enabled
```

## Results and Analysis

### FlowResult

```python
@dataclass
class FlowResult:
    objective: float
    flows: dict[tuple[str, str], float] = field(default_factory=dict)
    status: str = "optimal"
    iterations: int = 0
    duals: dict[str, float] = field(default_factory=dict)
```

Solution to a minimum-cost flow problem.

**Attributes:**

- `objective` (float): Total cost of the solution
- `flows` (dict): Mapping from `(tail, head)` to flow value
- `status` (str): Solution status
  - `"optimal"`: Optimal solution found
  - `"infeasible"`: No feasible solution exists
  - `"iteration_limit"`: Reached iteration limit
  - `"unbounded"`: Objective can decrease without bound
- `iterations` (int): Number of simplex iterations performed
- `duals` (dict): Node potentials (shadow prices) mapping node ID to dual value

**Example:**

```python
result = solve_min_cost_flow(problem)

print(f"Objective: ${result.objective:,.2f}")
print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")

for (tail, head), flow in result.flows.items():
    print(f"  {tail} -> {head}: {flow:.2f} units")

for node_id, dual in result.duals.items():
    print(f"  Node {node_id}: dual = ${dual:.2f}")
```

## Utility Functions

### extract_path

```python
def extract_path(
    result: FlowResult,
    problem: NetworkProblem,
    source: str,
    target: str,
    tolerance: float = 1e-6,
) -> FlowPath | None
```

Find a flow-carrying path from source to target using BFS.

**Parameters:**

- `result` (FlowResult): Solution containing flows
- `problem` (NetworkProblem): Problem definition (for arc costs)
- `source` (str): Starting node ID
- `target` (str): Ending node ID
- `tolerance` (float): Minimum flow to consider arc active

**Returns:**

- `FlowPath | None`: Path if found, None otherwise

**Raises:**

- `ValueError`: If source or target node doesn't exist

**Example:**

```python
path = extract_path(result, problem, "factory", "warehouse")
if path:
    print(f"Route: {' -> '.join(path.nodes)}")
    print(f"Flow: {path.flow} units")
    print(f"Cost: ${path.cost}")
```

### validate_flow

```python
def validate_flow(
    problem: NetworkProblem,
    result: FlowResult,
    tolerance: float = 1e-6,
) -> ValidationResult
```

Verify that a solution satisfies all constraints.

**Parameters:**

- `problem` (NetworkProblem): Problem definition
- `result` (FlowResult): Solution to validate
- `tolerance` (float): Numerical tolerance for violations

**Returns:**

- `ValidationResult`: Validation report

**Checks:**

1. Flow conservation at each node
2. Capacity constraints (flow ≤ capacity)
3. Lower bound constraints (flow ≥ lower)

**Example:**

```python
validation = validate_flow(problem, result)
if validation.is_valid:
    print("✓ Solution is valid")
else:
    for error in validation.errors:
        print(f"✗ {error}")
```

### compute_bottleneck_arcs

```python
def compute_bottleneck_arcs(
    problem: NetworkProblem,
    result: FlowResult,
    threshold: float = 0.95,
    tolerance: float = 1e-6,
) -> list[BottleneckArc]
```

Identify arcs at or near capacity that limit throughput.

**Parameters:**

- `problem` (NetworkProblem): Problem definition
- `result` (FlowResult): Solution to analyze
- `threshold` (float): Minimum utilization (default: 0.95 = 95%)
- `tolerance` (float): Minimum flow to consider arc active

**Returns:**

- `list[BottleneckArc]`: Bottlenecks sorted by utilization (descending)

**Example:**

```python
bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.90)
for b in bottlenecks:
    print(f"{b.tail} -> {b.head}: {b.utilization*100:.1f}% utilized")
    print(f"  Slack: {b.slack} units, Cost: ${b.cost}/unit")
```

### Utility Dataclasses

#### FlowPath

```python
@dataclass
class FlowPath:
    nodes: list[str]          # Node IDs from source to target
    arcs: list[tuple[str, str]]  # Arc tuples along path
    flow: float               # Minimum flow on path
    cost: float               # Total cost (flow * sum of arc costs)
```

#### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool                        # True if all constraints satisfied
    errors: list[str]                     # Error messages
    flow_balance: dict[str, float]        # Net flow at each node
    capacity_violations: list[tuple[str, str]]   # Arcs exceeding capacity
    lower_bound_violations: list[tuple[str, str]]  # Arcs below lower bound
```

#### BottleneckArc

```python
@dataclass
class BottleneckArc:
    tail: str                 # Source node
    head: str                 # Destination node
    flow: float               # Current flow
    capacity: float | None    # Arc capacity
    utilization: float | None # flow / capacity (None for infinite capacity)
    cost: float               # Cost per unit
    slack: float              # Remaining capacity (capacity - flow)
```

## Visualization

Functions for visualizing network structures, flow solutions, and bottleneck analysis.

**Installation:** Requires optional dependencies. Install with:
```bash
pip install 'network_solver[visualization]'
```

### visualize_network

```python
def visualize_network(
    problem: NetworkProblem,
    layout: str = "spring",
    figsize: tuple[float, float] = (12, 8),
    node_size: int = 1000,
    font_size: int = 10,
    show_arc_labels: bool = True,
    title: str | None = None,
) -> Figure
```

Visualize network structure showing nodes, arcs, supplies, and costs.

**Parameters:**

- `problem` (NetworkProblem): Network flow problem to visualize
- `layout` (str): Graph layout algorithm - "spring", "circular", "kamada_kawai", or "planar" (default: "spring")
- `figsize` (tuple[float, float]): Figure size (width, height) in inches (default: (12, 8))
- `node_size` (int): Size of node markers (default: 1000)
- `font_size` (int): Font size for labels (default: 10)
- `show_arc_labels` (bool): Whether to show cost/capacity labels on arcs (default: True)
- `title` (str | None): Custom title for the plot (default: "Network Structure")

**Returns:**

- `Figure`: matplotlib Figure object

**Raises:**

- `ImportError`: If matplotlib or networkx are not installed

**Features:**

- Automatic node categorization:
  - Sources (supply > 0): green
  - Sinks (supply < 0): red
  - Transshipment (supply = 0): lightblue
- Arc labels showing costs and capacities
- Supply/demand values displayed on nodes
- Legend for node types

**Example:**

```python
from network_solver import build_problem, visualize_network

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
fig = visualize_network(problem, layout="spring")
fig.savefig("network.png")
```

**See Also:**
- `visualize_flows()` - Visualize flow solution
- `visualize_bottlenecks()` - Visualize bottleneck analysis

### visualize_flows

```python
def visualize_flows(
    problem: NetworkProblem,
    result: FlowResult,
    layout: str = "spring",
    figsize: tuple[float, float] = (14, 10),
    node_size: int = 1200,
    font_size: int = 10,
    highlight_bottlenecks: bool = True,
    bottleneck_threshold: float = 0.9,
    show_zero_flows: bool = False,
    title: str | None = None,
) -> Figure
```

Visualize flow solution with optional bottleneck highlighting.

**Parameters:**

- `problem` (NetworkProblem): Network flow problem
- `result` (FlowResult): Flow solution from solve_min_cost_flow()
- `layout` (str): Graph layout algorithm (default: "spring")
- `figsize` (tuple[float, float]): Figure size in inches (default: (14, 10))
- `node_size` (int): Size of node markers (default: 1200)
- `font_size` (int): Font size for labels (default: 10)
- `highlight_bottlenecks` (bool): Whether to highlight high-utilization arcs (default: True)
- `bottleneck_threshold` (float): Utilization threshold for bottleneck (default: 0.9 = 90%)
- `show_zero_flows` (bool): Whether to show arcs with zero flow (default: False)
- `title` (str | None): Custom title (default: "Flow Solution (Cost: $...)")

**Returns:**

- `Figure`: matplotlib Figure object

**Raises:**

- `ImportError`: If matplotlib or networkx are not installed

**Features:**

- Flow values displayed on arcs
- Arc thickness proportional to flow magnitude
- Bottleneck arcs highlighted in red (utilization ≥ threshold)
- Utilization percentages displayed
- Statistics box showing objective, status, iterations
- Option to hide zero flows for cleaner visualization

**Example:**

```python
from network_solver import solve_min_cost_flow, visualize_flows

result = solve_min_cost_flow(problem)

# Visualize with bottleneck highlighting
fig = visualize_flows(
    problem,
    result,
    highlight_bottlenecks=True,
    bottleneck_threshold=0.9,
    show_zero_flows=False,
)
fig.savefig("flows.png")
```

**See Also:**
- `visualize_network()` - Visualize problem structure
- `visualize_bottlenecks()` - Focused bottleneck visualization
- `compute_bottleneck_arcs()` - Identify bottlenecks programmatically

### visualize_bottlenecks

```python
def visualize_bottlenecks(
    problem: NetworkProblem,
    result: FlowResult,
    threshold: float = 0.8,
    layout: str = "spring",
    figsize: tuple[float, float] = (14, 10),
    node_size: int = 1200,
    font_size: int = 10,
    title: str | None = None,
) -> Figure
```

Visualize bottleneck analysis with utilization heatmap.

**Parameters:**

- `problem` (NetworkProblem): Network flow problem
- `result` (FlowResult): Flow solution from solve_min_cost_flow()
- `threshold` (float): Minimum utilization to display (default: 0.8 = 80%)
- `layout` (str): Graph layout algorithm (default: "spring")
- `figsize` (tuple[float, float]): Figure size in inches (default: (14, 10))
- `node_size` (int): Size of node markers (default: 1200)
- `font_size` (int): Font size for labels (default: 10)
- `title` (str | None): Custom title (default: "Bottleneck Analysis (≥...% utilization)")

**Returns:**

- `Figure`: matplotlib Figure object

**Raises:**

- `ImportError`: If matplotlib or networkx are not installed

**Features:**

- Utilization heatmap with color gradient:
  - Red: high utilization (near capacity)
  - Yellow: medium utilization
  - Green: lower utilization (within threshold)
- Only shows arcs above threshold
- Displays utilization percentage and slack capacity
- Color bar for utilization scale
- Statistics box (bottleneck count, average utilization)
- Returns figure with "No bottlenecks found" message if none exist

**Example:**

```python
from network_solver import visualize_bottlenecks

# Show arcs with ≥80% utilization
fig = visualize_bottlenecks(problem, result, threshold=0.8)
fig.savefig("bottlenecks.png")
```

**See Also:**
- `visualize_flows()` - Full flow visualization
- `compute_bottleneck_arcs()` - Identify bottlenecks programmatically

## Network Specializations

The solver automatically detects special network structures and applies optimized pivot strategies.

### analyze_network_structure

```python
def analyze_network_structure(problem: NetworkProblem) -> NetworkStructure
```

Analyze a network problem to detect its structure and type.

**Parameters:**

- `problem` (NetworkProblem): Problem to analyze

**Returns:**

- `NetworkStructure`: Detected structure with classification and properties

**Algorithm:**

Uses BFS 2-coloring to detect bipartite graphs, categorizes nodes as sources/sinks/transshipment, and classifies problem type based on structure.

**Example:**

```python
from network_solver import build_problem, analyze_network_structure

problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
structure = analyze_network_structure(problem)

print(f"Network type: {structure.network_type.value}")
print(f"Is bipartite: {structure.is_bipartite}")
print(f"Sources: {len(structure.source_nodes)}")
print(f"Sinks: {len(structure.sink_nodes)}")
print(f"Transshipment nodes: {len(structure.transshipment_nodes)}")
print(f"Balanced: {structure.is_balanced}")
```

### get_specialization_info

```python
def get_specialization_info(structure: NetworkStructure) -> dict[str, Any]
```

Get human-readable information about detected specialization.

**Parameters:**

- `structure` (NetworkStructure): Analyzed network structure

**Returns:**

- `dict`: Information dictionary with keys:
  - `type` (str): Network type name
  - `description` (str): Human-readable description
  - `characteristics` (list[str]): Key characteristics
  - `is_bipartite` (bool): Whether graph is bipartite

**Example:**

```python
structure = analyze_network_structure(problem)
info = get_specialization_info(structure)

print(info['description'])
# "Transportation problem: 2 sources → 3 sinks"

for characteristic in info['characteristics']:
    print(f"  - {characteristic}")
# - Bipartite graph structure
# - Only sources and sinks (no transshipment)
# - Balanced (supply equals demand)
```

### NetworkType (Enum)

Classification of network problem types.

**Values:**

- `GENERAL` - General network flow (no special structure)
- `TRANSPORTATION` - Bipartite with only sources and sinks
- `ASSIGNMENT` - Transportation with unit supplies/demands, n×n structure
- `BIPARTITE_MATCHING` - Unit-value matching on bipartite graphs
- `MAX_FLOW` - Single source/sink with uniform costs
- `SHORTEST_PATH` - Unit flow from single source to single sink

**Example:**

```python
from network_solver import NetworkType

if structure.network_type == NetworkType.TRANSPORTATION:
    print("Detected transportation problem")
elif structure.network_type == NetworkType.ASSIGNMENT:
    print("Detected assignment problem")
```

### NetworkStructure (Dataclass)

```python
@dataclass
class NetworkStructure:
    network_type: NetworkType              # Detected problem type
    is_bipartite: bool                     # Whether graph is bipartite
    source_nodes: set[str]                 # Nodes with supply > 0
    sink_nodes: set[str]                   # Nodes with supply < 0
    transshipment_nodes: set[str]          # Nodes with supply = 0
    partitions: tuple[set[str], set[str]] | None  # Bipartite partitions
    total_supply: float                    # Sum of all positive supplies
    total_demand: float                    # Absolute value of sum of negative supplies
    is_balanced: bool                      # Whether supply equals demand
    has_lower_bounds: bool                 # Whether any arc has lower > 0
    has_finite_capacities: bool            # Whether all arcs have finite capacity
```

**Detection Priority:**

The detector classifies problems in this order (most specific first):
1. Transportation (bipartite, sources/sinks only, no lower bounds)
2. Assignment (transportation + unit values + n×n structure)
3. Bipartite Matching (bipartite + unit values)
4. Shortest Path (single source/sink + unit flow)
5. Max Flow (single source/sink + uniform costs)
6. General (fallback)

**Specialized Pivot Strategies:**

When a specialized structure is detected, the solver automatically uses optimized pivot selection:
- **Transportation**: Row-scan pricing exploiting bipartite structure
- **Assignment**: Min-cost selection for n×n unit problems (takes priority over bipartite matching)
- **Bipartite Matching**: Augmenting path methods (for non-assignment bipartite matching)
- **Max Flow**: Capacity-based selection prioritizing high-capacity arcs for larger flow increments
- **Shortest Path**: Distance-label-based selection (Dijkstra-like) maintaining distance labels and guiding arc selection toward sink
- **General**: Standard Devex or Dantzig pricing

**Automatic Integration:**

The solver calls `analyze_network_structure()` automatically during initialization. Detection and strategy selection are logged at INFO level:

```
INFO: Detected network type: Transportation problem: 2 sources → 3 sinks
INFO: Using specialized pivot strategy for transportation
```

## Progress Tracking

### ProgressInfo

```python
@dataclass(frozen=True)
class ProgressInfo:
    iteration: int            # Current total iteration
    max_iterations: int       # Maximum allowed iterations
    phase: int                # 1 (feasibility) or 2 (optimality)
    phase_iterations: int     # Iterations in current phase
    objective_estimate: float # Current objective estimate
    elapsed_time: float       # Seconds since solve started
```

### ProgressCallback

```python
ProgressCallback = Callable[[ProgressInfo], None]
```

Callback function type for progress updates.

**Example:**

```python
def my_callback(info: ProgressInfo):
    pct = 100 * info.iteration / info.max_iterations
    phase = "Phase 1" if info.phase == 1 else "Phase 2"
    print(f"{phase}: {pct:.1f}% complete, obj=${info.objective_estimate:,.2f}")

result = solve_min_cost_flow(
    problem,
    progress_callback=my_callback,
    progress_interval=50  # Call every 50 iterations
)
```

## Exceptions

All exceptions inherit from `NetworkSolverError`.

### NetworkSolverError

Base exception for all solver errors.

```python
class NetworkSolverError(Exception):
    """Base exception for network solver errors."""
```

### InvalidProblemError

Problem definition is malformed or invalid.

```python
class InvalidProblemError(NetworkSolverError):
    """Problem definition is invalid."""
```

**Common causes:**
- Unbalanced supply/demand
- Missing nodes referenced in arcs
- Self-loops
- Capacity < lower bound

### InfeasibleProblemError

No feasible solution exists.

```python
class InfeasibleProblemError(NetworkSolverError):
    """No feasible solution exists."""
    iterations: int  # Iterations before infeasibility detected
```

### UnboundedProblemError

Objective can decrease without bound (negative-cost cycle with infinite capacity).

```python
class UnboundedProblemError(NetworkSolverError):
    """Problem has unbounded objective."""
    entering_arc: tuple[str, str]  # Arc causing unboundedness
    reduced_cost: float             # Reduced cost of entering arc
```

### NumericalInstabilityError

Numerical issues prevent reliable computation.

```python
class NumericalInstabilityError(NetworkSolverError):
    """Numerical instability detected."""
```

### IterationLimitError

Maximum iterations reached (optional exception, usually returns status instead).

```python
class IterationLimitError(NetworkSolverError):
    """Iteration limit reached."""
    iterations: int
```

### SolverConfigurationError

Invalid solver parameters.

```python
class SolverConfigurationError(NetworkSolverError):
    """Invalid solver configuration."""
```

## Input/Output

### load_problem

```python
def load_problem(path: str | Path) -> NetworkProblem
```

Load a problem from a JSON file.

**JSON Format:**

```json
{
  "directed": true,
  "tolerance": 1e-6,
  "nodes": [
    {"id": "s", "supply": 100.0},
    {"id": "t", "supply": -100.0}
  ],
  "arcs": [
    {
      "tail": "s",
      "head": "t",
      "capacity": 150.0,
      "cost": 2.5,
      "lower": 0.0
    }
  ]
}
```

**Parameters:**

- `path` (str | Path): Path to JSON file

**Returns:**

- `NetworkProblem`: Loaded and validated problem

### save_result

```python
def save_result(path: str | Path, result: FlowResult) -> None
```

Save a solution to a JSON file.

**Output Format:**

```json
{
  "status": "optimal",
  "objective": 250.0,
  "iterations": 5,
  "flows": [
    {"tail": "s", "head": "t", "flow": 100.0}
  ],
  "duals": {
    "s": 0.0,
    "t": -2.5
  }
}
```

**Parameters:**

- `path` (str | Path): Output file path
- `result` (FlowResult): Solution to save

## Type Annotations

All functions are fully type-annotated for static analysis with mypy:

```python
from network_solver import solve_min_cost_flow, NetworkProblem, FlowResult

# Type checking works seamlessly
def analyze_solution(problem: NetworkProblem) -> tuple[float, int]:
    result: FlowResult = solve_min_cost_flow(problem)
    return result.objective, result.iterations
```

The library is compatible with:
- mypy (strict mode)
- pyright
- pyre
- Type checkers in modern IDEs (VS Code, PyCharm, etc.)
