# API Reference

Complete API documentation for the network flow solver library.

## Table of Contents

- [Main Functions](#main-functions)
- [Problem Definition](#problem-definition)
- [Solver Configuration](#solver-configuration)
- [Results and Analysis](#results-and-analysis)
- [Utility Functions](#utility-functions)
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
    block_size: int | None = None
    ft_update_limit: int = 64
```

Configuration options for the solver.

**Attributes:**

- `max_iterations` (int, optional): Maximum simplex iterations. Default: `max(100, 5*num_arcs)`
- `tolerance` (float): Numerical tolerance for feasibility/optimality (default: 1e-6)
- `pricing_strategy` (str): Arc selection strategy
  - `"devex"` (default): Devex normalized pricing (faster)
  - `"dantzig"`: Most negative reduced cost (simpler)
- `block_size` (int, optional): Arcs per pricing block. Default: `num_arcs/8`
- `ft_update_limit` (int): Forrest-Tomlin updates before refactorization (default: 64)

**Validation:**

- `tolerance > 0`
- `pricing_strategy` in `{"devex", "dantzig"}`
- `block_size > 0` if provided
- `ft_update_limit > 0`

**Example:**

```python
# High-precision solve
options = SolverOptions(tolerance=1e-10)

# Dantzig pricing with small blocks
options = SolverOptions(pricing_strategy="dantzig", block_size=10)

# Aggressive refactorization for stability
options = SolverOptions(ft_update_limit=20)
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
- **Assignment**: Min-cost selection for n×n unit problems
- **Bipartite Matching**: Augmenting path methods
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
