# Network Programming Solver

[![CI](https://github.com/jeffreyhorn/network_flow_solver/workflows/CI/badge.svg)](https://github.com/jeffreyhorn/network_flow_solver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jeffreyhorn/network_flow_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyhorn/network_flow_solver)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Python implementation of the network simplex algorithm for classic minimum-cost flow problems with node supplies/demands and optional capacities/lower bounds. The package exposes both a programmatic API and JSON-oriented utilities for loading/saving problem instances.

## Documentation

- **[Jupyter Notebook Tutorial](tutorials/network_flow_tutorial.ipynb)** - Interactive tutorial covering all major features ([How to run](TUTORIAL.md))
- **[Visualization Tutorial](tutorials/visualization_tutorial.ipynb)** - Interactive tutorial for network visualization utilities ([How to run](TUTORIAL.md))
- **[Algorithm Guide](docs/algorithm.md)** - Network simplex algorithm explanation, data structures, and complexity analysis
- **[API Reference](docs/api.md)** - Complete API documentation with all functions, classes, and examples
- **[Examples Guide](docs/examples.md)** - Annotated code examples for common use cases
- **[Performance Guide](docs/benchmarks.md)** - Benchmarks, optimization tips, and scaling behavior
- **[Troubleshooting Guide](docs/troubleshooting.md)** - 🆕 Diagnose and resolve numeric issues, convergence problems, and performance issues

## Installation

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install for development (recommended)
pip install -e ".[dev,umfpack]"

# Or install with tutorial support (Jupyter notebook)
pip install -e ".[tutorial]"

# Or install everything
pip install -e ".[all]"

# Or install runtime only
pip install -e .
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

## Project Layout

- `src/network_solver/` – production code  
  - `data.py` – lightweight data classes and validation helpers  
  - `io.py` – JSON ingestion/persistence helpers  
- `simplex.py` – core network simplex implementation with Forrest–Tomlin basis updates and Devex pricing  
  - `solver.py` – public façade (`load_problem`, `solve_min_cost_flow`, `save_result`)
- `tests/` – unit/property/integration suites exercising CLI flows, simplex edge cases, basis updates, and pricing logic  
- `examples/` – runnable sample including `solve_example.py`

## Prerequisites

- Python 3.12 or newer
- NumPy >= 1.26 and SciPy >= 1.11 (installed automatically)
- Optional: scikit-umfpack >= 0.3.7 for better sparse solver performance (Linux/macOS only)

After installation with `pip install -e .`, the package is available as `network_solver` without needing to modify `PYTHONPATH`.

## Quick Commands

A `Makefile` is provided for common workflows (run from the repository root):

```bash
# Code Quality
make lint          # Run ruff linting checks
make format        # Auto-format code with ruff
make format-check  # Check if code needs formatting (CI-friendly)
make typecheck     # Run mypy type checking
make check         # Run all checks (lint + format-check + typecheck)

# Testing
make test          # Run all tests
make unit          # Run unit tests only
make integration   # Run integration tests only
make coverage      # Run tests with coverage report

# Development
make install       # Install package in development mode
make dev-install   # Install package with dev dependencies
make clean         # Remove build artifacts and caches
make help          # Show all available targets
```

Property-based tests live alongside the unit suite and will execute automatically once Hypothesis is installed. Slow integration tests are marked with `@pytest.mark.slow`; to skip them use `pytest -m "not slow"`.

## Library Usage

```python
from network_solver import load_problem, solve_min_cost_flow, save_result

problem = load_problem("path/to/problem.json")
result = solve_min_cost_flow(problem)
save_result("solution.json", result)
```

- `result.flows` returns a dictionary keyed by `(tail, head)` tuples with optimal flow magnitudes.
- `result.objective` provides the minimized cost, and `result.status` indicates the termination condition (`optimal`, `iteration_limit`, or `infeasible`).

### Exception Handling

The solver uses a hierarchy of custom exceptions for better error handling:

```python
from network_solver import (
    NetworkSolverError,      # Base exception - catch all solver errors
    InvalidProblemError,     # Problem definition is invalid
    InfeasibleProblemError,  # No feasible solution exists
    UnboundedProblemError,   # Objective can decrease without limit
)

try:
    result = solve_min_cost_flow(problem)
except InvalidProblemError as e:
    print(f"Problem is malformed: {e}")
except InfeasibleProblemError as e:
    print(f"No feasible solution: {e}")
except UnboundedProblemError as e:
    print(f"Problem is unbounded: {e}")
    print(f"  Entering arc: {e.entering_arc}")
    print(f"  Reduced cost: {e.reduced_cost}")
except NetworkSolverError as e:
    print(f"Solver error: {e}")
```

**Exception Types:**
- `InvalidProblemError` - Raised for malformed input (unbalanced supply, missing nodes, invalid arcs)
- `InfeasibleProblemError` - No feasible flow exists (includes `iterations` attribute)
- `UnboundedProblemError` - Negative-cost cycle with infinite capacity (includes `entering_arc` and `reduced_cost`)
- `NumericalInstabilityError` - Numerical issues prevent reliable computation
- `SolverConfigurationError` - Invalid solver parameters or configuration
- `IterationLimitError` - Optional exception type (solver returns status instead by default)

See [API Reference](docs/api.md#exceptions) for complete exception documentation.

### Progress Logging

For long-running optimizations, you can monitor solver progress in real-time using a progress callback:

```python
from network_solver import solve_min_cost_flow, ProgressInfo

def progress_callback(info: ProgressInfo) -> None:
    """Called periodically during solve with progress information."""
    percent = 100 * info.iteration / info.max_iterations
    phase_name = "Phase 1" if info.phase == 1 else "Phase 2"
    print(f"{phase_name}: {percent:.1f}% | "
          f"Iter {info.iteration}/{info.max_iterations} | "
          f"Objective: ${info.objective_estimate:,.2f} | "
          f"Time: {info.elapsed_time:.2f}s")

result = solve_min_cost_flow(
    problem,
    progress_callback=progress_callback,
    progress_interval=100  # Callback every 100 iterations
)
```

**ProgressInfo attributes:**
- `iteration` - Current total iteration count
- `max_iterations` - Maximum allowed iterations
- `phase` - Current phase (1 for feasibility, 2 for optimality)
- `phase_iterations` - Iterations in current phase
- `objective_estimate` - Current objective value estimate
- `elapsed_time` - Seconds since solve started

**Use cases:**
- Monitor long-running optimizations
- Implement custom progress bars or GUIs
- Log progress to monitoring systems
- Detect slow convergence issues
- Cancel solver by raising exception in callback

See `examples/progress_logging_example.py` for a complete demonstration and [Examples Guide](docs/examples.md#progress-monitoring) for more details.

### Solver Configuration

Customize solver behavior using `SolverOptions` for fine-grained control:

```python
from network_solver import solve_min_cost_flow, SolverOptions

# Default settings
options = SolverOptions()

# Custom configuration
options = SolverOptions(
    max_iterations=10000,        # Override default iteration limit
    tolerance=1e-9,              # Tighter numerical precision
    pricing_strategy="dantzig",  # Use Dantzig pricing (default: "devex")
    block_size=50,               # Custom pricing block size
    ft_update_limit=100,         # Basis refactorization frequency
)

result = solve_min_cost_flow(problem, options=options)
```

**SolverOptions parameters:**
- `max_iterations` - Maximum simplex iterations (default: `max(100, 5*num_arcs)`)
- `tolerance` - Numerical tolerance for feasibility/optimality (default: `1e-6`)
- `pricing_strategy` - Arc selection strategy:
  - `"devex"` (default) - Normalized reduced costs with block pricing (faster convergence)
  - `"dantzig"` - Most negative reduced cost (simpler, may be slower)
- `block_size` - Number of arcs examined per pricing block:
  - `None` or `"auto"` (default) - Auto-tune based on problem size with runtime adaptation
  - int - Fixed block size (no adaptation)
- `ft_update_limit` - Forrest-Tomlin updates before full basis rebuild (default: `64`)

**Pricing strategies:**
- **Devex pricing** (default): Uses normalized reduced costs and block-based search for efficient arc selection. Generally faster on large problems. **Automatically uses vectorized NumPy operations** for 2-3x speedup on medium to large problems.
- **Dantzig pricing**: Selects the arc with the most negative reduced cost. Simpler but may require more iterations.

**Vectorized pricing (experimental):**
The Devex pricing strategy can optionally use vectorized NumPy array operations, providing significant performance improvements:
- **Small problems** (35 nodes, 300 arcs): **198% speedup** (3x faster)
- **Medium problems** (50 nodes, 600 arcs): **101% speedup** (2x faster)
- **Currently disabled by default** due to a known cycling bug in degenerate cases
- **Can be enabled**: Set `use_vectorized_pricing=True` to try the experimental feature
- **Implementation**: Replaces Python loops with vectorized reduced cost computation, residual calculation, and candidate selection using NumPy masked arrays

```python
# Default: vectorization disabled (stable)
options = SolverOptions(pricing_strategy="devex")  # use_vectorized_pricing=False

# Enable experimental vectorization (faster but may cycle on degenerate problems)
options = SolverOptions(pricing_strategy="devex", use_vectorized_pricing=True)
```

**Known issue:** The vectorized implementation may cause infinite cycling on problems with degenerate pivots (zero flow changes). The root cause is that vectorized pricing uses fixed Devex weights throughout arc selection, while the loop-based version dynamically updates weights for each candidate. Work is ongoing to resolve this architectural difference.

The vectorization works by maintaining parallel NumPy arrays that mirror the arc list, enabling batch operations for computing reduced costs, checking eligibility, and selecting the best entering arc. This optimization is particularly effective for problems with many arcs where pricing is a bottleneck, but should only be used when you can verify it doesn't cause cycling on your specific problem instances.

**Performance tuning:**
- Increase `tolerance` for faster (less precise) solutions
- Decrease `tolerance` for high-precision requirements
- Use `block_size="auto"` (default) for automatic tuning, or specify a fixed int for manual control
- Lower `ft_update_limit` for better numerical stability (more rebuilds)
- Raise `ft_update_limit` for faster performance (fewer rebuilds)

**Block size auto-tuning:**
By default (`block_size=None` or `"auto"`), the solver automatically selects and adapts the block size:
- **Initial heuristic**: Based on problem size (smaller blocks for smaller problems)
- **Runtime adaptation**: Monitors degenerate pivot ratio and adjusts every 50 iterations
  - High degeneracy (>30%) → increase block size (explore wider)
  - Low degeneracy (<10%) → decrease block size (focused search)

See `examples/solver_options_example.py` for a comprehensive demonstration. For performance benchmarks and optimization guidance, see the [Performance Guide](docs/benchmarks.md).

**Sparse vs Dense Basis Mode:**

The solver supports two modes for basis matrix operations:
- **Sparse mode** (default with scipy): Uses sparse LU factorization only, avoiding O(n³) dense inverse computation
- **Dense mode** (fallback): Computes full dense inverse matrix for Sherman-Morrison updates

```python
# Default: Auto-detect (sparse if scipy available, dense otherwise)
result = solve_min_cost_flow(problem)

# Force sparse mode (requires scipy)
options = SolverOptions(use_dense_inverse=False)

# Force dense mode (works without scipy, but less scalable)
options = SolverOptions(use_dense_inverse=True)
```

**Performance impact:**
- Small problems (<100 nodes): Dense mode may be slightly faster due to Sherman-Morrison updates
- Large problems (>1000 nodes): Sparse mode dramatically faster and uses less memory
  - Avoids O(n²) memory for dense inverse matrix
  - For n=10,000: saves ~800MB of memory

See `examples/sparse_vs_dense_example.py` for benchmark comparisons on different problem sizes.

### Automatic Problem Scaling

The solver automatically detects and scales problems with extreme value ranges to improve numerical stability. This is particularly useful when costs, capacities, or supplies span many orders of magnitude.

**Automatic Scaling (Enabled by Default):**
```python
from network_solver import build_problem, solve_min_cost_flow

# Problem with extreme value ranges
problem = build_problem(
    nodes=[
        {"id": "source", "supply": 100_000_000.0},  # 100 million units
        {"id": "sink", "supply": -100_000_000.0},
    ],
    arcs=[
        {"tail": "source", "head": "sink", "capacity": 200_000_000.0, "cost": 0.001},
        # Range: 0.001 to 200,000,000 (11 orders of magnitude!)
    ],
    directed=True,
    tolerance=1e-6,
)

# Solver automatically detects wide range and scales the problem
result = solve_min_cost_flow(problem)
# INFO: Applied automatic problem scaling
#       cost_scale=1000.0, capacity_scale=5e-09, supply_scale=1e-08
# Solution is automatically unscaled back to original units
print(f"Objective: ${result.objective:,.2f}")  # $100,000.00
print(f"Flow: {result.flows[('source', 'sink')]:,.0f}")  # 100,000,000 units
```

**How It Works:**
1. **Detection**: Scaling triggers when values differ by >6 orders of magnitude (threshold: 1,000,000)
2. **Scaling**: Uses geometric mean to normalize costs, capacities, and supplies independently
3. **Target Range**: Brings values into [0.1, 100] range for numerical stability
4. **Solving**: Solver works on scaled problem with well-conditioned values
5. **Unscaling**: Solution automatically converted back to original units

**Manual Control:**
```python
from network_solver import (
    should_scale_problem,
    compute_scaling_factors,
    SolverOptions,
)

# Check if scaling is recommended
if should_scale_problem(problem):
    factors = compute_scaling_factors(problem)
    print(f"Cost scale: {factors.cost_scale:.2e}")
    print(f"Capacity scale: {factors.capacity_scale:.2e}")
    print(f"Supply scale: {factors.supply_scale:.2e}")

# Disable automatic scaling if needed
options = SolverOptions(auto_scale=False)
result = solve_min_cost_flow(problem, options=options)
```

**When Scaling Helps:**
- Micro-costs (e.g., $0.0001) combined with macro-supplies (e.g., millions of units)
- Very large capacities with very small costs
- Mixed-scale transportation/assignment problems
- Any problem where values span >6 orders of magnitude

**When to Disable Scaling:**
- Testing specific numerical behaviors
- Working with pre-scaled problems
- Debugging scaling-related issues
- Well-balanced problems (scaling is skipped automatically anyway)

**Benefits:**
- **Improved stability**: Reduces round-off errors and catastrophic cancellation
- **Better convergence**: Well-conditioned problems may converge faster
- **Automatic**: No manual intervention needed
- **Transparent**: Solutions returned in original units
- **Safe**: Well-balanced problems are not affected

See `examples/automatic_scaling_example.py` for a comprehensive demonstration with transportation problems.

### Problem Preprocessing

The solver includes **problem preprocessing** to simplify network flow problems before solving, reducing problem size and improving performance while preserving optimal solutions. **Solutions are automatically translated back** to the original problem structure.

**Four Optimization Techniques:**

1. **Remove redundant arcs** - Parallel arcs with identical costs are merged (capacities combined)
2. **Detect disconnected components** - BFS-based connectivity analysis warns of potential infeasibility
3. **Simplify series arcs** - Merge consecutive arcs through zero-supply transshipment nodes
4. **Remove zero-supply nodes** - Eliminate transshipment nodes with single incident arc

**Basic Usage:**
```python
from network_solver import preprocess_problem, solve_min_cost_flow

# Preprocess then solve
result = preprocess_problem(problem)
print(f"Removed {result.removed_arcs} arcs, {result.removed_nodes} nodes")
print(f"Preprocessing time: {result.preprocessing_time_ms:.2f}ms")

# Solve the preprocessed problem
flow_result = solve_min_cost_flow(result.problem)
```

**Convenience Function (Recommended):**
```python
from network_solver import preprocess_and_solve

# Preprocess, solve, and automatically translate solution back to original problem
preproc_result, flow_result = preprocess_and_solve(problem)
print(f"Removed {preproc_result.removed_arcs} arcs")
print(f"Optimal cost: ${flow_result.objective:.2f}")

# flow_result contains flows and duals for the ORIGINAL problem
# - All original arcs have flow values (including removed/merged arcs)
# - All original nodes have dual values (including removed nodes)
print(f"Flow on original arc: {flow_result.flows[('factory', 'hub1')]}")
print(f"Dual for removed node: {flow_result.duals['hub1']}")
```

**Selective Preprocessing:**
```python
# Control which optimizations to apply
result = preprocess_problem(
    problem,
    remove_redundant=True,      # Merge parallel arcs (default: True)
    detect_disconnected=True,   # Connectivity analysis (default: True)
    simplify_series=True,       # Series arc merging (default: True)
    remove_zero_supply=True,    # Single-arc node removal (default: True)
)
```

**Example - Series Arc Simplification:**
```python
# Problem with chain of transshipment nodes
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

# Preprocessing merges series arcs
result = preprocess_problem(build_problem(nodes, arcs, directed=True, tolerance=1e-6))
print(f"Removed {result.removed_nodes} transshipment nodes")  # 3
print(f"Merged {result.merged_arcs} arcs")                     # 3
# Result: factory → customer (capacity: 120.0, cost: 5.0)
```

**PreprocessingResult Statistics:**
```python
result = preprocess_problem(problem)

print(f"Removed arcs: {result.removed_arcs}")
print(f"Removed nodes: {result.removed_nodes}")
print(f"Merged arcs: {result.merged_arcs}")
print(f"Redundant arcs: {result.redundant_arcs}")
print(f"Disconnected components: {result.disconnected_components}")
print(f"Preprocessing time: {result.preprocessing_time_ms:.2f}ms")
print(f"Optimizations: {result.optimizations}")
```

**Result Translation:**

When using `preprocess_and_solve()`, solutions are automatically translated back to the original problem:

- **Removed arcs** → assigned zero flow
- **Redundant arcs** (merged) → flows distributed proportionally by capacity
- **Series arcs** (merged) → all arcs in the series carry the same flow
- **Removed nodes** → duals computed from adjacent preserved arcs
- **Preserved arcs/nodes** → flows/duals copied directly from preprocessed solution

This means you can use preprocessing transparently—the solution always corresponds to your original problem structure.

**When Preprocessing Helps:**
- **Redundant network design** with parallel routes having same cost
- **Complex supply chains** with many transshipment nodes
- **Large-scale problems** where size reduction improves solve time
- **Disconnected networks** where early detection prevents wasted computation

**Performance Impact:**
- **Problem size reduction:** Typical 20-50% fewer arcs/nodes
- **Solve time improvement:** 1.2x-2x speedup for large problems with redundancy
- **Preprocessing overhead:** Minimal (<1% of solve time for most problems)
- **Semantics preserved:** Optimal solutions identical to original problem

**Benefits:**
- **Automatic optimization** - No manual problem simplification needed
- **Faster solving** - Smaller problems converge more quickly
- **Transparent translation** - Solutions automatically mapped back to original structure
- **Early detection** - Warns about disconnected components
- **Safe transformations** - Preserves problem structure and optimal solutions
- **Detailed statistics** - Track exactly what was simplified

**Note:** Preprocessing is independent and can be combined with automatic scaling, adaptive refactorization, and all other solver features.

See `examples/preprocessing_example.py` for comprehensive demonstrations including performance comparisons and result translation.

### Adaptive Basis Refactorization

The solver features **adaptive refactorization** that monitors numerical stability and automatically adjusts basis rebuild frequency to maintain accuracy while maximizing performance.

**How It Works:**
The solver tracks the **condition number** of the basis matrix during each pivot. When the condition number exceeds a threshold (indicating potential numerical issues), the solver triggers a basis rebuild and adaptively adjusts the refactorization frequency.

**Enabled by Default:**
```python
from network_solver import solve_min_cost_flow, SolverOptions

# Default settings enable adaptive refactorization
options = SolverOptions()  # adaptive_refactorization=True
result = solve_min_cost_flow(problem, options=options)
```

**Configuration Options:**
```python
# Customize adaptive behavior
options = SolverOptions(
    adaptive_refactorization=True,       # Enable adaptive mode (default)
    condition_number_threshold=1e12,     # Trigger threshold (default)
    adaptive_ft_min=20,                  # Minimum refactorization limit
    adaptive_ft_max=200,                 # Maximum refactorization limit
    ft_update_limit=64,                  # Initial/fixed limit
)
```

**Parameters:**
- `adaptive_refactorization` - Enable/disable adaptive behavior (default: `True`)
- `condition_number_threshold` - Condition number limit for triggering rebuild (default: `1e12`)
  - Lower (1e10): More conservative, more rebuilds, better stability
  - Higher (1e14): More aggressive, fewer rebuilds, faster but less stable
- `adaptive_ft_min` - Minimum value for adaptive ft_update_limit (default: `20`)
- `adaptive_ft_max` - Maximum value for adaptive ft_update_limit (default: `200`)
- `ft_update_limit` - Starting limit or fixed limit if adaptive disabled (default: `64`)

**When Adaptive Refactorization Helps:**
- **Ill-conditioned problems** with wide value ranges
- **Mixed-scale networks** (micro-costs with macro-capacities)
- **Long-running solves** where stability is critical
- **Unknown problem characteristics** (adaptive tuning finds optimal frequency)

**Disable for:**
- **Well-conditioned problems** with narrow value ranges
- **Predictable behavior** when you need fixed refactorization
- **Manual tuning** when you've optimized ft_update_limit for your workload

**Example - Ill-Conditioned Problem:**
```python
# Problem with extreme value ranges (costs: 0.001, capacities: millions)
nodes = [
    {"id": "factory", "supply": 1_000_000.0},
    {"id": "warehouse", "supply": -1_000_000.0},
]
arcs = [
    {"tail": "factory", "head": "warehouse", "capacity": 2_000_000.0, "cost": 0.001},
]
problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Adaptive refactorization automatically maintains stability
result = solve_min_cost_flow(problem)  # Works correctly with defaults
```

**Benefits:**
- **Automatic tuning** - No manual adjustment of ft_update_limit needed
- **Improved stability** - Detects and responds to numerical issues
- **Better performance** - Reduces unnecessary rebuilds for well-conditioned problems
- **Transparent** - Works seamlessly with automatic scaling

**Note:** Adaptive refactorization works in combination with automatic problem scaling. Together, these features provide robust numerical behavior across diverse problem types.

See `examples/adaptive_refactorization_example.py` for comprehensive demonstrations and tuning guidelines.

### Network Specializations and Optimized Pivots

The solver automatically detects special network structures and applies specialized pivot strategies for improved performance:

**Automatically Detected Problem Types:**
- **Transportation problems** - Bipartite networks with only sources and sinks
- **Assignment problems** - Transportation with unit supplies/demands and n×n structure
- **Bipartite matching** - Unit-value matching problems on bipartite graphs
- **Max flow problems** - Single source/sink with uniform costs
- **Shortest path problems** - Unit flow from single source to single sink

**Specialized Pivot Strategies:**
When a specialized structure is detected, the solver automatically uses optimized pivot selection:
- **Transportation**: Row-scan pricing exploiting bipartite structure
- **Assignment**: Min-cost selection for n×n unit problems
- **Bipartite matching**: Augmenting path methods (for non-assignment bipartite matching)
- **Max flow**: Capacity-based selection prioritizing high-capacity arcs for larger flow increments
- **Shortest path**: Distance-label-based selection (Dijkstra-like) guiding arc selection toward sink
- **General problems**: Standard Devex or Dantzig pricing

```python
from network_solver import build_problem, solve_min_cost_flow, analyze_network_structure

# Create a transportation problem
problem = build_problem(
    nodes=[
        {"id": "factory1", "supply": 100.0},
        {"id": "factory2", "supply": 150.0},
        {"id": "warehouse1", "supply": -120.0},
        {"id": "warehouse2", "supply": -130.0},
    ],
    arcs=[
        {"tail": "factory1", "head": "warehouse1", "capacity": 200.0, "cost": 5.0},
        {"tail": "factory1", "head": "warehouse2", "capacity": 200.0, "cost": 3.0},
        {"tail": "factory2", "head": "warehouse1", "capacity": 200.0, "cost": 2.0},
        {"tail": "factory2", "head": "warehouse2", "capacity": 200.0, "cost": 4.0},
    ],
    directed=True,
    tolerance=1e-6,
)

# Analyze structure manually (optional - solver does this automatically)
structure = analyze_network_structure(problem)
print(f"Problem type: {structure.network_type.value}")
print(f"Is bipartite: {structure.is_bipartite}")
print(f"Sources: {len(structure.source_nodes)}, Sinks: {len(structure.sink_nodes)}")

# Solver automatically detects structure and uses specialized pivots
result = solve_min_cost_flow(problem)
# INFO: Detected network type: Transportation problem: 2 sources → 2 sinks
# INFO: Using specialized pivot strategy for transportation
```

**Benefits:**
- **Automatic optimization** - No manual configuration needed
- **Better performance** - Specialized algorithms exploit problem structure
- **Transparent** - Falls back to general methods when specialized structure isn't detected
- **Logged** - Detection and strategy selection logged at INFO level

**API Functions:**
- `analyze_network_structure(problem)` - Manually analyze network structure
- `NetworkType` enum - Problem classification (TRANSPORTATION, ASSIGNMENT, etc.)
- Automatic detection integrated into `solve_min_cost_flow()`

The detection algorithm uses bipartite graph recognition (BFS 2-coloring) and analyzes node types (sources, sinks, transshipment), supply/demand patterns, and network topology to classify problems and select appropriate strategies.

### Utility Functions for Flow Analysis

The library provides utilities for analyzing and validating flow solutions:

#### Extract Flow Paths

Find specific routes that flow takes through the network:

```python
from network_solver import extract_path

# Find a flow-carrying path from source to target
path = extract_path(result, problem, source="factory_a", target="warehouse_1")

if path:
    print(f"Route: {' -> '.join(path.nodes)}")
    print(f"Flow: {path.flow} units")
    print(f"Cost: ${path.cost}")
    print(f"Arcs: {path.arcs}")  # List of (tail, head) tuples
```

**Use cases:**
- Trace shipment routes in supply chains
- Understand flow patterns in networks
- Visualize solution paths
- Debug unexpected routing

#### Validate Solutions

Verify that a flow solution satisfies all constraints:

```python
from network_solver import validate_flow

validation = validate_flow(problem, result)

if validation.is_valid:
    print("✓ Solution is valid")
else:
    print("✗ Solution has violations:")
    for error in validation.errors:
        print(f"  - {error}")
    
    # Check specific violation types
    if validation.capacity_violations:
        print(f"Capacity violations: {validation.capacity_violations}")
    if validation.lower_bound_violations:
        print(f"Lower bound violations: {validation.lower_bound_violations}")
```

**Validation checks:**
- Flow conservation at each node (inflow - outflow = supply)
- Capacity constraints (flow ≤ capacity)
- Lower bound constraints (flow ≥ lower)
- Configurable numerical tolerance

#### Identify Bottlenecks

Find arcs that limit network capacity:

```python
from network_solver import compute_bottleneck_arcs

# Find arcs at 90% or higher utilization
bottlenecks = compute_bottleneck_arcs(problem, result, threshold=0.90)

for bottleneck in bottlenecks:
    print(f"Arc ({bottleneck.tail} -> {bottleneck.head}):")
    print(f"  Utilization: {bottleneck.utilization * 100:.1f}%")
    print(f"  Flow: {bottleneck.flow} / Capacity: {bottleneck.capacity}")
    print(f"  Slack: {bottleneck.slack} units remaining")
    print(f"  Cost: ${bottleneck.cost}/unit")
```

**Use cases:**
- Identify capacity constraints limiting throughput
- Prioritize infrastructure investments
- Perform what-if analysis for capacity expansion
- Sensitivity analysis for network planning

See `examples/utils_example.py` for a complete demonstration of all utility functions. Full API documentation available in the [API Reference](docs/api.md#utility-functions).

### Visualization Utilities

The solver provides **visualization utilities** to create publication-quality graphs of network structures, flow solutions, and bottleneck analysis using matplotlib and networkx.

**Installation:**
```bash
pip install 'network_solver[visualization]'
```

#### Visualize Network Structure

Display network topology with nodes, arcs, costs, and capacities:

```python
from network_solver import build_problem, visualize_network

problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

# Visualize network structure
fig = visualize_network(problem, layout="spring", figsize=(12, 8))
fig.savefig("network.png")
```

**Features:**
- Automatic node categorization (sources=green, sinks=red, transshipment=blue)
- Arc labels showing costs and capacities
- Supply/demand values on nodes
- Multiple layout algorithms (spring, circular, kamada_kawai, planar)

#### Visualize Flow Solution

Display optimal flows with optional bottleneck highlighting:

```python
from network_solver import solve_min_cost_flow, visualize_flows

result = solve_min_cost_flow(problem)

# Visualize flows with bottleneck highlighting
fig = visualize_flows(
    problem,
    result,
    highlight_bottlenecks=True,
    bottleneck_threshold=0.9,  # Highlight arcs ≥90% utilization
    show_zero_flows=False,      # Hide zero flows for clarity
)
fig.savefig("flows.png")
```

**Features:**
- Flow values displayed on arcs
- Arc thickness proportional to flow magnitude
- Bottleneck highlighting in red (utilization ≥ threshold)
- Utilization percentages displayed
- Statistics box (objective, status, iterations)
- Option to hide zero flows

#### Visualize Bottlenecks

Focused analysis of capacity constraints with utilization heatmap:

```python
from network_solver import visualize_bottlenecks

# Show arcs with ≥80% utilization
fig = visualize_bottlenecks(problem, result, threshold=0.8)
fig.savefig("bottlenecks.png")
```

**Features:**
- Utilization heatmap with color gradient (red=high, yellow=medium, green=low)
- Only shows arcs above threshold
- Displays utilization % and slack capacity
- Color bar for scale reference
- Statistics (bottleneck count, average utilization)

#### Customization Options

All visualization functions support extensive customization:

```python
fig = visualize_network(
    problem,
    layout="kamada_kawai",      # Layout algorithm
    figsize=(14, 10),            # Figure size
    node_size=1200,              # Node marker size
    font_size=10,                # Label font size
    show_arc_labels=True,        # Show/hide arc labels
    title="My Custom Network",   # Custom title
)
```

**Use Cases:**
- Visual problem understanding (structure, complexity, bottlenecks)
- Flow pattern analysis (routing decisions, utilization)
- Capacity planning (identify constraints, prioritize investments)
- Publication-quality figures for reports and presentations
- Interactive problem exploration and debugging

**Note:** Requires optional dependencies matplotlib and networkx. Install with `pip install 'network_solver[visualization]'`.

See `examples/visualization_example.py` for comprehensive demonstrations generating 8 different visualizations.

### Sensitivity Analysis with Dual Values

The solver returns **dual values** (also called **shadow prices** or **node potentials**) which represent the marginal cost of supply/demand changes:

```python
result = solve_min_cost_flow(problem)

# Access dual values (shadow prices)
for node_id, dual in result.duals.items():
    print(f"Node {node_id}: dual value = {dual:.6f}")

# Predict cost change without re-solving
# If we increase supply at node 'supplier' by 10 units:
cost_change_per_unit = result.duals["supplier"] - result.duals["customer"]
predicted_change = 10 * cost_change_per_unit
print(f"Predicted cost change: ${-predicted_change:.2f}")

# Verify complementary slackness (optimality condition)
# For arcs with positive flow, reduced cost should be ~0:
for (tail, head), flow in result.flows.items():
    if flow > 1e-6:
        reduced_cost = arc_cost + result.duals[tail] - result.duals[head]
        print(f"{tail}->{head}: reduced_cost = {reduced_cost:.10f}")
```

**Use cases for dual values:**
- **"What-if" analysis**: Predict cost impact of supply/demand changes without re-solving
- **Capacity planning**: Identify which capacity expansions provide the most value
- **Pricing decisions**: Determine value of expedited delivery or premium sourcing
- **Bottleneck identification**: Find binding capacity constraints
- **Optimality verification**: Check complementary slackness conditions

See `examples/sensitivity_analysis_example.py` for comprehensive examples including production planning, marginal cost prediction, and bottleneck identification. For mathematical background, see the [Algorithm Guide](docs/algorithm.md#node-potentials-dual-variables) and [Examples Guide](docs/examples.md#sensitivity-analysis).

### Warm-Starting for Sequential Solves

**Warm-starting** reuses the basis (spanning tree structure) from a previous solve to accelerate solving similar problems. This is especially valuable for:
- Sequential optimization (rolling horizon planning)
- Sensitivity analysis with parameter variations
- Real-time scenario evaluation
- What-if analysis

```python
from network_solver import solve_min_cost_flow, build_problem

# Solve initial problem
nodes1 = [
    {"id": "warehouse", "supply": 100.0},
    {"id": "store", "supply": -100.0}
]
arcs = [{"tail": "warehouse", "head": "store", "capacity": 150.0, "cost": 2.5}]
problem1 = build_problem(nodes1, arcs, directed=True, tolerance=1e-6)
result1 = solve_min_cost_flow(problem1)

print(f"First solve: {result1.iterations} iterations")
# Extract basis for reuse
basis = result1.basis

# Solve modified problem with warm-start
nodes2 = [
    {"id": "warehouse", "supply": 120.0},  # Increased supply
    {"id": "store", "supply": -120.0}
]
problem2 = build_problem(nodes2, arcs, directed=True, tolerance=1e-6)
result2 = solve_min_cost_flow(problem2, warm_start_basis=basis)

print(f"Warm-start solve: {result2.iterations} iterations")  # Typically much fewer!
```

**Benefits:**
- **50-90% reduction in iterations** for similar problems
- **Faster solve times** for sequential optimization
- **Enables real-time** scenario evaluation
- **Essential for interactive applications**

**Works best when:**
- Network structure is similar (same nodes and arcs)
- Supply/demand or costs change moderately
- Capacities are adjusted but optimal routes remain similar

See `examples/warm_start_example.py` for comprehensive examples including supply changes, cost variations, capacity expansion analysis, and performance comparisons.

## CLI Example

Run the bundled examples to see the solver end-to-end:

```bash
python examples/solve_example.py  # Basic example with dual values
python examples/solve_dimacs_example.py  # DIMACS-style instance
python examples/solve_textbook_transport.py  # Textbook transportation problem
python examples/solve_large_transport.py  # 10×10 transportation instance
python examples/preprocessing_example.py  # Problem preprocessing and optimization
python examples/visualization_example.py  # Network and flow visualization (requires matplotlib)
python examples/sensitivity_analysis_example.py  # Dual values and shadow prices
python examples/incremental_resolving_example.py  # Scenario analysis and what-if modeling
python examples/performance_profiling_example.py  # Performance analysis and benchmarking
python examples/networkx_comparison_example.py  # Comparison with NetworkX
python examples/warm_start_example.py  # Warm-starting for sequential solves
python examples/progress_logging_example.py  # Progress monitoring
python examples/solver_options_example.py  # Solver configuration and tuning
python examples/adaptive_refactorization_example.py  # Adaptive basis refactorization
python examples/sparse_vs_dense_example.py  # Sparse vs dense basis performance comparison
python examples/utils_example.py  # Flow analysis utilities
python examples/undirected_graph_example.py  # Undirected graph handling
```

These scripts write companion solution files and print detailed results including dual values and solver statistics.

### Verbose Output

Many examples support the `--verbose` flag for detailed solver logging:

```bash
python examples/solve_example.py -v    # INFO: Phase transitions and progress
python examples/solve_example.py -vv   # DEBUG: Every pivot operation
```

**Log levels:**
- Default (no flag): WARNING and ERROR only (quiet operation)
- `-v`: INFO level - phase transitions, iteration counts
- `-vv`: DEBUG level - individual pivots, arc selection, numerical details

#### Structured Logging for Monitoring

All log messages include structured data in the `extra` dict for programmatic parsing. This enables JSON logging for monitoring systems, performance profiling, and automated testing:

```python
import json
import logging
from network_solver import load_problem, solve_min_cost_flow

# Configure JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            **{k: v for k, v in record.__dict__.items() 
               if k not in logging.LogRecord.__dict__}
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("network_solver").addHandler(handler)
logging.getLogger("network_solver").setLevel(logging.INFO)

# Solve - logs include structured metrics
problem = load_problem("examples/sample_problem.json")
result = solve_min_cost_flow(problem)
```

**Structured metrics included:**
- **Solver start**: `nodes`, `arcs`, `max_iterations`, `pricing_strategy`, `total_supply`, `tolerance`
- **Phase 1 complete**: `iterations`, `total_iterations`, `artificial_flow`, `elapsed_ms`
- **Phase 2 complete**: `iterations`, `total_iterations`, `objective`, `elapsed_ms`
- **Solver complete**: `status`, `objective`, `iterations`, `elapsed_ms`, `tree_arcs`, `nonzero_flows`, `ft_rebuilds`

Example JSON output:
```json
{"level": "INFO", "message": "Starting network simplex solver", "nodes": 3, "arcs": 3, "max_iterations": 100, "pricing_strategy": "devex", "total_supply": 10.0, "tolerance": 1e-06}
{"level": "INFO", "message": "Phase 1 complete", "iterations": 2, "total_iterations": 2, "artificial_flow": 0, "elapsed_ms": 2.23}
{"level": "INFO", "message": "Solver complete", "status": "optimal", "objective": 15.0, "iterations": 2, "elapsed_ms": 4.04, "tree_arcs": 2, "nonzero_flows": 2, "ft_rebuilds": 0}
```

## Problem File Format

Problem instances are JSON documents with the following shape:

- Top-level keys: `directed`, `tolerance`, `nodes`, `edges` (or `arcs`)
- Each node is an object with `id` and optional `supply` (positive supply, negative demand)
- Each edge includes:
  - `tail`, `head` (node IDs)
  - optional `capacity` (omit for infinite), `cost`, and `lower` bound

### Undirected Graphs

Set `"directed": false` to create an undirected graph. **Important requirements:**

- **All edges must have finite capacity** (no infinite capacity)
- **Do not specify custom lower bounds** (leave at default 0.0)
- **Costs are symmetric** in both directions

**How it works:** Each undirected edge `{u, v}` with capacity `C` is automatically transformed into a directed arc `(u, v)` with:
- Capacity: `C` (upper bound)
- Lower bound: `-C` (allows reverse flow)
- Cost: same in both directions

**Interpreting results:**
- Positive flow value → flow goes `tail → head`
- Negative flow value → flow goes `head → tail`  
- Magnitude `|flow|` is the amount of flow

See [API Reference - Undirected Graphs](docs/api.md#working-with-undirected-graphs) for detailed examples and `examples/undirected_graph_example.py` for a complete demonstration.

## Numeric Validation and Diagnostics

The solver includes built-in tools to detect and diagnose numeric and convergence issues:

### Numeric Validation

Analyze problem properties before solving to detect potential numeric issues:

```python
from network_solver import analyze_numeric_properties, validate_numeric_properties

# Analyze numeric properties
analysis = analyze_numeric_properties(problem)

if not analysis.is_well_conditioned:
    print("Numeric issues detected:")
    for warning in analysis.warnings:
        print(f"  {warning.severity}: {warning.message}")
        print(f"  → {warning.recommendation}")

# Strict validation (raises exception on high-severity issues)
validate_numeric_properties(problem, strict=True, warn=True)
```

**Detects:**
- Extreme values (very large or very small coefficients)
- Wide coefficient ranges (may cause precision loss)
- Ill-conditioned problems
- Provides actionable recommendations for scaling

See [Troubleshooting Guide](docs/troubleshooting.md) for detailed guidance on resolving numeric issues.

### Convergence Monitoring

Track solver progress and detect convergence issues:

```python
from network_solver import ConvergenceMonitor

monitor = ConvergenceMonitor(window_size=50, stall_threshold=1e-8)

def track_convergence(info):
    monitor.record_iteration(
        objective=info.objective_estimate,
        is_degenerate=False,  # Can detect from solver state
        iteration=info.iteration
    )
    
    if monitor.is_stalled():
        print(f"Warning: Stalling detected at iteration {info.iteration}")
        diagnostics = monitor.get_diagnostic_summary()
        print(f"  Degeneracy ratio: {diagnostics['degeneracy_ratio']:.2%}")
        print(f"  Recent improvement: {diagnostics['recent_improvement']:.2e}")

result = solve_min_cost_flow(
    problem,
    progress_callback=track_convergence,
    progress_interval=100
)
```

**Features:**
- Stalling detection (objective not improving)
- Degeneracy monitoring (zero-pivot ratio)
- Cycling detection (basis state history)
- Improvement rate tracking
- Adaptive tolerance recommendations

## Testing

- `tests/unit/` – validation, IO, simplex edge cases (pricing, pivots, flow cleanup), and property-based generators
  - `test_validation.py` – 🆕 Numeric property analysis and validation (13 tests)
  - `test_diagnostics.py` – 🆕 Convergence monitoring and diagnostics (18 tests)
- `tests/integration/` – CLI round-trips, JSON contracts, unbounded/infeasible detection, performance/expansion guards, and failure-path checks for malformed configs
- `examples/dimacs_small_problem.json` – small DIMACS-inspired chain (5 nodes, 4 arcs)
- `examples/textbook_transport_problem.json` – 2×3 transportation example (85.0 optimal cost)
- `examples/large_transport_problem.json` – 10×10 balanced transport with diagonal optimum
- `tests/test_large_directed.py` – high-volume directed chain scenarios
- `tests/test_property_min_cost_flow.py` – Hypothesis-driven invariants (requires `hypothesis`)

Run everything with `make test` or invoke your preferred pytest subsets directly.

## Numerical Notes

- The solver maintains a Forrest–Tomlin update engine backed by sparse LU factors when SciPy (and UMFPACK) are available; otherwise it falls back to dense NumPy solves, with failover paths verified by unit tests.
- Phase 1 terminates early once all artificial arcs drop to zero flow so Phase 2 receives the remaining iteration budget; unit tests cover infeasible outcomes and iteration-limited runs.
- Devex pricing leverages basis solves; unit tests now cover wrap-around, zero-reduced-cost selection, projection fallbacks, and weight clamping.

## Contributing

- Format code with `make format`
- Check code quality with `make check` (runs lint, format-check, and typecheck)
- Run `make test` before submitting changes
- Keep new examples in `examples/` and note structural changes in `AGENTS.md`

Feel free to open issues or PRs—feedback and improvements are welcome.
