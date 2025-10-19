# Network Programming Solver

[![CI](https://github.com/jeffreyhorn/network_flow_solver/workflows/CI/badge.svg)](https://github.com/jeffreyhorn/network_flow_solver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jeffreyhorn/network_flow_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyhorn/network_flow_solver)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Python implementation of the network simplex algorithm for classic minimum-cost flow problems with node supplies/demands and optional capacities/lower bounds. The package exposes both a programmatic API and JSON-oriented utilities for loading/saving problem instances.

## Documentation

- **[Algorithm Guide](docs/algorithm.md)** - Network simplex algorithm explanation, data structures, and complexity analysis
- **[API Reference](docs/api.md)** - Complete API documentation with all functions, classes, and examples
- **[Examples Guide](docs/examples.md)** - Annotated code examples for common use cases
- **[Performance Guide](docs/benchmarks.md)** - Benchmarks, optimization tips, and scaling behavior

## Installation

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install for development (recommended)
pip install -e ".[dev,umfpack]"

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
make lint          # ruff check
make unit          # unit test suite
make integration   # integration & CLI tests
make test          # full pytest run
make fmt           # ruff format + lint (see docs/ for details)
make coverage      # pytest with coverage report (requires pytest-cov)
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
- `block_size` - Number of arcs examined per pricing block (default: `num_arcs/8`)
- `ft_update_limit` - Forrest-Tomlin updates before full basis rebuild (default: `64`)

**Pricing strategies:**
- **Devex pricing** (default): Uses normalized reduced costs and block-based search for efficient arc selection. Generally faster on large problems.
- **Dantzig pricing**: Selects the arc with the most negative reduced cost. Simpler but may require more iterations.

**Performance tuning:**
- Increase `tolerance` for faster (less precise) solutions
- Decrease `tolerance` for high-precision requirements
- Adjust `block_size` based on problem sparsity
- Lower `ft_update_limit` for better numerical stability (more rebuilds)
- Raise `ft_update_limit` for faster performance (fewer rebuilds)

See `examples/solver_options_example.py` for a comprehensive demonstration. For performance benchmarks and optimization guidance, see the [Performance Guide](docs/benchmarks.md).

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

### Sensitivity Analysis with Dual Values

The solver returns dual values (node potentials) which represent shadow prices for supply/demand constraints:

```python
result = solve_min_cost_flow(problem)

# Access dual values (shadow prices)
for node_id, dual in result.duals.items():
    print(f"Node {node_id}: dual value = {dual:.6f}")

# Dual values indicate marginal cost of changing supply/demand
# For optimal solutions with arc (i,j) at positive flow:
#   cost[i,j] + dual[i] - dual[j] ≈ 0  (complementary slackness)
```

Dual values enable sensitivity analysis to understand how the objective changes with supply/demand perturbations. See `examples/sensitivity_analysis_example.py` for detailed examples and the [Algorithm Guide](docs/algorithm.md#node-potentials-dual-variables) for mathematical background.

## CLI Example

Run the bundled examples to see the solver end-to-end:

```bash
python examples/solve_example.py  # Basic example with dual values
python examples/solve_dimacs_example.py  # DIMACS-style instance
python examples/solve_textbook_transport.py  # Textbook transportation problem
python examples/solve_large_transport.py  # 10×10 transportation instance
python examples/sensitivity_analysis_example.py  # Dual values and shadow prices
python examples/progress_logging_example.py  # Progress monitoring
python examples/solver_options_example.py  # Solver configuration and tuning
python examples/utils_example.py  # Flow analysis utilities
```

These scripts write companion solution files and print detailed results including dual values and solver statistics.

## Problem File Format

Problem instances are JSON documents with the following shape:

- Top-level keys: `directed`, `tolerance`, `nodes`, `edges` (or `arcs`)
- Each node is an object with `id` and optional `supply` (positive supply, negative demand)
- Each edge includes:
  - `tail`, `head` (node IDs)
  - optional `capacity` (omit for infinite), `cost`, and `lower` bound

Undirected graphs set `"directed": false` and must specify finite capacities. During preprocessing, each undirected edge is expanded into a directed arc with symmetric capacity (`[-capacity, capacity]`) so the simplex solver can operate on a standard directed network.

## Testing

- `tests/unit/` – validation, IO, simplex edge cases (pricing, pivots, flow cleanup), and property-based generators
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

- Format/lint with `make lint`
- Run `make test` before submitting changes
- Keep new examples in `examples/` and note structural changes in `AGENTS.md`

Feel free to open issues or PRs—feedback and improvements are welcome.
