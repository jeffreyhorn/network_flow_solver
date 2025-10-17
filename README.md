# Network Programming Solver

[![CI](https://github.com/jeffreyhorn/network_flow_solver/workflows/CI/badge.svg)](https://github.com/jeffreyhorn/network_flow_solver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jeffreyhorn/network_flow_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyhorn/network_flow_solver)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Python implementation of the network simplex algorithm for classic minimum-cost flow problems with node supplies/demands and optional capacities/lower bounds. The package exposes both a programmatic API and JSON-oriented utilities for loading/saving problem instances.

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

## CLI Example

Run the bundled example to see the solver end-to-end:

```bash
python examples/solve_example.py
python examples/solve_dimacs_example.py  # DIMACS-style instance
python examples/solve_textbook_transport.py  # textbook transportation problem
python examples/solve_large_transport.py  # 10×10 transportation instance
```

These scripts write companion solution files and print a one-line summary.

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
