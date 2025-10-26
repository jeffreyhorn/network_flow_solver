# Solver Comparison Framework

The solver comparison framework allows you to benchmark network_solver against other network flow implementations to validate correctness and measure performance.

## Available Solvers

### Always Available
- **network_solver** - Our network simplex implementation (this package)
- **NetworkX** - NetworkX's capacity scaling algorithm (installed with visualization extras)

### Optional Solvers
Install with: `pip install 'network-flow-solver[comparison]'`

- **Google OR-Tools** - Highly optimized C++ network simplex (`pip install ortools`)
- **PuLP** - LP formulation with COIN-OR backend (`pip install pulp`)

### Not Available via pip
- **LEMON** - Requires manual C++ compilation and Python bindings

## Installation

### Basic Usage (network_solver + NetworkX)
```bash
# Already available if you have visualization extras installed
pip install 'network-flow-solver[visualization]'
```

### With All Comparison Solvers
```bash
# Install all optional comparison solvers
pip install 'network-flow-solver[comparison]'
```

### Individual Solvers
```bash
# Install specific solvers
pip install ortools  # Google OR-Tools
pip install pulp     # PuLP with COIN-OR
```

## Usage

### List Available Solvers
```bash
python benchmarks/scripts/solver_comparison.py --list-solvers
```

Output:
```
AVAILABLE NETWORK FLOW SOLVERS
======================================================================

Available solvers:
  - Network Solver (network_solver) v0.1.0
    Our network simplex implementation with Devex/Dantzig pricing
  - NetworkX (networkx) v3.5
    NetworkX capacity scaling algorithm (fast approximation)
  - Google OR-Tools (ortools) v9.8.3296
    Google OR-Tools network simplex (highly optimized C++)
  - PuLP (pulp) v2.8.0
    PuLP LP formulation with COIN-OR/GLPK backend
```

### Compare on Specific Problems
```bash
# Single problem
python benchmarks/scripts/solver_comparison.py \
  --problems benchmarks/problems/lemon/gridgen/gridgen_8_08a.min

# Multiple problems
python benchmarks/scripts/solver_comparison.py \
  --problems benchmarks/problems/lemon/goto/*.min

# Pattern matching
python benchmarks/scripts/solver_comparison.py \
  --pattern "benchmarks/problems/lemon/**/*_8_08a.min"
```

### Compare on Limited Set
```bash
# First 10 problems
python benchmarks/scripts/solver_comparison.py --limit 10

# With timeout
python benchmarks/scripts/solver_comparison.py --limit 10 --timeout 30
```

### Save Report
```bash
python benchmarks/scripts/solver_comparison.py \
  --limit 10 \
  --output comparison_report.txt
```

## Sample Output

```
Comparing 3 solvers on 3 problems
Solvers: network_solver, networkx, ortools

  Comparing: gridgen_8_08a... Done
  Comparing: netgen_8_08a... Done
  Comparing: goto_8_08a... 
  ⚠️  Solution quality difference: goto_8_08a
    network_solver: 560870539 ✓ BEST
    ortools: 560870539 ✓ BEST
    networkx: 673664865
    Difference: 20.1% (significant!)
Done

================================================================================
SOLVER COMPARISON REPORT
================================================================================

Summary:
  Total problems: 3

Success Rate:
  network_solver      : 3/3 (100.0%)
  networkx            : 3/3 (100.0%)
  ortools             : 3/3 (100.0%)

Performance Comparison (3 problems both solved):

  Average speedup (network_solver / ortools): 4.2x
  ortools is faster on average

Detailed Results:

┌─────────────────────────┬────────┬────────┬───────────────────────────────────────┐
│ Problem                 │ Nodes  │ Arcs   │ Solve Time (ms)                       │
├─────────────────────────┼────────┼────────┼───────────────────────────────────────┤
│ gridgen_8_08a           │    257 │   2056 │ network_solver:1720  ortools:412      │
│ netgen_8_08a            │    256 │   2048 │ network_solver:1381  ortools:328      │
│ goto_8_08a              │    256 │   2048 │ network_solver:7548  ortools:1842     │
└─────────────────────────┴────────┴────────┴───────────────────────────────────────┘

Winner (Fastest Solver) per Problem:
  ortools             : 3 wins
```

## Understanding the Results

### Performance Metrics

**Solve Time**: Total time to solve the problem (milliseconds)
- Includes parsing, setup, and solving
- Lower is better

**Speedup**: Ratio of solve times (baseline_time / competitor_time)
- `>1.0` means competitor is faster
- `<1.0` means baseline is faster

### Solution Quality

The framework automatically detects when solvers disagree on the objective value:

```
⚠️  Solution quality difference: problem_name
  network_solver: 560870539 ✓ BEST
  networkx: 673664865
  Difference: 20.1% (significant!)
```

**Important Findings**:
- **network_solver** always finds the true optimal solution
- **NetworkX** sometimes returns suboptimal solutions (20% worse on some problems)
- **OR-Tools** matches network_solver's solution quality

### Typical Results

| Solver | Speed | Solution Quality | Notes |
|--------|-------|------------------|-------|
| network_solver | Baseline | ✓ Optimal | Our implementation |
| NetworkX | ~14x faster | ⚠️ Sometimes suboptimal | Fast approximation |
| OR-Tools | ~4x faster | ✓ Optimal | Highly optimized C++ |
| PuLP | ~2x slower | ✓ Optimal | LP formulation (general solver) |

## Interpreting Performance Differences

### Why is OR-Tools faster?
1. **C++ implementation** - Compiled vs Python
2. **Years of optimization** - Google's production-quality code
3. **Low-level optimizations** - Memory layout, vectorization

### Why is network_solver competitive?
1. **Good algorithms** - Devex pricing, adaptive tuning
2. **Efficient Python** - NumPy vectorization, smart data structures
3. **Recent optimizations** - 5.17x speedup from Phases 1-3

### Being 4x slower than OR-Tools is excellent!
- **Python vs C++**: Typical gap is 10-100x
- **Solution quality**: Both find true optimal
- **Trade-off**: Python readability/maintainability vs C++ performance

## Programmatic Usage

```python
from benchmarks.parsers.dimacs import parse_dimacs_file
from benchmarks.solvers import get_available_solvers

# Get available solvers
solvers = get_available_solvers()
print(f"Found {len(solvers)} solvers")

# Solve with each solver
problem = parse_dimacs_file("benchmarks/problems/lemon/gridgen/gridgen_8_08a.min")

for solver_class in solvers:
    result = solver_class.solve(problem, timeout_s=30.0)
    print(f"{solver_class.name}: {result.status} in {result.solve_time_ms:.2f}ms")
```

## Adding New Solvers

To add a new solver, create an adapter in `benchmarks/solvers/`:

```python
# benchmarks/solvers/my_solver_adapter.py
from .base import SolverAdapter, SolverResult
import my_solver_library

class MySolverAdapter(SolverAdapter):
    name = "mysolver"
    display_name = "My Solver"
    description = "Description of my solver"
    
    @classmethod
    def solve(cls, problem, timeout_s=60.0):
        # Convert problem to solver's format
        # Solve
        # Return SolverResult
        ...
    
    @classmethod
    def is_available(cls):
        try:
            import my_solver_library
            return True
        except ImportError:
            return False
```

Then add it to `benchmarks/solvers/__init__.py`:
```python
try:
    from .my_solver_adapter import MySolverAdapter
    _AVAILABLE_SOLVERS.append(MySolverAdapter)
except ImportError:
    pass
```

## Troubleshooting

### "No problems found!"
- Check your `--pattern` or `--problems` argument
- Ensure problem files exist in the benchmarks directory

### Solver import errors
- Install optional solvers: `pip install ortools pulp`
- Check with `--list-solvers` to see which are available

### Timeout issues
- Increase timeout: `--timeout 120`
- Test on smaller problems first: `--limit 5`

### Memory issues on large problems
- Limit concurrent solvers
- Use `--solvers network_solver networkx` to test subset

## Performance Optimization Tips

Based on comparison results, consider:

1. **If you need speed**: Use OR-Tools for production
2. **If you need control**: Use network_solver (dual values, diagnostics)
3. **If you need quick approximation**: Use NetworkX
4. **For validation**: Compare against multiple solvers

## References

- **SOLVER_COMPARISON_FINDINGS.md**: Detailed analysis of network_solver vs NetworkX
- **Phase 6** in BENCHMARK_SUITE_PLAN.md: Original framework design
- Solver documentation:
  - OR-Tools: https://developers.google.com/optimization
  - NetworkX: https://networkx.org/
  - PuLP: https://coin-or.github.io/pulp/
