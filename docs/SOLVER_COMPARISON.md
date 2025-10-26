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

### Compare Specific Solvers
```bash
# Compare only exact optimization solvers (exclude approximation algorithms)
python benchmarks/scripts/solver_comparison.py --solvers network_solver ortools pulp --limit 10

# Compare just OR-Tools vs PuLP (both C++ backends)
python benchmarks/scripts/solver_comparison.py --solvers ortools pulp --limit 10

# Benchmark single solver
python benchmarks/scripts/solver_comparison.py --solvers network_solver --limit 10

# Compare network_solver vs OR-Tools only
python benchmarks/scripts/solver_comparison.py --solvers network_solver ortools --limit 10
```

The `--solvers` argument accepts a list of solver names:
- `network_solver` - Our network simplex implementation
- `networkx` - NetworkX capacity scaling
- `ortools` - Google OR-Tools network simplex
- `pulp` - PuLP LP formulation with COIN-OR

If `--solvers` is not specified, all available solvers are used.

### Compare on Limited Set
```bash
# First 10 problems
python benchmarks/scripts/solver_comparison.py --limit 10

# With timeout (default: 60 seconds per problem per solver)
python benchmarks/scripts/solver_comparison.py --limit 10 --timeout 30

# Note: Timeout behavior varies by solver:
#   - PuLP: Enforces timeout strictly
#   - network_solver: Uses max_iterations limit instead
#   - OR-Tools: Uses internal heuristics
#   - NetworkX: Fast enough that timeout rarely matters
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

  Average speedup (network_solver / ortools): 150-300x
  ortools is significantly faster on average

Detailed Results:

┌─────────────────────────┬────────┬────────┬───────────────────────────────────────┐
│ Problem                 │ Nodes  │ Arcs   │ Solve Time (ms)                       │
├─────────────────────────┼────────┼────────┼───────────────────────────────────────┤
│ gridgen_8_08a           │    257 │   2056 │ network_solver:1843  ortools:12       │
│ netgen_8_08a            │    256 │   2048 │ network_solver:~1500 ortools:~10      │
│ goto_8_08a              │    256 │   2048 │ network_solver:~8000 ortools:~50      │
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
| network_solver | Baseline | ✓ Optimal | Our implementation (Python) |
| NetworkX | ~14x faster | ⚠️ Sometimes suboptimal | Fast approximation |
| OR-Tools | 150-300x faster | ✓ Optimal | Highly optimized C++ |
| PuLP | Varies | ✓ Optimal | LP formulation (general solver) |

## Interpreting Performance Differences

### Why is OR-Tools faster?
1. **C++ implementation** - Compiled native code vs Python interpreted
2. **Years of optimization** - Google's production-quality code with extensive profiling
3. **Low-level optimizations** - Memory layout, SIMD vectorization, cache optimization
4. **Specialized data structures** - Highly-tuned sparse matrix operations

### Why is network_solver slower?
1. **Python overhead** - Interpreted language with dynamic typing
2. **NumPy abstraction** - While fast, still slower than hand-optimized C++
3. **General-purpose design** - Supports diagnostics, logging, rich features
4. **Newer implementation** - Less profiling and optimization time

### Performance Context

**Honest Assessment:**
- **150-300x gap is NOT typical** - expected Python/C++ gap is 10-50x
- The larger gap indicates network_solver has room for further optimization
- However, fundamental limits of pure Python mean it will never match C++ performance
- **Solution quality**: Both find identical optimal solutions ✓

**When to Use Each Solver:**

**Use OR-Tools when:**
- Speed is critical (production systems, large-scale problems)
- You need maximum performance (150-300x faster)
- You're willing to use a black-box solver

**Use network_solver when:**
- Learning and understanding network simplex algorithm
- Prototyping and Python ecosystem integration
- Code clarity and debuggability are priorities
- Research requiring customization
- Small-to-medium problems (1-2 second solve times acceptable)
- Educational purposes where implementation transparency matters

**Trade-off Summary:**
- **OR-Tools**: Fast ✓✓✓, Optimal ✓, Black box ✗, C++ only
- **network_solver**: Fast ✗, Optimal ✓, Transparent ✓, Pure Python ✓, Educational ✓

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
- Increase timeout: `--timeout 120` (default is 60 seconds)
- Test on smaller problems first: `--limit 5`
- Note: Not all solvers honor the timeout parameter
  - PuLP: Respects timeout
  - network_solver: Controlled by max_iterations instead
  - OR-Tools: Very fast, rarely times out
  - NetworkX: Very fast, rarely times out

### Memory issues on large problems
- Limit concurrent solvers
- Use `--solvers network_solver ortools` to test exact solvers only
- Exclude slow solvers: `--solvers network_solver networkx ortools`

## Performance Recommendations

**Be realistic about performance needs:**

1. **Need maximum speed?** → Use OR-Tools (150-300x faster, production-ready)
2. **Learning/research?** → Use network_solver (clear, customizable, debuggable)
3. **Need quick approximation?** → Use NetworkX (fast but may be suboptimal)
4. **Validation/testing?** → Compare network_solver vs OR-Tools for correctness

**Don't expect:**
- network_solver to match OR-Tools on speed (fundamental Python limitations)
- NetworkX to always find optimal solutions (approximation algorithm)

**Do expect:**
- All optimization solvers (network_solver, OR-Tools, PuLP) to agree on optimal solutions
- network_solver to be useful for problems where 1-2 second solves are acceptable
- Clear trade-offs between speed, transparency, and ease of use

## References

- **SOLVER_COMPARISON_FINDINGS.md**: Detailed analysis of network_solver vs NetworkX
- **Phase 6** in BENCHMARK_SUITE_PLAN.md: Original framework design
- Solver documentation:
  - OR-Tools: https://developers.google.com/optimization
  - NetworkX: https://networkx.org/
  - PuLP: https://coin-or.github.io/pulp/
