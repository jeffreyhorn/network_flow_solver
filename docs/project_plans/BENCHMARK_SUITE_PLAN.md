# Benchmark Suite Development Plan

## Overview

This document outlines a phased plan to establish a comprehensive benchmark suite for comparing our network simplex solver against publicly available network flow problems and other solvers. The goal is to validate correctness and measure performance on standardized test instances without incorporating these into our regular test suite.

## Objectives

1. **Identify** publicly available network flow benchmark problems
2. **Support** multiple problem input formats beyond our JSON format
3. **Organize** benchmark problems in a separate directory structure
4. **Validate** correctness of our solver on standard test instances
5. **Compare** performance against other solvers (e.g., Google OR-Tools, LEMON)
6. **Document** results and performance characteristics

## Project Phases

### Phase 1: Research and Cataloging

**Goal**: Identify and document publicly available benchmark sources with licensing information

**Tasks**:
1. Research benchmark problem repositories
   - DIMACS Implementation Challenge (1st Challenge: Network Flows and Matching, 1990-1991)
   - OR-Library (J.E. Beasley collection)
   - LEMON library benchmark data
   - CommaLAB Multicommodity Flow Problems (University of Pisa)
   - Computer Vision Research Group benchmarks (University of Western Ontario)
   - Network Repository (networkrepository.com)

2. Catalog problem generators
   - NETGEN: General purpose minimum cost flow generator (Klingman et al., 1974)
   - GRIDGEN: Grid-like network generator
   - GOTO: Transportation problem generator
   - GRIDGRAPH: Grid-based graph generator
   - **Note**: Original generators are in Fortran/C with no maintained Python ports

3. Document problem characteristics
   - Problem sizes (nodes, arcs)
   - Problem types (transportation, assignment, general MCF, max flow)
   - Data formats (DIMACS, OR-Library, custom formats)
   - Known optimal solutions (if available)

4. **License audit** (CRITICAL)
   - Document license terms for each benchmark source
   - Identify restrictions on redistribution, mirroring, or derivative works
   - DIMACS: Check specific license terms
   - OR-Library: Review distribution permissions
   - LEMON: Review usage rights
   - Determine if we can redistribute or must provide download scripts
   - Create `benchmarks/metadata/licenses.json` with license information

**Deliverable**: `docs/benchmarks/BENCHMARK_SOURCES.md` documenting all identified sources with licenses

**Estimated Effort**: 2-3 days (extended for license audit)

---

### Phase 2: Directory Structure and Organization

**Goal**: Create unified directory structure for all benchmark-related content

**Tasks**:
1. Create directory structure under single `benchmarks/` root:
   ```
   benchmarks/
   ├── README.md                          # Overview and usage instructions
   ├── problems/                          # Benchmark problem files
   │   ├── dimacs/                        # DIMACS format problems
   │   │   ├── netgen/                    # NETGEN-generated instances
   │   │   ├── gridgen/                   # GRIDGEN instances
   │   │   ├── goto/                      # GOTO instances
   │   │   └── real_world/                # Real-world instances (TIGER/Line, etc.)
   │   ├── or_library/                    # OR-Library instances
   │   ├── lemon/                         # LEMON benchmark instances
   │   ├── commalab/                      # CommaLAB multicommodity instances
   │   └── generated/                     # Locally generated test problems
   ├── metadata/                          # Problem metadata, solutions, licenses
   │   ├── problem_catalog.json           # Catalog of all problems
   │   ├── known_solutions.json           # Known optimal solutions
   │   └── licenses.json                  # License information for datasets
   ├── results/                           # Benchmark results
   │   ├── latest/                        # Most recent benchmark run
   │   └── reports/                       # Generated reports
   ├── solvers/                           # Solver adapter implementations
   │   ├── __init__.py
   │   ├── base.py                        # Base solver adapter interface
   │   ├── ortools_adapter.py
   │   ├── networkx_adapter.py
   │   └── lemon_adapter.py
   ├── parsers/                           # Format parsers (separate from src/)
   │   ├── __init__.py
   │   ├── dimacs.py                      # DIMACS format parser
   │   ├── orlibrary.py                   # OR-Library format parser
   │   └── lemon.py                       # LEMON format parser
   ├── run_benchmark_suite.py             # Main benchmark runner
   ├── run_solver_comparison.py           # Multi-solver comparison runner
   ├── analyze_benchmark_results.py       # Result analysis
   ├── visualize_benchmark_results.py     # Visualization generation
   ├── benchmark_cumulative_speedup.py    # Existing speedup benchmark
   └── benchmark_jit_compilation.py       # Existing JIT benchmark
   ```
   
   **Rationale**: 
   - All benchmark-related content (problems, runners, parsers, results) under single `benchmarks/` root
   - Avoids collision with existing `src/network_solver/io.py` module by placing parsers in `benchmarks/parsers/`
   - Extends existing `benchmarks/` directory structure consistently
   - Clear separation between production code (`src/`) and benchmark infrastructure (`benchmarks/`)

2. Create `.gitignore` entries to exclude large problem files from repository
   - Add `benchmarks/problems/**/*.min` to exclude DIMACS problem files
   - Add `benchmarks/problems/**/*.dat` to exclude data files
   - Consider using Git LFS for a few representative medium-sized instances
   - Provide download scripts instead of committing large files

3. Document directory structure and conventions in `benchmarks/README.md`

**Deliverable**: Directory structure and documentation

**Estimated Effort**: 0.5 day

---

### Phase 3: Format Parser Development

**Goal**: Implement parsers for standard network flow problem formats

**Tasks**:
1. **DIMACS Format Parser** (Priority 1)
   - Implement parser for DIMACS minimum cost flow format
   - Support both standard and extended formats
   - Location: `benchmarks/parsers/dimacs.py` (separate from `src/network_solver/io.py`)
   - Reference: http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm (HTTP only)
   - Convert DIMACS format to our `NetworkProblem` data structure

   DIMACS format structure:
   ```
   c Comment lines
   p min <nodes> <arcs>
   n <node_id> <supply>
   a <tail> <head> <lower> <capacity> <cost>
   ```

2. **OR-Library Format Parser** (Priority 2 - Scoped)
   - **Important**: OR-Library is NOT a single format; it's a collection with varying formats
   - Many OR-Library network flow instances reuse DIMACS format or custom layouts
   - **Scoped approach for MVP**:
     - Start with OR-Library instances that use DIMACS format (reuse parser)
     - Document which OR-Library problem families we support in Phase 1
     - Add family-specific parsers only if clearly documented and valuable
   - Location: `benchmarks/parsers/orlibrary.py`
   - May need multiple sub-parsers for different OR-Library families

3. **LEMON Format Parser** (Priority 3 - Scoped)
   - **Important**: LEMON benchmark data often uses DIMACS variants
   - Research which LEMON instances use standard DIMACS vs. custom formats
   - **Scoped approach for MVP**:
     - Support LEMON instances that use standard DIMACS format
     - Document format variations discovered
     - Add custom parser only if needed and format is documented
   - Location: `benchmarks/parsers/lemon.py`

4. **Format Converter Utilities**
   - Create utility to convert parsed formats to our JSON format
   - Support converting benchmark formats to `NetworkProblem`
   - Location: `benchmarks/parsers/converter.py`

5. **Unit Tests**
   - Test parsers with small hand-crafted examples
   - Validate round-trip conversion (parse → convert to NetworkProblem → solve)
   - Location: `tests/unit/test_benchmark_parsers.py`
   - Keep separate from production code tests

**Deliverable**: DIMACS parser with unit tests; scoped OR-Library/LEMON support

**Estimated Effort**: 3-5 days (extended for format research and scoping)

---

### Phase 4: Benchmark Problem Acquisition

**Goal**: Download and organize benchmark problem instances

**Tasks**:
1. Create download scripts
   - `benchmarks/scripts/download_dimacs.py`
   - `benchmarks/scripts/download_or_library.py`
   - `benchmarks/scripts/download_lemon.py`
   - Scripts should download, verify checksums (if available), and organize files
   - Respect license terms identified in Phase 1

2. Start with small/medium instances
   - Focus on problems with 100-10,000 nodes initially
   - Gradually add larger instances
   - Prioritize problems with known optimal solutions

3. **Problem Generation** (FUTURE EXTENSION - Not in MVP)
   - Original NETGEN/GRIDGEN generators are in Fortran/C
   - No maintained Python ports exist
   - Building Python versions will dominate schedule (weeks of effort)
   - **Recommendation**: Defer to future extension or use external tools
   - **Alternative**: Wrap original C/Fortran tools with Python subprocess calls
   - **For MVP**: Focus on downloading existing benchmark instances

4. Organize metadata
   - Create `benchmarks/metadata/problem_catalog.json` with:
     - Problem name
     - Source (DIMACS, OR-Library, etc.)
     - Size (nodes, arcs)
     - Type (transportation, assignment, general)
     - Known optimal objective (if available)
     - Reference/citation
     - License information
     - Format (DIMACS, custom, etc.)
     - File path (relative to benchmarks/)

**Deliverable**: Collection of downloaded benchmark problems with metadata

**Estimated Effort**: 2-3 days

---

### Phase 5: Benchmark Runner Development

**Goal**: Create infrastructure to run benchmarks and collect results

**Tasks**:
1. **Benchmark Runner Script**
   - Location: `benchmarks/run_benchmark_suite.py`
   - Features:
     - Load problems from `metadata/problem_catalog.json`
     - Parse problems using appropriate format parser
     - Solve using our solver
     - Collect results: objective, runtime, iterations, status
     - Handle timeouts (e.g., 5 minutes per problem)
     - Handle memory limits (requires platform-specific tracking)
     - Compare against known optimal solutions
     - Generate detailed logs

2. **Performance Metrics** (with implementation notes)
   - Runtime (total, per iteration) - straightforward with `time.time()`
   - **Memory usage** - requires `psutil` library or platform-specific tools
     - Add `psutil` as optional dependency for benchmarking
     - Gracefully degrade if unavailable (skip memory tracking)
     - Document that memory tracking requires `psutil`
   - Iteration count - available from `FlowResult`
   - Convergence characteristics - track objective value per iteration

3. **Result Storage**
   - Store results in structured format (JSON/CSV)
   - Include: problem name, solver, objective, runtime, iterations, status, correctness, memory (if available)
   - Location: `benchmarks/results/latest/`
   - Archive previous runs with timestamps

4. **Correctness Validation**
   - Compare computed objective against known optimal
   - Use tolerance for floating-point comparison (e.g., 1e-4 relative error)
   - Flag discrepancies for investigation
   - Validate flow conservation and capacity constraints using existing validation functions

**Deliverable**: Benchmark runner with result collection

**Estimated Effort**: 2-3 days

**Dependencies**: Add `psutil` as optional dependency for memory tracking

---

### Phase 6: Solver Comparison Framework

**Goal**: Compare our solver against other established solvers

**Tasks**:
1. **Identify Comparison Solvers** (with performance expectations)
   - **Primary baseline: Google OR-Tools** (min_cost_flow)
     - Highly optimized C++ implementation
     - Use as primary performance target for "2-5x" comparison
   - **LEMON** (NetworkSimplex, CostScaling, CapacityScaling)
     - C++ implementations, excellent performance
     - Good secondary baseline
   - **NetworkX** (min_cost_flow) - **CORRECTNESS VALIDATION ONLY**
     - Pure Python implementation, correctness-oriented
     - Significantly slower than OR-Tools/LEMON (10-100x+)
     - **Do not use NetworkX for performance comparisons**
     - Use only to validate correctness against third-party implementation
   - SciPy (if applicable - likely uses NetworkX internally)
   - Commercial solvers (CPLEX, Gurobi) - optional, licensing required

2. **Implement Solver Adapters**
   - Create unified interface for different solvers
   - Location: `benchmarks/solvers/`
   - `base.py`: Abstract base class defining solver interface
   - `ortools_adapter.py`: OR-Tools adapter
   - `lemon_adapter.py`: LEMON adapter (may require subprocess calls to C++ binary)
   - `networkx_adapter.py`: NetworkX adapter (correctness validation only)
   - Measure runtime and collect results consistently

3. **Comparison Runner**
   - Run same problems through multiple solvers
   - Location: `benchmarks/run_solver_comparison.py`
   - Handle solver-specific requirements (installation, licensing)
   - Gracefully skip unavailable solvers
   - Separate correctness validation from performance comparison

4. **Fair Comparison Practices**
   - Use single-threaded execution for all solvers where possible
   - Warm-up runs to account for JIT compilation
   - Multiple runs for statistical significance (5+ runs)
   - Report mean ± std dev for runtimes
   - Document solver versions used

**Deliverable**: Multi-solver comparison framework

**Estimated Effort**: 3-4 days

---

### Phase 7: Reporting and Visualization

**Goal**: Create comprehensive reports and visualizations of benchmark results

**Tasks**:
1. **Result Aggregation**
   - Location: `benchmarks/analyze_benchmark_results.py`
   - Aggregate results across problem families
   - Compute summary statistics
   - Separate correctness from performance analysis

2. **Visualizations**
   - Performance profiles (Dolan-Moré plots) - OR-Tools/LEMON only
   - Runtime comparisons (scatter plots, bar charts)
   - Scaling analysis (runtime vs. problem size)
   - Success rate by problem type
   - Correctness validation summary (NetworkX comparison)
   - Location: `benchmarks/visualize_benchmark_results.py`

3. **Reports**
   - Generate markdown/HTML reports
   - Summary tables: solver × problem family
   - Detailed per-problem results
   - Location: `benchmarks/results/reports/`
   - Clearly separate correctness and performance sections

4. **Documentation**
   - Document how to run benchmarks
   - Interpret results
   - Reproduce experiments
   - Location: `docs/benchmarks/RUNNING_BENCHMARKS.md`

**Deliverable**: Visualization and reporting tools

**Estimated Effort**: 2-3 days

---

### Phase 8: Continuous Benchmarking (Optional)

**Goal**: Integrate benchmarking into development workflow

**Tasks**:
1. **Regression Testing**
   - Run subset of benchmarks on each PR
   - Flag performance regressions (e.g., >10% slowdown)
   - Separate CI job for benchmarks (longer timeout)

2. **Performance Tracking**
   - Track performance metrics over time
   - Detect performance regressions
   - Visualize performance trends

3. **GitHub Actions Integration**
   - Optional: nightly benchmark runs
   - Store results as artifacts
   - Post summary to PR comments

**Deliverable**: Automated benchmark CI/CD

**Estimated Effort**: 2-3 days

---

## Implementation Order

### Minimal Viable Benchmark Suite (MVP)
1. Phase 1: Research and Cataloging (with license audit)
2. Phase 2: Directory Structure
3. Phase 3: DIMACS Parser only (scoped OR-Library/LEMON)
4. Phase 4: Small set of DIMACS problems (download only, no generation)
5. Phase 5: Basic benchmark runner (with optional memory tracking)
6. Phase 6: OR-Tools comparison + NetworkX validation

**Estimated Total for MVP**: 12-16 days (extended for scoping and license audit)

### Full Implementation
Complete all phases 1-7

**Estimated Total**: 19-26 days (extended for scoping and research)

### With Continuous Integration
Complete all phases 1-8

**Estimated Total**: 21-29 days

---

## Success Criteria

### Correctness
- [ ] Solver produces optimal solutions on all benchmark problems with known optima
- [ ] Solution values match within tolerance (< 0.01% relative error)
- [ ] All flows satisfy conservation and capacity constraints
- [ ] Results validated against NetworkX on subset of problems

### Performance
- [ ] Solver completes all small/medium benchmarks (< 1000 nodes) within reasonable time (< 60s)
- [ ] **Competitive performance (within 2-5x) compared to OR-Tools on small/medium problems**
- [ ] Identify problem classes where our solver excels or struggles
- [ ] Document performance characteristics vs. LEMON

### Infrastructure
- [ ] Easy to add new benchmark problems
- [ ] Easy to run benchmarks and generate reports
- [ ] Well-documented process for reproducing results
- [ ] Clear separation between production code and benchmark infrastructure

---

## Dependencies and Prerequisites

### Software Dependencies
- Benchmark problem parsers: no new dependencies (standard library)
- **Optional for memory tracking**: `psutil` (gracefully degrade if unavailable)
- Comparison solvers (optional, for Phase 6):
  - `ortools` (Google OR-Tools) - **primary performance baseline**
  - `networkx` (already used in visualization) - **correctness validation only**
  - `lemon` (may require C++ bindings or command-line interface)

### Data Dependencies
- Benchmark problem files (to be downloaded, not committed)
- Known optimal solutions (may not exist for all problems)
- License information for each benchmark source

### Development Dependencies
- For visualization: `matplotlib`, `pandas` (already available)
- For statistical analysis: `scipy`, `numpy` (already available)

---

## Risks and Mitigations

### Risk 1: Problem Format Complexity
**Risk**: Some benchmark formats may be complex or poorly documented (OR-Library, LEMON variants)
**Mitigation**: 
- Start with well-documented DIMACS format
- Scope OR-Library/LEMON to DIMACS-compatible instances initially
- Document format variations discovered during Phase 3
- Add custom parsers only for well-documented, valuable formats

### Risk 2: Large Problem Files
**Risk**: Benchmark files may be very large (>100MB)
**Mitigation**: 
- Use download scripts; exclude from repository
- Optional: Git LFS for a few representative medium-sized instances
- Document download process clearly

### Risk 3: Solver Availability
**Risk**: Comparison solvers may have complex installation or licensing requirements
**Mitigation**: 
- Make comparisons optional; gracefully skip unavailable solvers
- Focus on OR-Tools (easy pip install) as primary baseline
- NetworkX already available (correctness validation)
- LEMON optional (may require manual compilation)

### Risk 4: Performance Expectations
**Risk**: Our Python solver will be slower than highly optimized C++ implementations (OR-Tools, LEMON)
**Mitigation**: 
- Set realistic expectations: 2-5x slower than OR-Tools is acceptable
- Focus on correctness first
- Identify specific problem classes for optimization
- Document trade-offs (Python vs. C++, clarity vs. raw speed)

### Risk 5: Known Solutions Unavailable
**Risk**: Many benchmark problems may not have published optimal solutions
**Mitigation**: 
- Compare against multiple solvers for cross-validation
- Use OR-Tools + NetworkX agreement as correctness signal
- Flag discrepancies for investigation
- Document confidence level for each problem

### Risk 6: License Restrictions
**Risk**: Some benchmark datasets may have restrictive licenses prohibiting redistribution
**Mitigation**:
- Comprehensive license audit in Phase 1
- Provide download scripts instead of committing files
- Document license terms for each source
- Only redistribute if explicitly permitted

### Risk 7: Problem Generator Complexity
**Risk**: Building Python versions of NETGEN/GRIDGEN will take weeks
**Mitigation**:
- Defer problem generation to future extension (not in MVP)
- Focus on downloading existing benchmark instances
- Consider wrapping original C/Fortran tools instead of rewriting

---

## Future Extensions

1. **Problem Generators**: Implement Python wrappers for NETGEN/GRIDGEN C/Fortran tools, or implement from scratch

2. **Multi-commodity Flow Problems**: Extend to multicommodity network flow benchmarks (CommaLAB dataset)

3. **Large-Scale Problems**: Add very large instances (>100K nodes) to test scalability

4. **Real-World Problems**: Incorporate real-world network datasets (transportation networks, logistics, etc.)

5. **Algorithm Variants**: Benchmark different algorithmic choices (pricing strategies, basis update methods)

6. **Parallel Benchmarking**: Distribute benchmark runs across multiple machines

7. **Additional Format Support**: Add parsers for OR-Library/LEMON format variants as needed

---

## References

1. Klingman, D., A. Napier, and J. Stutz. "NETGEN: A Program for Generating Large Scale Capacitated Assignment, Transportation, and Minimum Cost Flow Network Problems." Management Science 20.5 (1974): 814-821.

2. Beasley, J.E. "OR-Library: distributing test problems by electronic mail." Journal of the Operational Research Society 41.11 (1990): 1069-1072.

3. Johnson, D.S. and McGeoch, C.C. (Eds.). "Network Flows and Matching: First DIMACS Implementation Challenge." DIMACS Series in Discrete Mathematics and Theoretical Computer Science, Volume 12 (1993).

4. LEMON Library Benchmark Data: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData

5. DIMACS Min-Cost Flow Problems: http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm (HTTP only - HTTPS not available)

6. CommaLAB Multicommodity Flow Problems: https://commalab.di.unipi.it/datasets/mmcf/

---

## Appendix: DIMACS Format Specification

### Minimum Cost Flow Format

```
c This is a comment line
c Problem line: p min <number of nodes> <number of arcs>
p min 4 5
c Node descriptors: n <node id> <supply/demand>
c Positive supply = source, Negative supply = sink, Zero = transshipment
n 1 20
n 2 -10
n 3 -10
n 4 0
c Arc descriptors: a <tail> <head> <lower bound> <capacity> <cost>
a 1 2 0 15 2
a 1 3 0 10 3
a 2 4 0 10 1
a 3 4 0 10 2
a 4 1 0 5 0
```

### Field Descriptions
- `c`: Comment line
- `p min <nodes> <arcs>`: Problem line declaring min-cost flow problem
- `n <id> <supply>`: Node with supply (positive = source, negative = sink)
- `a <tail> <head> <lower> <upper> <cost>`: Arc with bounds and cost

### Constraints
- Node IDs are integers from 1 to n
- Supply/demand must balance: Σ supply = 0
- Lower bounds ≤ capacity
- Costs are integers or floating-point numbers

---

## Appendix: Directory Structure Rationale

### Why `benchmarks/parsers/` instead of `src/network_solver/io/`?

**Problem**: `src/network_solver/io.py` already exists as a module. Adding `src/network_solver/io/dimacs_parser.py` would require refactoring `io.py` into a package, which affects production code.

**Solution**: Place benchmark-specific parsers in `benchmarks/parsers/`:
- **Separation of concerns**: Benchmark infrastructure separate from production code
- **No collision**: Avoids conflicts with existing `io.py` module
- **Cleaner imports**: Benchmark scripts import from `benchmarks.parsers`, production code imports from `network_solver.io`
- **Easier maintenance**: Changes to benchmark parsers don't affect production code
- **Optional dependencies**: Benchmark parsers can use additional dependencies without adding them to production requirements

If needed, parsers can be promoted to `src/network_solver/io/` package in the future after proper refactoring.
