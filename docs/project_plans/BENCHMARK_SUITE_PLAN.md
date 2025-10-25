# Benchmark Suite Development Plan

## Overview

This document outlines a phased plan to establish a comprehensive benchmark suite for comparing our network simplex solver against publicly available network flow problems and other solvers. The goal is to validate correctness and measure performance on standardized test instances without incorporating these into our regular test suite.

## Objectives

1. **Identify** publicly available network flow benchmark problems
2. **Support** multiple problem input formats beyond our JSON format
3. **Organize** benchmark problems in a separate directory structure
4. **Validate** correctness of our solver on standard test instances
5. **Compare** performance against other solvers (e.g., Google OR-Tools, NetworkX, LEMON)
6. **Document** results and performance characteristics

## Project Phases

### Phase 1: Research and Cataloging

**Goal**: Identify and document publicly available benchmark sources

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

3. Document problem characteristics
   - Problem sizes (nodes, arcs)
   - Problem types (transportation, assignment, general MCF, max flow)
   - Data formats (DIMACS, OR-Library, custom formats)
   - Known optimal solutions (if available)

**Deliverable**: `docs/benchmarks/BENCHMARK_SOURCES.md` documenting all identified sources

**Estimated Effort**: 1-2 days

---

### Phase 2: Directory Structure and Organization

**Goal**: Create directory structure for benchmark problems separate from test suite

**Tasks**:
1. Create directory structure:
   ```
   benchmark_problems/
   ├── README.md                    # Overview and usage instructions
   ├── dimacs/                      # DIMACS format problems
   │   ├── netgen/                  # NETGEN-generated instances
   │   ├── gridgen/                 # GRIDGEN instances
   │   ├── goto/                    # GOTO instances
   │   └── real_world/              # Real-world instances (TIGER/Line, etc.)
   ├── or_library/                  # OR-Library instances
   ├── lemon/                       # LEMON benchmark instances
   ├── commalab/                    # CommaLAB multicommodity instances
   ├── generated/                   # Locally generated test problems
   └── metadata/                    # Problem metadata, solutions, references
       ├── problem_catalog.json     # Catalog of all problems
       └── known_solutions.json     # Known optimal solutions
   ```

2. Create `.gitignore` entries to exclude large problem files from repository
   - Consider using Git LFS for larger instances
   - Or provide download scripts instead of committing files

3. Document directory structure and conventions in `benchmark_problems/README.md`

**Deliverable**: Directory structure and documentation

**Estimated Effort**: 0.5 day

---

### Phase 3: Format Parser Development

**Goal**: Implement parsers for standard network flow problem formats

**Tasks**:
1. **DIMACS Format Parser** (Priority 1)
   - Implement parser for DIMACS minimum cost flow format
   - Support both standard and extended formats
   - Location: `src/network_solver/io/dimacs_parser.py`
   - Reference: http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm (HTTP only)

   DIMACS format structure:
   ```
   c Comment lines
   p min <nodes> <arcs>
   n <node_id> <supply>
   a <tail> <head> <lower> <capacity> <cost>
   ```

2. **OR-Library Format Parser** (Priority 2)
   - Research specific OR-Library network flow format
   - Implement parser
   - Location: `src/network_solver/io/orlibrary_parser.py`

3. **LEMON Format Parser** (Priority 3)
   - Support LEMON library format (if different from DIMACS)
   - Location: `src/network_solver/io/lemon_parser.py`

4. **Format Converter Utilities**
   - Create utility to convert between formats
   - Support converting benchmark formats to our JSON format
   - Location: `src/network_solver/io/format_converter.py`

5. **Unit Tests**
   - Test parsers with small hand-crafted examples
   - Validate round-trip conversion (parse → convert to JSON → solve)
   - Location: `tests/unit/test_dimacs_parser.py`, etc.

**Deliverable**: Format parsers with unit tests

**Estimated Effort**: 3-4 days

---

### Phase 4: Benchmark Problem Acquisition

**Goal**: Download and organize benchmark problem instances

**Tasks**:
1. Create download scripts
   - `scripts/download_dimacs_benchmarks.py`
   - `scripts/download_or_library_benchmarks.py`
   - `scripts/download_lemon_benchmarks.py`
   - Scripts should download, verify checksums (if available), and organize files

2. Start with small/medium instances
   - Focus on problems with 100-10,000 nodes initially
   - Gradually add larger instances

3. Generate problems using NETGEN/GRIDGEN
   - Install or implement Python versions of generators
   - Generate variety of problem sizes and characteristics
   - Document generation parameters

4. Organize metadata
   - Create `problem_catalog.json` with:
     - Problem name
     - Source
     - Size (nodes, arcs)
     - Type (transportation, assignment, general)
     - Known optimal objective (if available)
     - Reference/citation
     - Format
     - File path

**Deliverable**: Collection of benchmark problems with metadata

**Estimated Effort**: 2-3 days

---

### Phase 5: Benchmark Runner Development

**Goal**: Create infrastructure to run benchmarks and collect results

**Tasks**:
1. **Benchmark Runner Script**
   - Location: `benchmarks/run_benchmark_suite.py`
   - Features:
     - Load problems from `problem_catalog.json`
     - Parse problems using appropriate format parser
     - Solve using our solver
     - Collect results: objective, runtime, iterations, status
     - Handle timeouts (e.g., 5 minutes per problem)
     - Handle memory limits
     - Compare against known optimal solutions
     - Generate detailed logs

2. **Result Storage**
   - Store results in structured format (JSON/CSV)
   - Include: problem name, solver, objective, runtime, iterations, status, correctness
   - Location: `benchmark_problems/results/`

3. **Correctness Validation**
   - Compare computed objective against known optimal
   - Use tolerance for floating-point comparison (e.g., 1e-4 relative error)
   - Flag discrepancies for investigation
   - Validate flow conservation and capacity constraints

4. **Performance Metrics**
   - Runtime (total, per iteration)
   - Memory usage
   - Iteration count
   - Convergence characteristics

**Deliverable**: Benchmark runner with result collection

**Estimated Effort**: 2-3 days

---

### Phase 6: Solver Comparison Framework

**Goal**: Compare our solver against other established solvers

**Tasks**:
1. **Identify Comparison Solvers**
   - Google OR-Tools (min_cost_flow)
   - NetworkX (min_cost_flow)
   - LEMON (NetworkSimplex, CostScaling, CapacityScaling, CycleCanceling)
   - SciPy (if applicable)
   - Commercial solvers (CPLEX, Gurobi) - optional, licensing required

2. **Implement Solver Adapters**
   - Create unified interface for different solvers
   - Location: `benchmarks/solvers/`
   - Adapters for each solver to translate our problem format
   - Measure runtime and collect results consistently

3. **Comparison Runner**
   - Run same problems through multiple solvers
   - Location: `benchmarks/run_solver_comparison.py`
   - Handle solver-specific requirements (installation, licensing)
   - Gracefully skip unavailable solvers

4. **Fair Comparison Practices**
   - Use single-threaded execution for all solvers
   - Warm-up runs to account for JIT compilation
   - Multiple runs for statistical significance
   - Report mean ± std dev for runtimes

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

2. **Visualizations**
   - Performance profiles (Dolan-Moré plots)
   - Runtime comparisons (scatter plots, bar charts)
   - Scaling analysis (runtime vs. problem size)
   - Success rate by problem type
   - Location: `benchmarks/visualize_benchmark_results.py`

3. **Reports**
   - Generate markdown/HTML reports
   - Summary tables: solver × problem family
   - Detailed per-problem results
   - Location: `benchmark_problems/results/reports/`

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
1. Phase 1: Research and Cataloging
2. Phase 2: Directory Structure
3. Phase 3: DIMACS Parser only
4. Phase 4: Small set of DIMACS problems
5. Phase 5: Basic benchmark runner

**Estimated Total for MVP**: 7-10 days

### Full Implementation
Complete all phases 1-7

**Estimated Total**: 15-20 days

### With Continuous Integration
Complete all phases 1-8

**Estimated Total**: 17-23 days

---

## Success Criteria

### Correctness
- [ ] Solver produces optimal solutions on all benchmark problems with known optima
- [ ] Solution values match within tolerance (< 0.01% relative error)
- [ ] All flows satisfy conservation and capacity constraints

### Performance
- [ ] Solver completes all small/medium benchmarks (< 1000 nodes) within reasonable time (< 60s)
- [ ] Competitive performance (within 2-5x) compared to established solvers on small/medium problems
- [ ] Identify problem classes where our solver excels or struggles

### Infrastructure
- [ ] Easy to add new benchmark problems
- [ ] Easy to run benchmarks and generate reports
- [ ] Well-documented process for reproducing results

---

## Dependencies and Prerequisites

### Software Dependencies
- Benchmark problem parsers: no new dependencies (standard library)
- Comparison solvers (optional, for Phase 6):
  - `ortools` (Google OR-Tools)
  - `networkx` (already used in visualization)
  - `lemon` (may require C++ bindings or command-line interface)

### Data Dependencies
- Benchmark problem files (to be downloaded)
- Known optimal solutions (may not exist for all problems)

### Development Dependencies
- For visualization: `matplotlib`, `pandas` (already available)
- For statistical analysis: `scipy`, `numpy` (already available)

---

## Risks and Mitigations

### Risk 1: Problem Format Complexity
**Risk**: Some benchmark formats may be complex or poorly documented
**Mitigation**: Start with well-documented DIMACS format; add others incrementally

### Risk 2: Large Problem Files
**Risk**: Benchmark files may be very large (>100MB)
**Mitigation**: Use Git LFS or download scripts; exclude from repository

### Risk 3: Solver Availability
**Risk**: Comparison solvers may have complex installation or licensing requirements
**Mitigation**: Make comparisons optional; focus on OR-Tools and NetworkX (open source, easy to install)

### Risk 4: Performance Expectations
**Risk**: Our solver may be slower than highly optimized C++ implementations
**Mitigation**: Focus on correctness first; identify specific problem classes for optimization; document trade-offs (Python vs. C++)

### Risk 5: Known Solutions Unavailable
**Risk**: Many benchmark problems may not have published optimal solutions
**Mitigation**: Compare against multiple solvers for cross-validation; flag discrepancies for investigation

---

## Future Extensions

1. **Multi-commodity Flow Problems**: Extend to multicommodity network flow benchmarks (CommaLAB dataset)

2. **Large-Scale Problems**: Add very large instances (>100K nodes) to test scalability

3. **Real-World Problems**: Incorporate real-world network datasets (transportation networks, logistics, etc.)

4. **Problem Generators**: Implement Python versions of NETGEN, GRIDGEN for controlled experiments

5. **Algorithm Variants**: Benchmark different algorithmic choices (pricing strategies, basis update methods)

6. **Parallel Benchmarking**: Distribute benchmark runs across multiple machines

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
