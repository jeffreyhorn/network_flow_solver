# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-17

### Added
- Initial release with proper Python packaging
- `pyproject.toml` for modern Python packaging (PEP 621 compliant)
- Package versioning (`__version__ = "0.1.0"`)
- `py.typed` marker for type checking support
- `LICENSE` file (MIT License)
- `INSTALL.md` with detailed installation instructions
- `verify_install.py` script to test package installation
- Updated `.gitignore` to include build artifacts
- Updated `requirements.txt` with clear dependency organization
- Enhanced README.md with installation section
- **Custom exception hierarchy** (`exceptions.py`)
  - `NetworkSolverError` - Base exception for all solver errors
  - `InvalidProblemError` - Malformed problem definitions
  - `InfeasibleProblemError` - No feasible solution exists (tracks iterations)
  - `UnboundedProblemError` - Unbounded objective (includes diagnostic info)
  - `NumericalInstabilityError` - Numerical computation issues
  - `SolverConfigurationError` - Invalid solver parameters
  - `IterationLimitError` - Iteration limit reached (optional)
- Test suite for exception hierarchy (`tests/unit/test_exceptions.py`)

### Changed
- Package is now installable via `pip install -e .`
- No longer requires manual `PYTHONPATH` manipulation
- Dependencies automatically installed via pip
- **Replaced generic exceptions with specific custom exceptions:**
  - `ValueError` → `InvalidProblemError` (better error messages)
  - `KeyError` → `InvalidProblemError` (consistent error handling)
  - `RuntimeError` → `UnboundedProblemError` (includes diagnostic info)
- All exceptions now provide detailed, actionable error messages
- Exceptions include contextual information (arc details, iteration counts, etc.)

### Documentation
- Comprehensive installation guide with platform-specific notes
- Troubleshooting section for common installation issues
- Clear distinction between runtime and development dependencies
- Exception handling guide in README.md with examples
- Updated AGENTS.md to accurately reflect project structure

### Infrastructure
- Configured pytest, mypy, ruff, and coverage tools in pyproject.toml
- Optional dependency groups for development and performance
- Type hints fully configured for mypy checking
- **GitHub Actions CI/CD workflows:**
  - `ci.yml` - Comprehensive testing across Linux, macOS, Windows
  - Multi-job workflow: lint, typecheck, test, coverage, examples, build
  - Coverage reporting with Codecov integration
  - Artifact uploads for coverage reports and build distributions
  - `release.yml` - Automated PyPI publishing on GitHub releases
  - `dependency-review.yml` - Security scanning for dependencies
- CI badges added to README
- Workflow documentation in `.github/workflows/README.md`

## [Unreleased]

### Added
- **Dual values (node potentials) for sensitivity analysis**
  - `FlowResult.duals` field containing shadow prices for all nodes
  - Enables marginal cost analysis and sensitivity analysis
  - Dual values automatically computed and returned by solver
  - JSON serialization/deserialization support for dual values
  - Exported `build_problem()` function for programmatic problem creation
- **New test suite for dual values** (`tests/unit/test_dual_values.py`)
  - Complementary slackness verification
  - Shadow price interpretation tests
  - Sensitivity analysis validation (6 new tests)
- **New example: `sensitivity_analysis_example.py`**
  - Demonstrates shadow price interpretation
  - Shows marginal cost analysis with supply/demand changes
  - Verifies complementary slackness conditions
- Updated `solve_example.py` to display dual values

### Changed
- Enhanced `FlowResult` dataclass with comprehensive documentation
  - Added detailed docstring explaining all fields
  - Documented dual values and their interpretation
- All solution JSON files now include `duals` field

### Fixed
- Improved code quality with ruff linting
  - Fixed SIM108 (use ternary operators)
  - Fixed N806 (variable naming conventions)
  - Fixed C409 (unnecessary tuple wrappers)
  - Fixed B007 (unused loop variables)
  - Fixed B905 (missing strict parameter in zip)
- **CI/CD fixes:**
  - Added SuiteSparse installation for scikit-umfpack support
  - Invalidated pip cache to force rebuild with correct dependencies
  - Added swig installation for macOS builds
  - All platforms (Ubuntu, macOS, Windows) now build successfully

### Infrastructure
- All tests passing (190 tests total, including 6 new dual value tests)
- CI/CD pipeline fully operational across all platforms
- Code formatted and linted with ruff

### Added (Progress Logging)
- **Progress logging for long-running solves**
  - `ProgressInfo` dataclass with solver state (iteration, phase, objective, elapsed time)
  - `ProgressCallback` type alias for type-safe callback functions
  - Optional `progress_callback` parameter in `solve()` and `solve_min_cost_flow()`
  - Configurable `progress_interval` to control callback frequency (default: 100)
  - Real-time tracking of Phase 1 (feasibility) and Phase 2 (optimality)
  - Objective estimate computation during solve
- **New test suite for progress logging** (`tests/unit/test_progress_logging.py`)
  - Callback invocation verification
  - ProgressInfo field validation
  - Interval control testing
  - Phase tracking and iteration monotonicity (6 new tests)
- **New example: `progress_logging_example.py`**
  - Real-time progress bar with percentage, iterations, objective, time
  - Phase-aware formatting
  - Demonstrates custom progress formatting and monitoring

### Changed (Progress Logging)
- Extended `solve()` API with progress tracking capabilities
- Enhanced `solve_min_cost_flow()` to support progress callbacks
- Backward compatible - all progress parameters are optional

### Infrastructure (Progress Logging)
- All tests passing (196 tests total, including 6 new progress logging tests)
- Zero performance impact when progress_callback is None
- Fully type-annotated for IDE support

### Added (Solver Configuration)
- **SolverOptions for configurable solver behavior**
  - `SolverOptions` dataclass for centralized solver configuration
  - `max_iterations` - Override default iteration limit
  - `tolerance` - Numerical precision control (default: 1e-6)
  - `pricing_strategy` - Choose "devex" (default) or "dantzig" pricing
  - `block_size` - Control pricing block size for arc selection
  - `ft_update_limit` - Forrest-Tomlin refactorization frequency (default: 64)
  - Comprehensive validation with helpful error messages
- **Dantzig pricing strategy implementation**
  - Most negative reduced cost selection
  - Alternative to Devex pricing for different problem characteristics
- **Forrest-Tomlin update limit enforcement**
  - Periodic basis rebuilds for numerical stability
  - Configurable via `ft_update_limit` parameter
  - Tracking of basis updates between rebuilds
- **New test suite for SolverOptions** (`tests/unit/test_solver_options.py`)
  - Default values validation
  - Custom configuration testing
  - Invalid parameter detection
  - Pricing strategy comparison
  - Block size and FT limit verification (15 new tests)
- **New example: `solver_options_example.py`**
  - Demonstrates all configuration options
  - Shows impact of different settings on performance
  - Transportation problem with 8 configuration scenarios
  - Comparison of Devex vs Dantzig pricing

### Changed (Solver Configuration)
- `NetworkSimplex` constructor accepts optional `SolverOptions` parameter
- `solve_min_cost_flow()` accepts `SolverOptions` for full configuration control
- Tolerance now configurable via `SolverOptions` (previously fixed to problem tolerance)
- `max_iterations` parameter overrides options value for convenience
- Backward compatible - all parameters are optional with sensible defaults

### Infrastructure (Solver Configuration)
- All tests passing (211 tests total, including 15 new SolverOptions tests)
- Full type annotations for SolverOptions
- Comprehensive documentation in docstrings

### Added (Utility Functions)
- **extract_path() for flow path discovery**
  - BFS-based algorithm to find flow-carrying paths between nodes
  - Returns `FlowPath` dataclass with nodes, arcs, flow value, and total cost
  - Handles edge cases (no path exists, source equals target, invalid nodes)
  - Useful for tracing shipment routes and understanding flow patterns
- **validate_flow() for solution verification**
  - Verifies flow conservation at all nodes
  - Checks capacity and lower bound constraints
  - Returns `ValidationResult` with detailed violation information
  - Works with both directed and undirected problems (uses expanded arcs)
  - Configurable tolerance for numerical precision
- **compute_bottleneck_arcs() for capacity analysis**
  - Identifies arcs at or near capacity (default: 95% utilization threshold)
  - Returns list of `BottleneckArc` objects with utilization metrics
  - Sorted by utilization (highest first) for priority identification
  - Includes slack, cost, and capacity information
  - Excludes infinite capacity arcs
  - Enables sensitivity analysis for capacity expansion planning
- **New dataclasses for utility results**
  - `FlowPath` - Path representation with nodes, arcs, flow, cost
  - `ValidationResult` - Validation report with errors and violations
  - `BottleneckArc` - Bottleneck information with utilization metrics
- **New test suite** (`tests/unit/test_utils.py`)
  - Path extraction tests (simple, multi-hop, branching, edge cases) - 6 tests
  - Flow validation tests (valid solutions, violations, tolerance) - 5 tests
  - Bottleneck detection tests (thresholds, sorting, integration) - 7 tests
  - Total: 18 new comprehensive tests
- **New example: `utils_example.py`**
  - Demonstrates all three utility functions
  - Transportation problem with validation, path extraction, bottleneck analysis
  - Shows sensitivity analysis using bottleneck information
  - Real-world interpretation with factories and warehouses

### Changed (Utility Functions)
- All utility functions exported from main `network_solver` module
- Type-safe with full annotations for IDE support
- Comprehensive docstrings with usage examples

### Infrastructure (Utility Functions)
- All tests passing (229 tests total, including 18 new utility tests)
- Full type annotations for all utilities
- Example demonstrates practical usage patterns

### Added (Documentation)
- **Comprehensive `docs/` directory with 4 detailed guides**
  - `docs/algorithm.md` - Network simplex algorithm explanation (~550 lines)
    - Problem formulation with mathematical notation
    - Algorithm structure with Phase 1 (feasibility) and Phase 2 (optimality)
    - Data structures: spanning trees, node potentials, reduced costs
    - Pricing strategies: Devex vs Dantzig with detailed comparisons
    - Basis management with Forrest-Tomlin updates
    - Complexity analysis (theoretical and practical)
    - Implementation details: cost perturbation, cycle detection, potential computation
    - Academic references to key papers
  - `docs/api.md` - Complete API reference (~500 lines)
    - All functions with parameters, return types, and examples
    - Problem definition classes (NetworkProblem, Node, Arc)
    - Solver configuration (SolverOptions with all parameters)
    - Results and analysis (FlowResult with flows, objective, duals)
    - Utility functions (extract_path, validate_flow, compute_bottleneck_arcs)
    - Progress tracking (ProgressInfo, ProgressCallback)
    - Exception hierarchy with all 7 exception types
    - I/O functions (load_problem, save_result)
    - Type annotations and IDE support
  - `docs/examples.md` - Annotated code examples (~430 lines)
    - Basic transportation problem with output
    - Supply chain with transshipment nodes
    - Maximum flow problem (conversion to min-cost flow)
    - Minimum cost circulation with lower bounds
    - Progress monitoring with real-time callbacks
    - Sensitivity analysis using dual values
    - Solver configuration comparisons
    - Flow validation and bottleneck analysis workflows
  - `docs/benchmarks.md` - Performance characteristics (~400 lines)
    - Complexity analysis (theoretical O() bounds)
    - Benchmark problems by size (small to very large)
    - Performance characteristics by problem type and structure
    - Empirical scaling formulas and comparison tables
    - Optimization tips for 5 different scenarios
    - Comparison with other solvers (LP, OR-Tools, etc.)
    - Hardware impact analysis
    - Profiling and benchmarking code examples
    - Future optimization opportunities

### Changed (Documentation)
- Total of ~1,880 lines of production-quality documentation added
- Cross-references between documentation files
- Code examples with expected outputs
- Performance tables and complexity analysis

### Refactored (Code Quality)
- **Improved code quality in simplex.py**
  - Extracted `_update_devex_weight()` helper method to eliminate ~20 lines of duplication
  - Extracted `_is_better_candidate()` helper method for cleaner merit comparison logic
  - Reduced `_find_entering_arc_devex()` from ~68 lines to ~48 lines
  - Added clearer section comments for forward/backward direction checking
  - No functional changes - all 229 tests continue to pass
  - Improved maintainability and readability of pricing logic

### Improved (Undirected Graph Handling)
- **Enhanced undirected graph support and documentation**
  - Improved error messages in `undirected_expansion()` with detailed explanations
    - Infinite capacity error now explains why finite capacity is required
    - Custom lower bound error explains automatic transformation to -capacity
    - Error messages include node names and specific values for debugging
  - Comprehensive docstring for `undirected_expansion()` explaining transformation
    - Documents how edges {u,v} become arcs (u,v) with lower=-C, upper=C
    - Explains flow interpretation: positive = tail→head, negative = head→tail
    - Lists all requirements: finite capacity, no custom lower bounds, symmetric costs
  - Added extensive documentation in `docs/api.md` (~155 lines)
    - "Working with Undirected Graphs" section with detailed explanation
    - Transformation mechanics with examples
    - Flow interpretation guide (positive/negative/zero flows)
    - Common errors section with fixes
    - When to use undirected vs directed comparison
    - Internal transformation details
  - Enhanced README.md with "Undirected Graphs" subsection
    - Clear requirements list (finite capacity, no custom lower bounds)
    - Transformation explanation with bullet points
    - Flow interpretation guide
    - Link to comprehensive API documentation
  - New comprehensive example: `undirected_graph_example.py` (~200 lines)
    - Campus network design scenario (4 buildings, fiber optic cables)
    - Shows graph structure, internal transformation, solution interpretation
    - Demonstrates flow conservation and optimal routing
    - Explains key insights and undirected vs directed comparison
  - New comprehensive test suite: `tests/unit/test_undirected_graphs.py` (13 tests)
    - Simple chain, bidirectional flow, triangle network
    - Expansion transformation verification
    - Error handling (infinite capacity, custom lower bounds)
    - Multiple edges, transshipment nodes
    - Negative flow interpretation, capacity constraints
    - Undirected vs directed equivalence
    - Error message quality verification
  - All tests passing (242 total, +13 new undirected tests)
- **Updated documentation files**
  - Added "Undirected Graphs" section to `docs/examples.md` (~90 lines)
    - Complete working example with campus network scenario
    - Flow interpretation guide (positive/negative values)
    - Common errors section with fixes
    - When to use undirected vs directed guidance
  - Added "Undirected Graphs" subsection to `docs/algorithm.md`
    - Explains transformation in mathematical notation
    - Shows how edges become arcs with lower=-C, upper=C
    - Links to API reference for details

### Improved (Docstrings)
- **Comprehensive docstring improvements across all main modules**
  - **data.py enhancements:**
    - Node: Added examples for supply/demand/transshipment nodes, attributes documentation
    - Arc: Added examples for basic/infinite capacity/lower bound arcs, raises documentation, link to undirected graphs
    - NetworkProblem: Added directed/undirected examples, methods documentation, "See Also" links to build_problem, solve_min_cost_flow, and docs/algorithm.md
    - FlowResult: Expanded status codes documentation, added complete working example showing usage, "See Also" links
    - SolverOptions: Added detailed parameter guidance (ranges, tradeoffs), 4 configuration examples (default, high-precision, fast, stable), link to docs/benchmarks.md
  - **solver.py enhancements:**
    - solve_min_cost_flow(): Added time/space complexity analysis (O(n²m) best, O(nm log n) average), expanded examples, detailed returns/raises documentation, "See Also" links to docs
    - load_problem(): Added complexity analysis, examples, "See Also" links
    - save_result(): Added complexity analysis, examples, "See Also" links
  - **simplex.py enhancements:**
    - NetworkSimplex: Added algorithm overview (Phase 1/Phase 2), implementation details (spanning tree, node potentials, pricing strategies, Forrest-Tomlin), attributes documentation, "See Also" links
  - All docstrings now follow consistent format with:
    - Detailed parameter descriptions
    - Return value documentation
    - Raises documentation where applicable
    - Time and space complexity analysis for key functions
    - Working code examples
    - "See Also" cross-references to related functions and documentation
  - Total: ~250 lines of enhanced documentation added to docstrings
  - All tests passing (242 tests)
  - Type checking and linting passing

### Improved (Logging)
- **Comprehensive logging at appropriate levels**
  - **INFO level** - Phase transitions and solver progress:
    - "Starting network simplex solver" with problem size and configuration
    - "Phase 1: Finding initial feasible solution"
    - "Phase 1 complete" with iteration counts
    - "Phase 2: Optimizing from feasible basis"
    - "Phase 2 complete" with iteration counts
  - **DEBUG level** - Individual pivot details:
    - Entering arc selection with reduced cost and direction
    - Leaving arc selection with theta and degeneracy flag
    - Degenerate pivot detection
    - FT update limit reached notifications
  - **WARNING level** - Numerical issues:
    - Devex weights going non-finite (NaN/Inf) with clamping
    - Devex weights exceeding maximum bounds
    - Forrest-Tomlin update failures requiring basis rebuild
    - Iteration limit reached before optimality
  - **ERROR level** - Solver failures:
    - Infeasible problems (no feasible solution exists)
    - Iteration limit reached before finding feasible solution
  - All log messages include structured extra data for programmatic parsing
- **Added --verbose flag to example scripts**
  - solve_example.py: `-v` for INFO, `-vv` for DEBUG
  - solve_dimacs_example.py: `-v` for INFO, `-vv` for DEBUG
  - undirected_graph_example.py: `-v` for INFO, `-vv` for DEBUG
  - Consistent logging format across all examples
  - Logs go to stderr, normal output to stdout
- **Documentation updates**
  - README.md: Added "Verbose Output" section explaining log levels
  - docs/examples.md: Added note about --verbose flag at top of document
- **Benefits:**
  - Easy debugging with `-vv` flag to see every pivot
  - Monitor solver progress with `-v` flag
  - Production use with default (WARNING+) for quiet operation
  - Structured logging with extra fields for monitoring/analytics
- All tests passing (242 tests)
- Type checking and linting passing

### Improved (Enhanced Structured Logging)
- **Comprehensive structured logging with performance metrics**
  - **Starting solver** log includes:
    - Problem size: `nodes`, `arcs`
    - Configuration: `max_iterations`, `pricing_strategy`, `tolerance`
    - Problem characteristics: `total_supply`
  - **Phase 1 start** log includes:
    - `elapsed_ms`: Time since solver start (0.0 at start)
  - **Phase 1 complete** log includes:
    - Phase metrics: `iterations`, `total_iterations`
    - Feasibility metrics: `artificial_flow` (sum of artificial arc flows)
    - Performance: `elapsed_ms` (time since solver start)
  - **Phase 2 start** log includes:
    - `remaining_iterations` (max_iterations - total_iterations)
  - **Phase 2 complete** log includes:
    - Phase metrics: `iterations`, `total_iterations`
    - Solution quality: `objective` (preliminary objective value)
    - Performance: `elapsed_ms` (time since solver start)
  - **Solver complete** log includes:
    - Final status: `status` ("optimal", "iteration_limit", "infeasible")
    - Solution: `objective` (rounded to 12 decimals)
    - Iterations: `iterations` (total across both phases)
    - Performance: `elapsed_ms` (total solve time)
    - Basis metrics: `tree_arcs` (non-artificial tree arcs), `nonzero_flows`
    - Numerical stability: `ft_rebuilds` (Forrest-Tomlin basis rebuilds)
- **All metrics available in structured `extra` dict for:**
  - JSON logging for monitoring systems
  - Performance profiling and analysis
  - Automated testing and validation
  - Real-time dashboards
- **Example JSON output:**
  ```json
  {"level": "INFO", "logger": "network_solver.simplex", "message": "Starting network simplex solver", "nodes": 3, "arcs": 3, "max_iterations": 100, "pricing_strategy": "devex", "total_supply": 10.0, "tolerance": 1e-06}
  {"level": "INFO", "logger": "network_solver.simplex", "message": "Phase 1 complete", "iterations": 2, "total_iterations": 2, "artificial_flow": 0, "elapsed_ms": 2.23}
  {"level": "INFO", "logger": "network_solver.simplex", "message": "Phase 2 complete", "iterations": 0, "total_iterations": 2, "objective": 15.0, "elapsed_ms": 3.64}
  {"level": "INFO", "logger": "network_solver.simplex", "message": "Solver complete", "status": "optimal", "objective": 15.0, "iterations": 2, "elapsed_ms": 4.04, "tree_arcs": 2, "nonzero_flows": 2, "ft_rebuilds": 0}
  ```
- All tests passing (243 tests)
- Type checking and linting passing

### Improved (Dual Variables Documentation and Examples)
- **Enhanced sensitivity analysis example** (`examples/sensitivity_analysis_example.py`)
  - Added comprehensive production planning use case
    - Two-factory scenario with different costs and capacities
    - Decision analysis: which factory to expand based on dual values
    - Marginal value calculations for capacity expansion
  - Added capacity constraint analysis section
    - Identify bottlenecks using flow vs capacity comparison
    - Utilization percentage calculations
    - Recommendations for capacity increases
  - Added key concepts summary section
    - Dual values interpretation (shadow prices)
    - Complementary slackness explanation
    - Sensitivity analysis formulas
    - Practical applications list (production planning, logistics, pricing, bottlenecks)
  - Improved code organization with helper functions
    - `print_section_header()` for major sections
    - `print_subsection()` for subsections
    - Better visual separation and readability
  - Enhanced example output (~150 lines total)
- **Comprehensive dual variables documentation** (`docs/examples.md`)
  - Expanded "Sensitivity Analysis" section from ~40 lines to ~210 lines
  - Added "What are Dual Values?" conceptual introduction
    - Clear explanation of negative vs positive duals
    - Interpretation guide for practitioners
  - New subsections with complete working examples:
    - **Basic Example**: Marginal cost prediction with verification
    - **Complementary Slackness**: Optimality condition verification
    - **Production Planning**: Two-factory capacity expansion decision
    - **Capacity Bottleneck Identification**: Finding binding constraints
  - Added "Key Concepts" reference table
    - Dual value, reduced cost, complementary slackness formulas
    - Clear mathematical notation and meaning
  - Added "When to Use Dual Values" section
    - 6 practical use cases with descriptions
    - What-if analysis, capacity planning, pricing decisions, etc.
  - Added example output excerpt from running script
  - Cross-references to algorithm guide and API documentation
- **Enhanced README.md dual values section**
  - Expanded from basic explanation to practical examples
  - Added code showing cost prediction without re-solving
  - Added complementary slackness verification example
  - Added "Use cases for dual values" bulleted list
    - What-if analysis, capacity planning, pricing, bottlenecks, optimality
  - Better cross-references to examples and documentation
- **Benefits:**
  - Users can now understand and apply dual values for real-world decisions
  - Clear examples for production planning and capacity expansion
  - Mathematical foundations with practical interpretation
  - Complete working code that can be adapted to user problems
- All tests passing (243 tests)
- Type checking and linting passing

### Added (Incremental Resolving Example and Documentation)
- **New comprehensive example**: `examples/incremental_resolving_example.py` (~370 lines)
  - **Scenario 1: Capacity Expansion Analysis**
    - Demonstrates incremental capacity increases
    - Shows diminishing returns on expansion
    - Tracks objective improvement at each step
  - **Scenario 2: Cost Updates**
    - Fuel price increase impact analysis
    - Flow pattern comparison before/after
    - Percentage cost increase calculation
  - **Scenario 3: Demand Fluctuations**
    - Weekly demand variation handling
    - Multi-period re-solving
    - Cost progression tracking
  - **Scenario 4: Network Topology Changes**
    - Evaluate adding new direct routes
    - Calculate savings from topology changes
    - ROI analysis for infrastructure investments
  - **Scenario 5: Iterative Optimization**
    - Bottleneck identification → expand → re-solve loop
    - Demonstrates systematic network improvement
    - Shows when to stop iterating (diminishing returns)
  - Formatted output with clear sections and tables
  - Total solve time < 100ms for all 5 scenarios
- **Comprehensive documentation** in `docs/examples.md`
  - New "Incremental Resolving" section (~260 lines)
  - "Why Incremental Resolving?" explanation
    - Scenario analysis, cost sensitivity, demand forecasting
    - Network design, iterative optimization
  - 5 complete working code examples (one per scenario)
  - Best practices section:
    - Start simple, track metrics, validate incrementally
    - Use dual values for prediction
    - Consider tolerance in small changes
  - Performance notes:
    - Fast for networks <10,000 arcs (<100ms typical)
    - No warm-start support but re-solving is efficient
    - Can parallelize independent variants
  - Example output excerpts showing real results
- **README.md updates**
  - Added `incremental_resolving_example.py` to CLI examples list
  - Listed with clear description: "Scenario analysis and what-if modeling"
- **Use cases enabled**:
  - Capacity planning: "What if we expand this route?"
  - Cost sensitivity: "How do price changes affect solutions?"
  - Demand adaptation: Handle varying demand over time
  - Network design: Evaluate topology modifications
  - Iterative improvement: Systematically reduce costs
- **Benefits:**
  - Users can perform sophisticated what-if analysis
  - Clear patterns for common re-solving scenarios
  - Practical examples adaptable to real problems
  - Demonstrates efficient workflow for scenario analysis
- All tests passing (243 tests)
- Type checking and linting passing

### Added (Performance Profiling Example and Documentation)
- **New comprehensive example**: `examples/performance_profiling_example.py` (~440 lines)
  - **Scaling Analysis**
    - Profile solver performance across problem sizes (3×3 to 20×20 grids)
    - Measure solve time, iteration count, and throughput (iterations/sec)
    - Demonstrates quadratic time growth with problem size
    - Formatted table output for easy comparison
  - **Pricing Strategy Comparison**
    - Compare Devex vs Dantzig pricing on multiple problem types
    - Grid networks and bipartite (assignment) networks
    - Shows iteration counts, solve times, and speedup factors
    - Helps users choose optimal pricing strategy
  - **Solver Options Impact**
    - Test different `ft_update_limit` values (16, 32, 64, 128)
    - Test different `block_size` values (50, 100, 200)
    - Shows how configuration affects performance
    - Guidance for tuning solver parameters
  - **Problem Structure Analysis**
    - Compare sparse (grid), medium (hybrid), and dense (bipartite) networks
    - Calculate and display network density
    - Shows how structure affects iteration count and solve time
  - **Performance Summary**
    - Benchmark standard problem (10×10 grid)
    - Performance expectations by problem size
    - Optimization tips (5 recommendations)
    - When to profile (4 scenarios)
  - **Structured Logging for Profiling**
    - Demonstrates capturing metrics via structured logs
    - Shows integration with INFO-level logging
  - Helper functions for problem generation:
    - `generate_grid_network()`: 4-connected grid with source/sink
    - `generate_bipartite_network()`: Complete bipartite graph
    - `profile_problem()`: Unified profiling with timing
  - Suppresses solver logging during profiling for clean output
  - Total profiling time < 15s for all scenarios
- **Comprehensive documentation** in `docs/examples.md`
  - New "Performance Profiling" section (~260 lines)
  - "Why Profile Performance?" explanation (6 use cases)
    - Understand scaling, compare strategies, tune configuration
    - Identify bottlenecks, regression testing, capacity planning
  - Complete working code examples:
    - **Basic Profiling**: Simple time/iteration measurement
    - **Scaling Analysis**: Profile multiple problem sizes with table output
    - **Comparing Pricing Strategies**: Devex vs Dantzig with speedup calculation
    - **Configuration Tuning**: Test different solver options
    - **Problem Structure Analysis**: Compare sparse/dense/medium networks
    - **Structured Logging**: Capture metrics programmatically
  - Best practices (6 guidelines)
    - Profile representative problems, run multiple iterations
    - Isolate variables, track over time, document baselines
  - Performance expectations table:
    - Small (<100 nodes): <10ms
    - Medium (100-1000): 10-100ms
    - Large (1000-10000): 100ms-2s
    - Very large (>10000): Several seconds
  - Example output excerpts showing real profiling results
- **README.md updates**
  - Added `performance_profiling_example.py` to CLI examples list
  - Listed with clear description: "Performance analysis and benchmarking"
- **Use cases enabled**:
  - Performance benchmarking for different problem types
  - Solver configuration optimization
  - Regression testing across code versions
  - Capacity planning for production systems
  - Understanding scaling characteristics
  - Identifying performance bottlenecks
- **Benefits:**
  - Users can optimize solver configuration for their workloads
  - Clear guidance on what to expect performance-wise
  - Systematic approach to performance analysis
  - Helps identify when performance is abnormal
  - Enables data-driven configuration decisions
- All tests passing (243 tests)
- Type checking and linting passing

### Added (Jupyter Notebook Tutorial)
- **Interactive Jupyter notebook tutorial**: `tutorials/network_flow_tutorial.ipynb`
  - **Installation and Setup** - Quick start with package installation
  - **First Network Flow Problem** - Transportation problem walkthrough
    - Define nodes (factories, warehouses) with supply/demand
    - Define arcs (shipping routes) with costs and capacities
    - Build and solve the problem
  - **Solving and Interpreting Results**
    - Access optimal flows, objective value, and solver status
    - Validate flow conservation at nodes
    - Interpret solution quality and iteration counts
  - **Dual Values and Sensitivity Analysis**
    - Understand shadow prices and node potentials
    - Predict cost impact of supply/demand changes
    - Verify complementary slackness conditions
  - **Maximum Flow Problem**
    - Convert max-flow to min-cost flow formulation
    - Add super-source and super-sink nodes
    - Extract maximum throughput from results
  - **Solver Configuration**
    - Customize iteration limits, tolerance, pricing strategy
    - Compare Devex vs Dantzig pricing
    - Tune performance parameters
  - **Incremental Resolving**
    - Modify and re-solve efficiently
    - Capacity expansion analysis
    - Cost update scenarios
  - **Bottleneck Analysis**
    - Identify capacity-constrained arcs
    - Calculate utilization percentages
    - Prioritize infrastructure investments
  - **Visualization** (Optional)
    - Plot network graphs with matplotlib and networkx
    - Visualize flow patterns and bottlenecks
    - Generate publication-quality diagrams
  - **Summary and Next Steps**
    - Key takeaways and concepts learned
    - Links to comprehensive documentation
    - Suggested learning paths
  - Total: 22 cells (12 markdown, 10 code)
  - All code cells are executable and self-contained
  - Generated programmatically via `tutorials/create_tutorial.py`
- **Documentation updates**
  - Added link to notebook in README.md Documentation section
  - Added prominent callout in `docs/examples.md` for new users
  - Positioned notebook as starting point for interactive learning
- **Use cases enabled**:
  - Interactive learning environment for beginners
  - Hands-on experimentation with immediate feedback
  - Quick reference for common workflows
  - Teaching material for courses and workshops
  - Rapid prototyping and exploration
- **Benefits:**
  - Lower barrier to entry for new users
  - Visual and interactive learning experience
  - All examples executable without setup
  - Self-contained tutorial covering all major features
  - Easy to customize and extend for specific use cases
- Notebook structure generated by Python script for maintainability
- All 243 tests continue to pass

### Planned
- PyPI publication
- Additional optimization algorithms
- C++/Cython performance extensions
