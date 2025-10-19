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

### Planned
- PyPI publication
- Additional optimization algorithms
- C++/Cython performance extensions
