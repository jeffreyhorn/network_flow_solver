# Developer Guidelines

This document provides guidelines for contributors working on the Network Flow Solver project.

## Project Structure

```
network_flow_solver/
├── src/network_solver/          # Production code
│   ├── __init__.py              # Public API exports
│   ├── data.py                  # Data structures (Node, Arc, NetworkProblem)
│   ├── io.py                    # JSON I/O utilities
│   ├── solver.py                # High-level solver facade
│   ├── simplex.py               # Network simplex algorithm implementation
│   ├── basis.py                 # Tree basis management
│   ├── basis_lu.py              # LU factorization wrapper
│   └── core/
│       ├── __init__.py
│       └── forrest_tomlin.py    # Forrest-Tomlin update engine
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests for individual modules
│   │   ├── test_data_validation.py
│   │   ├── test_simplex.py
│   │   ├── test_basis_*.py
│   │   └── test_io.py
│   ├── integration/             # End-to-end integration tests
│   │   ├── test_solver_end_to_end.py
│   │   └── test_solver_performance.py
│   ├── test_property_min_cost_flow.py  # Property-based tests
│   └── test_large_directed.py          # Large problem tests
├── examples/                    # Runnable examples
│   ├── solve_example.py
│   ├── solve_dimacs_example.py
│   └── *.json                   # Sample problem instances
├── pyproject.toml               # Package configuration
├── Makefile                     # Common development tasks
└── README.md                    # Project documentation
```

## Module Responsibilities

- **`data.py`**: Immutable data structures with validation (Node, Arc, NetworkProblem, FlowResult)
- **`io.py`**: JSON serialization/deserialization for problems and results
- **`solver.py`**: Public API facade (load_problem, solve_min_cost_flow, save_result)
- **`simplex.py`**: Core network simplex algorithm with two-phase method, Devex pricing, and degeneracy handling
- **`basis.py`**: Spanning tree basis management, potential computation, cycle detection
- **`basis_lu.py`**: Sparse LU factorization wrapper (SciPy/dense NumPy fallback)
- **`core/forrest_tomlin.py`**: Incremental basis update mechanism

## Installation & Setup

```bash
# Clone repository
git clone <repo-url>
cd network_flow_solver

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with all tools
pip install -e ".[dev,umfpack]"

# Verify installation
python verify_install.py
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run only unit tests
make unit

# Run integration tests
make integration

# Run with coverage report
make coverage

# Skip slow tests during development
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
make fmt

# Run linter
make lint

# Type check
make typecheck

# All quality checks before commit
make lint && make typecheck && make test
```

### Adding Features

1. **Write tests first** (TDD approach recommended)
   - Add unit tests in `tests/unit/test_<module>.py`
   - Add integration tests in `tests/integration/` for end-to-end flows
   - Use `@pytest.mark.slow` for expensive tests

2. **Implement the feature**
   - Follow existing code structure and patterns
   - Add type hints for all public APIs
   - Update docstrings with examples where helpful

3. **Verify quality**
   - Run `make lint` (no ruff or mypy errors)
   - Run `make test` (all tests pass)
   - Check coverage: `make coverage` (aim for ≥90%)

4. **Update documentation**
   - Update README.md if user-facing changes
   - Add entry to CHANGELOG.md
   - Update examples if API changed

5. **Submit pull request**
   - GitHub Actions will automatically run CI checks
   - All checks must pass before merge
   - CI runs: lint, typecheck, tests (Linux/macOS/Windows), coverage, examples, build

## Coding Conventions

### Style Guidelines

- **Python version**: 3.12+
- **Line length**: 100 characters (configured in pyproject.toml)
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Modules/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Type Hints

All public APIs must have complete type hints:

```python
def solve_min_cost_flow(
    problem: NetworkProblem, max_iterations: Optional[int] = None
) -> FlowResult:
    """Solve minimum-cost flow problem."""
    ...
```

### Dataclasses

Use frozen dataclasses for immutable data structures:

```python
@dataclass(frozen=True)
class Node:
    id: str
    supply: float = 0.0
```

### Error Handling

- Use specific exceptions for different error cases
- Provide helpful error messages with context
- Validate inputs early and fail fast

```python
if abs(total_supply) > self.tolerance:
    raise ValueError(
        f"Problem is unbalanced: total supply {total_supply:.6f} "
        f"exceeds tolerance {self.tolerance}."
    )
```

## Testing Guidelines

### Test Organization

- **Unit tests** (`tests/unit/`): Test individual functions/classes in isolation
  - Mock external dependencies when needed
  - Test edge cases, error conditions, and boundary values
  - Fast execution (< 1 second per test)

- **Integration tests** (`tests/integration/`): Test complete workflows
  - Load real problem files from `examples/`
  - Verify end-to-end correctness
  - May be marked `@pytest.mark.slow`

- **Property-based tests** (`test_property_min_cost_flow.py`): Use Hypothesis
  - Generate random valid problems
  - Check invariants (flow conservation, optimality conditions)

### Test Naming

```python
def test_<what>_<scenario>_<expected_outcome>():
    """Test that <what> <expected_outcome> when <scenario>."""
```

Examples:
- `test_pivot_clamps_flow_to_bounds()`
- `test_pricing_returns_none_without_candidates()`
- `test_undirected_expansion_requires_finite_capacity()`

### Test Structure

```python
def test_feature():
    # Arrange: Set up test data
    problem = build_problem(...)
    
    # Act: Execute the code under test
    result = solve_min_cost_flow(problem)
    
    # Assert: Verify expected behavior
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected_value)
```

## Algorithm Implementation Notes

### Numerical Stability

- Use tolerance checks for floating-point comparisons
- Clamp values to feasible bounds after pivots
- Reset Devex weights after basis refactorization
- Implement multiple fallback strategies (FT → sparse LU → dense)

### Performance Considerations

- Forrest-Tomlin updates amortize factorization cost
- Block pricing reduces pricing overhead
- Sparse matrix operations when SciPy/UMFPACK available
- Early termination in Phase 1 when artificial arcs reach zero

### Debugging Tips

- Enable debug logging: `logging.getLogger("network_solver.simplex").setLevel(logging.DEBUG)`
- Check `ft_rebuilds` counter for numerical issues
- Verify basis tree connectivity in `basis.rebuild()`
- Use `verify_install.py` to test installation

## Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code restructuring without behavior change
- `perf:` - Performance improvements
- `chore:` - Build/tooling changes

Examples:
```
feat: add dual variables to solution result
fix: handle degenerate pivots in ratio test
docs: add installation troubleshooting guide
test: add property-based tests for pricing
```

## Release Process

1. Update version in `pyproject.toml` and `src/network_solver/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite: `make test`
4. Verify type checking: `make typecheck`
5. Tag release: `git tag v0.x.0`
6. Build package: `python -m build`
7. Upload to PyPI: `python -m twine upload dist/*`

## Getting Help

- Review existing tests for usage examples
- Check README.md for algorithm overview
- Read inline docstrings and comments
- Open an issue for questions or bugs

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all quality checks pass
5. Submit a pull request

Thank you for contributing to Network Flow Solver!
