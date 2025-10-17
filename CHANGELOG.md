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

### Planned
- Custom exception hierarchy for better error handling
- GitHub Actions CI/CD workflow
- Enhanced API with dual variables and sensitivity analysis
- Performance benchmarking suite
- PyPI publication
