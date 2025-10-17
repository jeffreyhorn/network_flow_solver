# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Network Flow Solver project.

## Workflows

### 1. CI Workflow (`ci.yml`)

**Trigger:** Push to `main`/`develop` branches, pull requests, or manual dispatch

**Jobs:**

- **Lint** - Code style checking with ruff
  - Checks code formatting
  - Lints for common issues
  - Fast fail for style violations

- **Type Check** - Static type analysis with mypy
  - Ensures type hints are correct
  - Catches type-related bugs early

- **Test** - Multi-platform testing matrix
  - **Platforms:** Ubuntu, macOS, Windows
  - **Python versions:** 3.12 (expandable to 3.13)
  - Runs unit tests, integration tests, property-based tests
  - Tests with and without UMFPACK (Windows doesn't support it)

- **Coverage** - Code coverage analysis
  - Generates coverage report with pytest-cov
  - Uploads to Codecov (requires `CODECOV_TOKEN` secret)
  - Creates HTML coverage report artifact
  - Target: ≥90% coverage

- **Examples** - Verify example scripts work
  - Runs all example scripts
  - Ensures examples stay synchronized with API
  - Runs installation verification script

- **Build** - Package building and validation
  - Builds source distribution and wheel
  - Validates package metadata with twine
  - Archives build artifacts

- **All Checks Pass** - Status check for branch protection
  - Aggregates all job results
  - Required for merge to main

**Caching:**
- Pip packages cached by platform and Python version
- Speeds up workflow execution significantly

### 2. Release Workflow (`release.yml`)

**Trigger:** Published GitHub release or manual dispatch

**Jobs:**

- **Build and Publish** - PyPI publication
  - Verifies version matches Git tag
  - Builds distribution packages
  - Publishes to PyPI using trusted publishing
  - Attaches artifacts to GitHub release
  - Manual dispatch publishes to Test PyPI

**Required Secrets:**
- `PYPI_API_TOKEN` - For PyPI publication (release events)
- `TEST_PYPI_API_TOKEN` - For Test PyPI (manual dispatch)

**Publishing Steps:**
1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Commit changes: `git commit -m "Release v0.x.0"`
4. Tag release: `git tag v0.x.0`
5. Push: `git push && git push --tags`
6. Create GitHub release from tag
7. Workflow automatically publishes to PyPI

### 3. Dependency Review (`dependency-review.yml`)

**Trigger:** Pull requests to `main`/`develop`

**Purpose:** Security scanning for new dependencies

**Features:**
- Scans for vulnerable dependencies
- Fails on high-severity vulnerabilities
- Comments on PR with security findings
- Integrates with GitHub's dependency graph

## Setup Instructions

### Initial Setup

1. **Enable GitHub Actions** in repository settings
2. **Configure branch protection** for `main`:
   - Require "All Checks Pass" status check
   - Require pull request reviews
   - Require status checks to pass before merging

### Optional: Codecov Integration

1. Sign up at [codecov.io](https://codecov.io)
2. Link your repository
3. Add `CODECOV_TOKEN` to repository secrets
4. Coverage badge will appear in README

### Optional: PyPI Publishing

For automated PyPI releases:

1. **Create PyPI account** at [pypi.org](https://pypi.org)
2. **Generate API token:**
   - Go to Account Settings → API tokens
   - Create token with scope limited to this project
3. **Add secret to GitHub:**
   - Settings → Secrets → Actions
   - Add `PYPI_API_TOKEN` with the token value

For Test PyPI (optional):
1. Create account at [test.pypi.org](https://test.pypi.org)
2. Generate API token
3. Add `TEST_PYPI_API_TOKEN` secret

### Running Workflows Locally

While you can't run GitHub Actions exactly as on GitHub, you can approximate:

```bash
# Lint
make lint

# Type check
make typecheck

# Unit tests
make unit

# Integration tests
make integration

# All tests
make test

# Coverage
make coverage

# Build package
python -m build

# Check package
twine check dist/*
```

For more faithful local testing, use [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or follow instructions at https://github.com/nektos/act

# Run CI workflow
act push

# Run specific job
act -j test

# Run with specific event
act pull_request
```

## Workflow Badges

Add to README.md:

```markdown
[![CI](https://github.com/jeffreyhorn/network_flow_solver/workflows/CI/badge.svg)](https://github.com/jeffreyhorn/network_flow_solver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jeffreyhorn/network_flow_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyhorn/network_flow_solver)
```

## Troubleshooting

### Tests fail on Windows

- UMFPACK is not available on Windows (expected)
- The workflow has `continue-on-error` for UMFPACK installation
- Tests should still pass using NumPy fallback

### Coverage upload fails

- Codecov upload has `continue-on-error: true`
- Workflow will still pass if Codecov is unavailable
- Check if `CODECOV_TOKEN` secret is set

### Build artifacts not appearing

- Artifacts are retained for 30 days
- Download from Actions tab → Workflow run → Artifacts section

### PyPI publish fails

- Verify version number doesn't already exist on PyPI
- Check `PYPI_API_TOKEN` secret is set
- Ensure token has correct permissions
- Try Test PyPI first with manual dispatch

## Maintenance

### Updating Dependencies

When updating dependencies:
1. Update `pyproject.toml`
2. Update `requirements.txt` (for documentation)
3. Test locally: `pip install -e ".[dev,umfpack]"`
4. Push changes - CI will test with new dependencies
5. Dependency review action will scan for vulnerabilities

### Adding New Tests

1. Add test files to `tests/unit/` or `tests/integration/`
2. Follow naming convention: `test_*.py`
3. Use `@pytest.mark.slow` for expensive tests
4. CI will automatically discover and run new tests

### Extending Test Matrix

To test additional Python versions, edit `ci.yml`:

```yaml
matrix:
  python-version: ["3.12", "3.13"]
```

## Best Practices

1. **Run tests locally before pushing** - Use `make test`
2. **Keep CI fast** - Use caching, mark slow tests
3. **Fix broken main immediately** - Revert if needed
4. **Monitor coverage** - Aim for ≥90%
5. **Review dependency updates** - Check security advisories
6. **Tag releases properly** - Use semantic versioning

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
