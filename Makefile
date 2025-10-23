# Configuration variables
PYTEST?=.venv/bin/pytest
MYPY?=mypy
PYTHON?=python
RUFF?=ruff

.PHONY: help lint format format-check typecheck unit integration test coverage clean install dev-install build all check

# Default target - show help
help:
	@echo "Available targets:"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run ruff linting checks"
	@echo "  make format        - Auto-format code with ruff"
	@echo "  make format-check  - Check if code needs formatting (CI-friendly)"
	@echo "  make typecheck     - Run mypy type checking"
	@echo "  make check         - Run all checks (lint + format-check + typecheck)"
	@echo "  make all           - Run all checks + tests"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make unit          - Run unit tests only"
	@echo "  make integration   - Run integration tests only"
	@echo "  make coverage      - Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install package in development mode"
	@echo "  make dev-install   - Install package with dev dependencies"
	@echo "  make build         - Build distribution packages"
	@echo "  make clean         - Remove build artifacts and caches"

# Linting
lint:
	$(RUFF) check .

# Formatting
format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

# Type checking
typecheck:
	$(MYPY) src/network_solver/ --strict

# Combined check target for CI/CD
check: lint format-check typecheck
	@echo "✓ All checks passed!"

# Run all checks and tests
all: check test
	@echo "✓ All checks and tests passed!"

# Testing
unit:
	$(PYTEST) tests/unit -q

integration:
	$(PYTEST) tests/integration -q

test:
	$(PYTEST) -q

coverage:
	$(PYTEST) --cov=src/network_solver --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Installation
install:
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"

# Build
build:
	$(PYTHON) -m build

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
