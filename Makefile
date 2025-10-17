PYTEST?=.venv/bin/pytest
MYPY?=mypy

.PHONY: lint typecheck unit integration test coverage

lint:
	ruff check

typecheck:
	$(MYPY) src/network_solver/

unit:
	$(PYTEST) tests/unit -q

integration:
	$(PYTEST) tests/integration -q

test:
	$(PYTEST) -q

coverage:
	$(PYTEST) --cov=src/network_solver --cov-report=term-missing
