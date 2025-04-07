.PHONY: test install-dev install install-requirements clean format lint typecheck compile-requirements sync-pyproject-deps

test:
	pytest

install-dev:
	pip install -e .[dev,postgresql,mongodb]

install:
	pip install .

install-requirements:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt

compile-requirements:
	pip-compile requirements.in
	pip-compile requirements-dev.in
	pip-compile requirements-test.in

format:
	black src tests

lint:
	ruff check src tests

typecheck:
	mypy src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache dist build

sync-pyproject-deps:
	@echo "Updating pyproject.toml dependencies from requirements.in..."
	@python3 scripts/sync_pyproject_deps.py requirements.in

