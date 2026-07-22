.PHONY: fix lint format format-check test test-unit test-integration ci setup

fix:
	uv run --locked ruff check --fix
	uv run --locked ruff format

lint:
	uv run --locked ruff check src/ tests/

format:
	uv run --locked ruff format src/ tests/

format-check:
	uv run --locked ruff format --check src/ tests/

test:
	uv run --locked pytest

test-unit:
	uv run --locked pytest -m "not integration"

test-integration:
	uv run --locked pytest -m integration --no-cov

ci: lint test
	@echo ""
	@echo "All CI checks passed!"

setup:
	uv sync --locked --group dev
	pre-commit install
