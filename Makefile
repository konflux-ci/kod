.PHONY: fix lint format test ci setup

fix:
	uv run --locked ruff check --fix
	uv run --locked ruff format

lint:
	uv run --locked ruff check src/ tests/

format:
	uv run --locked ruff format src/ tests/

test:
	uv run --locked pytest

ci: lint test
	@echo ""
	@echo "All CI checks passed!"

setup:
	uv sync --locked --group dev
	pre-commit install
