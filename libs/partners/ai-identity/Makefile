.PHONY: test lint format check_imports

test:
	uv run pytest tests/unit_tests

integration_test:
	uv run pytest tests/integration_tests

lint:
	uv run ruff check .
	uv run ruff format . --check

format:
	uv run ruff check . --fix
	uv run ruff format .

check_imports:
	uv run python scripts/check_imports.py
