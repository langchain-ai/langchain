.PHONY: test lint format

test:
	uv run pytest tests/unit_tests

lint:
	uv run ruff check .
	uv run mypy .

format:
	uv run ruff format .
	uv run ruff check --fix .
