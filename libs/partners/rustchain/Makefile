.PHONY: all format lint test tests help

.DEFAULT_GOAL := help

test tests:
	uv run --group test pytest tests/unit_tests

format:
	uv run --group lint ruff format .

lint:
	uv run --group lint ruff check .

help:
	@echo "targets: test, format, lint"