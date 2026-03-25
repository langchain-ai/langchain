.PHONY: all format lint test test_integration help

all: help

help:
	@echo "Available targets:"
	@echo "  format         -- run ruff format"
	@echo "  lint           -- run ruff lint"
	@echo "  test           -- run unit tests"
	@echo "  test_integration -- run integration tests (requires cortex-ai-memory)"

format:
	uv run ruff format langchain_cortex tests

lint:
	uv run ruff check langchain_cortex tests

test:
	uv run pytest tests/unit_tests/ -v

test_integration:
	uv run pytest tests/integration_tests/ -v
