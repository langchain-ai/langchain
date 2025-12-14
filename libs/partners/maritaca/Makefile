.PHONY: all format lint test tests test_watch integration_tests help

all: help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  format           Format code with ruff"
	@echo "  lint             Run linting checks"
	@echo "  test             Run unit tests"
	@echo "  test_watch       Run tests in watch mode"
	@echo "  integration_tests Run integration tests"
	@echo "  help             Show this help message"

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .
	uv run --group typing mypy .

test:
	uv run --group test pytest tests/unit_tests

tests:
	uv run --group test pytest tests/unit_tests

test_watch:
	uv run --group test pytest-watcher tests/unit_tests

integration_tests:
	uv run --group test_integration pytest tests/integration_tests
