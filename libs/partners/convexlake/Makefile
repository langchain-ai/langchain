.PHONY: install lint format test test_unit test_integration build publish

install:
	uv sync --dev

lint:
	uv run ruff check .
	uv run mypy langchain_convexlake

format:
	uv run ruff check --fix .
	uv run ruff format .

test: test_unit

test_unit:
	uv run pytest tests/unit_tests -v

test_integration:
	uv run pytest tests/integration_tests -v -m integration

build:
	uv build

publish: build
	uv publish
