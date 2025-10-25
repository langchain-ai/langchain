# Makefile for langchain-serpex

.PHONY: all format lint test tests integration_tests help

all: help

format:
	poetry run ruff format .
	poetry run ruff check --select I --fix .

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest tests/

tests: test

integration_tests:
	poetry run pytest tests/ -m "not skip"

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'integration_tests           - run integration tests'
