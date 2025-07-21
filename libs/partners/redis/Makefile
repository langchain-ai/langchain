.PHONY: all format lint test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Run unit tests.
test:
	python -m pytest tests/unit_tests/ --cov langchain_redis --cov-report term-missing --cov-report html --cov-report xml --cov-fail-under 85

integration_tests:
	python -m pytest tests/integration_tests/

# Run all tests
tests:
	python -m pytest tests/ --cov langchain_redis --cov-report term-missing --cov-report html --cov-report xml --cov-fail-under 70

extended_tests:
	python -m pytest tests/ --cov langchain_redis --cov-report term-missing --cov-report html --cov-report xml --cov-fail-under 70

# Format code
format:
	python -m ruff format .

# Lint code
lint:
	python -m ruff check .

help:
	@echo 'Makefile for langchain-redis                                            '
	@echo '                                                                             '
	@echo 'Usage:                                                                       '
	@echo '   make test                    run unit tests                              '
	@echo '   make integration_tests       run integration tests                       '
	@echo '   make tests                   run all tests                              '
	@echo '   make lint                    run ruff linting                           '
	@echo '   make format                  run code formatting                        '
	@echo '                                                                             '