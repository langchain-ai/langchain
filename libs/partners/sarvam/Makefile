.PHONY: all lint format test tests integration_tests help

all: help

######################
# LINTING AND FORMATTING
######################

lint format: PYTHON_FILES=.
lint:
	poetry run ruff check $(PYTHON_FILES)
	poetry run mypy $(PYTHON_FILES)

format:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)

######################
# TESTS
######################

test tests:
	poetry run pytest tests/unit_tests

integration_tests:
	poetry run pytest tests/integration_tests

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format         - run code formatters'
	@echo 'lint           - run linters'
	@echo 'test           - run unit tests'
	@echo 'integration_tests - run integration tests'
