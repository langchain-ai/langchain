.PHONY: all format lint test help

# Default target executed when no arguments are given to make.
all: help

start:
	poetry run uvicorn ____project_name_identifier.server:app --reload

# Define a variable for the test file path.
TEST_FILE ?= tests/

test:
	poetry run pytest $(TEST_FILE)

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	poetry run mypy $(PYTHON_FILES)
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

format format_diff:
	poetry run black $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'make start                        - start server'
	@echo 'make format                       - run code formatters'
	@echo 'make lint                         - run linters'
	@echo 'make test                         - run unit tests'
