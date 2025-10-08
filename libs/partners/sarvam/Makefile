.PHONY: all format lint test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	poetry run pytest $(TEST_FILE)

tests:
	poetry run pytest $(TEST_FILE)

integration_tests:
	poetry run pytest tests/integration_tests

extended_tests:
	poetry run pytest --only-extended tests/unit_tests

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/sarvam --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_sarvam
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	poetry run ruff check .
	poetry run ruff format . --check
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_sarvam -name '*.py')
	poetry run python -m scripts.check_imports $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports         - check imports'
	@echo 'format                - run code formatters'
	@echo 'lint                  - run linters'
	@echo 'test                  - run unit tests'
	@echo 'tests                 - run unit tests'
	@echo 'integration_tests     - run integration tests'
	@echo 'extended_tests        - run extended tests'
	@echo 'spell_check           - run codespell on the project'
	@echo 'spell_fix             - run codespell on the project and fix issues'
