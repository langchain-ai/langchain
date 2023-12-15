.PHONY: all format lint test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	poetry run pytest $(TEST_FILE)

tests:
	poetry run pytest $(TEST_FILE)

check_imports: $(shell find langchain_nvidia_aiplay -name '*.py')
	poetry run python ./scripts/check_imports.py $^

integration_tests:
	poetry run pytest tests/integration_tests


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_nvidia_aiplay
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/check_pydantic.sh .
	./scripts/lint_imports.sh
	poetry run ruff .
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy $(PYTHON_FILES)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
