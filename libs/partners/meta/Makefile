.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/meta --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_meta
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run --all-groups codespell --toml pyproject.toml

spell_fix:
	uv run --all-groups codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_meta -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
