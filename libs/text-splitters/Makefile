.PHONY: all format lint test tests test_watch integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

test tests:
	uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest tests/integration_tests/

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv -x tests/unit_tests

test_profile:
	uv run --group test pytest -vv tests/unit_tests/ --profile-svg

check_imports: $(shell find langchain_text_splitters -name '*.py')
	uv run --group test python ./scripts/check_imports.py $^

extended_tests:
	uv run --group test pytest --disable-socket --allow-unix-socket --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/core --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_text_splitters
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

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
	@echo 'test_watch                   - run unit tests in watch mode'
