.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_tests: TEST_FILE = tests/integration_tests/

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Run unit tests and generate a coverage report.
coverage:
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

test tests:
	uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

integration_tests:
	uv run --group test --group test_integration pytest $(TEST_FILE)

test_watch:
	uv run --group test ptw --disable-socket --allow-unix-socket --snapshot-update --now . -- -vv tests/unit_tests

check_imports: $(shell find langchain_community -name '*.py')
	uv run --group test python ./scripts/check_imports.py $^

extended_tests:
	uv run --no-sync --group test pytest --disable-socket --allow-unix-socket --only-extended tests/unit_tests


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/community --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_community
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/check_pydantic.sh .
	./scripts/lint_imports.sh .
	./scripts/check_pickle.sh .
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --group typing --group lint mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run --group typing --group lint codespell --toml pyproject.toml

spell_fix:
	uv run --group typing --group lint codespell --toml pyproject.toml -w

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
