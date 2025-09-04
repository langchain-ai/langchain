.PHONY: all format lint test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest -n auto $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)

make benchmark:
	uv run --group test pytest ./tests -m benchmark

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache

lint:
	uv run --group lint ruff check $(PYTHON_FILES)
	uv run --group lint ruff format $(PYTHON_FILES) --diff
	mkdir -p $(MYPY_CACHE) && uv run --group typing mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_diff:
	$(eval PYTHON_FILES := $(shell git diff --relative=libs/partners/llamastack --name-only --diff-filter=d master | grep -E '\.(py|ipynb)$$' | sed 's|^libs/partners/llamastack/||'))
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --group typing mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_package:
	uv run --group lint ruff check langchain_llamastack
	uv run --group lint ruff format langchain_llamastack --diff
	mkdir -p $(MYPY_CACHE) && uv run --group typing mypy langchain_llamastack --cache-dir $(MYPY_CACHE)

lint_tests:
	uv run --group lint ruff check tests
	uv run --group lint ruff format tests --diff
	mkdir -p .mypy_cache_test && uv run --group typing mypy tests --cache-dir .mypy_cache_test

format:
	uv run --group lint ruff format $(PYTHON_FILES)
	uv run --group lint ruff check $(PYTHON_FILES) --fix --unsafe-fixes

format_diff:
	$(eval PYTHON_FILES := $(shell git diff --relative=libs/partners/llamastack --name-only --diff-filter=d master | grep -E '\.(py|ipynb)$$' | sed 's|^libs/partners/llamastack/||'))
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff check $(PYTHON_FILES) --fix --unsafe-fixes

spell_check:
	uv run --group codespell codespell --toml pyproject.toml

spell_fix:
	uv run --group codespell codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '===================='
	@echo 'langchain-llamastack'
	@echo '===================='
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'integration_test             - run integration tests'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'spell_check                  - run codespell'
	@echo 'spell_fix                    - run codespell --fix'
