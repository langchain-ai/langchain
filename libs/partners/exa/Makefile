.PHONY: all format lint type test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=

integration_tests: TEST_FILE=tests/integration_tests/

test integration_tests:
	uv run --group test --group test_integration pytest $(PYTEST_EXTRA) $(TEST_FILE)

tests:
	uv run --group test pytest $(PYTEST_EXTRA) $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/exa --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_exa
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test
UV_RUN_LINT = uv run --all-groups
UV_RUN_TYPE = uv run --all-groups
lint_package lint_tests: UV_RUN_LINT = uv run --group lint

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(UV_RUN_TYPE) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && $(UV_RUN_TYPE) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_exa -name '*.py')
	$(UV_RUN_LINT) python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
