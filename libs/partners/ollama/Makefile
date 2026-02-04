.PHONY: all format lint typing test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test: TEST_FILE = tests/integration_tests/
# TODO(erick) configure ollama server to run in CI, in separate repo

# Define variables for test model configuration
OLLAMA_TEST_MODEL ?= llama3.1
OLLAMA_REASONING_TEST_MODEL ?= deepseek-r1:1.5b


# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)


# integration tests are run without the --disable-socket flag to allow network calls
integration_test:
	OLLAMA_TEST_MODEL=$(OLLAMA_TEST_MODEL) OLLAMA_REASONING_TEST_MODEL=$(OLLAMA_REASONING_TEST_MODEL) uv run --group test --group test_integration pytest $(TEST_FILE)

# CI integration tests - disabled until ollama service is configured in CI

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/ollama --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_ollama
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff

typing:
	mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_ollama -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'typing                       - run type checking'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_test             - run integration tests'
	@echo 'integration_test OLLAMA_TEST_MODEL=<model> - run integration tests with specific model'
	@echo '  Example: make integration_test OLLAMA_TEST_MODEL=llama3.1'
