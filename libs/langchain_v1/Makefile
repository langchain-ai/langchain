.PHONY: all clean docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck format lint test tests test_watch integration_tests help extended_tests start_services stop_services

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

start_services:
	docker compose -f tests/unit_tests/agents/compose-postgres.yml -f tests/unit_tests/agents/compose-redis.yml up -V --force-recreate --wait --remove-orphans

stop_services:
	docker compose -f tests/unit_tests/agents/compose-postgres.yml -f tests/unit_tests/agents/compose-redis.yml down -v

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Run unit tests and generate a coverage report.
coverage:
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

test:
	make start_services && LANGGRAPH_TEST_FAST=0 uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE) --snapshot-update --cov-report term-missing:skip-covered; \
	EXIT_CODE=$$?; \
	make stop_services; \
	exit $$EXIT_CODE

test_fast:
	LANGGRAPH_TEST_FAST=1 uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

extended_tests:
	make start_services && LANGGRAPH_TEST_FAST=0 uv run --group test pytest --disable-socket --allow-unix-socket --only-extended tests/unit_tests; \
	EXIT_CODE=$$?; \
	make stop_services; \
	exit $$EXIT_CODE

test_watch:
	make start_services && LANGGRAPH_TEST_FAST=0 uv run --group test ptw --snapshot-update --now . -- -x --disable-socket --allow-unix-socket --disable-warnings tests/unit_tests; \
	EXIT_CODE=$$?; \
	make stop_services; \
	exit $$EXIT_CODE

test_watch_extended:
	make start_services && LANGGRAPH_TEST_FAST=0 uv run --group test ptw --snapshot-update --now . -- -x --disable-socket --allow-unix-socket --only-extended tests/unit_tests; \
	EXIT_CODE=$$?; \
	make stop_services; \
	exit $$EXIT_CODE

integration_tests:
	uv run --group test --group test_integration pytest tests/integration_tests

check_imports: $(shell find langchain -name '*.py')
	uv run python ./scripts/check_imports.py $^

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

spell_check:
	uv run --all-groups codespell --toml pyproject.toml

spell_fix:
	uv run --all-groups codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '===================='
	@echo 'clean                        - run docs_clean and api_docs_clean'
	@echo 'docs_build                   - build the documentation'
	@echo 'docs_clean                   - clean the documentation build artifacts'
	@echo 'docs_linkcheck               - run linkchecker on the documentation'
	@echo 'api_docs_build               - build the API Reference documentation'
	@echo 'api_docs_clean               - clean the API Reference documentation build artifacts'
	@echo 'api_docs_linkcheck           - run linkchecker on the API Reference documentation'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'
	@echo '-- TESTS --'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'test                         - run unit tests with all services'
	@echo 'test_fast                    - run unit tests with in-memory services only'
	@echo 'tests                        - run unit tests (alias for "make test")'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'extended_tests               - run only extended unit tests'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'integration_tests            - run integration tests'
	@echo '-- DOCUMENTATION tasks are from the top-level Makefile --'
