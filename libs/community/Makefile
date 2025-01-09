.PHONY: all format lint test tests test_watch integration_tests hpu_tests skip_hpu_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
HPU_TEST_FILES=$(shell find tests -name '*_hpu.py')

# Run unit tests and generate a coverage report.
coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_tests:
	poetry run pytest $(TEST_FILE)

test_watch:
	poetry run ptw --disable-socket --allow-unix-socket --snapshot-update --now . -- -vv tests/unit_tests

# Run HPU-specific tests only if RUN_HPU_TEST is set.
hpu_tests:
	@echo "Setting RUN_HPU_TEST to 1 to enable HPU-specific tests."
	export RUN_HPU_TEST=1 && poetry run pytest --disable-socket --allow-unix-socket $(HPU_TEST_FILES)

extended_tests:
	poetry run pytest --disable-socket --allow-unix-socket --only-extended tests/unit_tests


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
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

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
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'hpu_tests                    - run all HPU-specific tests (files ending with *_hpu.py) if RUN_HPU_TEST is set'
