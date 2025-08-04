.PHONY: format lint test tests unit_test unit_tests integration_tests help extended_tests format_diff lint_diff lint_package lint_tests spell_check spell_fix check_imports

## help: Show this help info.
help: Makefile
	@printf "\n\033[1mUsage: make <TARGETS> ...\033[0m\n\n\033[1mTargets:\033[0m\n\n"
	@sed -n 's/^## //p' $< | awk -F':' '{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort | sed -e 's/^/  /'

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
INTEGRATION_TEST_FILE ?= tests/integration_tests/

integration_test integration_tests: TEST_FILE=$(INTEGRATION_TEST_FILE)

## test: Run both unit and integration tests
test tests: unit_test integration_tests

## unit_test: Run only unit tests
unit_test unit_tests:
	uv run --group test pytest $(TEST_FILE)

## integration_test: Run only integration tests
integration_test integration_tests:
	uv run --group test --group test_integration pytest $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/standard-tests --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_tests
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

## lint: Run linters on all files
## lint_diff: Run linters on changed files only
## lint_package: Run linters on package files only
## lint_tests: Run linters on test files only
lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

## format: Format all code
## format_diff: Format changed files only
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

## spell_check: Check spelling in code
spell_check:
	uv run --all-groups codespell --toml pyproject.toml

## spell_fix: Apply spelling fixes in code
spell_fix:
	uv run --all-groups codespell --toml pyproject.toml -w

## check_imports: Check import statements
check_imports: $(shell find langchain_tests -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

