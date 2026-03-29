.PHONY: all format lint type test tests integration_tests help

all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=

integration_tests: TEST_FILE = tests/integration_tests/

test integration_tests:
uv run --group test --group test_integration pytest $(PYTEST_EXTRA) $(TEST_FILE)

tests:
uv run --group test pytest $(PYTEST_EXTRA) $(TEST_FILE)

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=langchain_forcefield
lint_tests: PYTHON_FILES=tests
UV_RUN_LINT = uv run --all-groups
UV_RUN_TYPE = uv run --all-groups
lint_package lint_tests: UV_RUN_LINT = uv run --group lint

lint lint_diff lint_package lint_tests:
./scripts/lint_imports.sh
[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(UV_RUN_TYPE) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --fix $(PYTHON_FILES)

check_imports:
$(UV_RUN_LINT) python ./scripts/check_imports.py

help:
@echo '----'
@echo 'format                       - run code formatters'
@echo 'lint                         - run linters'
@echo 'test                         - run unit tests'
