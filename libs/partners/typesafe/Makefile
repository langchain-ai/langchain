.PHONY: all format lint type test tests integration_tests help

all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=
integration_test integration_tests: TEST_FILE = tests/integration_tests/

test tests:
	env -u LANGCHAIN_TRACING_V2 -u LANGCHAIN_API_KEY -u LANGSMITH_API_KEY -u LANGSMITH_TRACING -u LANGCHAIN_PROJECT uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest -v --tb=short -n auto $(PYTEST_EXTRA) $(TEST_FILE)

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/typesafe --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_typesafe
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

check_imports: $(shell find langchain_typesafe -name '*.py')
	$(UV_RUN_LINT) python ./scripts/check_imports.py $^

check_version:
	uv run python ./scripts/check_version.py

help:
	@echo '----'
	@echo 'check_imports             - check imports'
	@echo 'check_version             - validate version consistency'
	@echo 'format                    - run code formatters'
	@echo 'lint                      - run linters and type checking'
	@echo 'test                      - run unit tests'
