.PHONY: all format lint typing test tests integration_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE=tests/integration_tests/

# unit tests are run with the --disable-socket flag to prevent network calls
# use tiktoken cache to enable token counting without socket (internet) access
test tests:
	mkdir -p tiktoken_cache
	@if [ ! -f tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 ]; then \
		curl -o tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken; \
	fi
	@if [ ! -f tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790 ]; then \
		curl -o tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790 https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken; \
	fi
	TIKTOKEN_CACHE_DIR=tiktoken_cache uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

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
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/openai --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_openai
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

check_imports: $(shell find langchain_openai -name '*.py')
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
