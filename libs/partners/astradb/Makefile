.PHONY: all format lint test tests integration_tests docker_tests help extended_tests duplicate_check

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
INTEGRATION_TEST_FILE ?= tests/integration_tests/

test:
	poetry run pytest $(TEST_FILE)

tests:
	poetry run pytest $(TEST_FILE)

integration_test:
	poetry run pytest $(INTEGRATION_TEST_FILE)

integration_tests:
	poetry run pytest $(INTEGRATION_TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/astradb --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_astradb
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	poetry run ruff .
	poetry run ruff format $(PYTHON_FILES) --diff
	poetry run ruff --select I $(PYTHON_FILES)
	mkdir $(MYPY_CACHE); poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_astradb -name '*.py')
	poetry run python ./scripts/check_imports.py $^

##############################################
# TRANSITIONAL DUPLICATED-CODE ALIGNMENT CHECK
##############################################

duplicate_check:
	# vector store
	diff \
		-I "import maximal_marginal_relevance" \
		-I alternative_import \
		-I "since=" \
		langchain_astradb/vectorstores.py ../../community/langchain_community/vectorstores/astradb.py
	diff \
		-I ".vectorstores import AstraDBVectorStore" \
		tests/integration_tests/test_vectorstores.py ../../community/tests/integration_tests/vectorstores/test_astradb.py 
	echo "\n\nALL GOOD"

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'duplicate_check              * check duplicate modules community/partners (transitional)'
