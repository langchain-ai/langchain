.PHONY: all format lint test tests integration_tests help extended_tests clean install

# Default target
all: help

######################
# INSTALLATION
######################

install:
	pip install -e ".[test]"

######################
# TESTING
######################

test tests:
	pytest tests/unit_tests -v \
		--cov=langchain_sarvam \
		--cov-report=term-missing \
		--cov-report=xml \
		--junitxml=junit/test-results.xml

integration_tests:
	pytest tests/integration_tests -v \
		--cov=langchain_sarvam \
		--cov-report=term-missing \
		--cov-report=xml

extended_tests:
	$(MAKE) test
	$(MAKE) integration_tests

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/sarvam --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_sarvam
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	ruff check $(PYTHON_FILES)
	ruff format --check $(PYTHON_FILES)
	mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_sarvam -name '*.py')
	python -m scripts.check_imports $^

######################
# CLEANING
######################

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache*" -exec rm -rf {} +
	find . -type d -name "junit" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage*
	rm -rf coverage.xml
	rm -rf htmlcov/

######################
# BUILD
######################

build:
	python -m build

publish:
	python -m twine upload dist/*

######################
# HELP
######################

help:
	@echo '----'
	@echo 'install               - install package in editable mode with test dependencies'
	@echo 'format                - run code formatters'
	@echo 'lint                  - run linters'
	@echo 'test                  - run unit tests'
	@echo 'tests                 - run unit tests (alias for test)'
	@echo 'integration_tests     - run integration tests'
	@echo 'extended_tests        - run all tests'
	@echo 'clean                 - clean all build and test artifacts'
	@echo 'build                 - build package'
	@echo 'publish               - publish package to PyPI'
	@echo 'spell_check           - run codespell on the project'
	@echo 'spell_fix             - run codespell on the project and fix issues'
	@echo 'check_imports         - check imports'
	@echo 'help                  - print this help message'
