.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test tests:
	env \
	-u LANGCHAIN_TRACING_V2 \
	-u LANGCHAIN_API_KEY \
	-u LANGSMITH_TRACING \
	-u LANGCHAIN_PROJECT \
	poetry run pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	env \
	-u LANGCHAIN_TRACING_V2 \
	-u LANGCHAIN_API_KEY \
	-u LANGSMITH_TRACING \
	-u LANGCHAIN_PROJECT \
	poetry run ptw --snapshot-update --now . --disable-socket --allow-unix-socket -- -vv $(TEST_FILE)

test_profile:
	poetry run pytest -vv tests/unit_tests/ --profile-svg

check_imports: $(shell find langchain_core -name '*.py')
	poetry run python ./scripts/check_imports.py $^

extended_tests:
	poetry run pytest --only-extended --disable-socket --allow-unix-socket $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/core --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_core
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --fix $(PYTHON_FILES)

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

##########
# CUSTOM #
##########

bench-imports:
	time poetry run python -c "from typing import Any"
	time poetry run python -c "from typing import Any"
	time poetry run python -c "from langchain_core import __version__"
	time poetry run python -c "from langchain_core.tools import tool"
	time poetry run python -c "from langchain_core.messages import AIMessage"
	time poetry run python -c "from langchain_core.documents import Document"
	time poetry run python -c "from langchain_core._api import deprecated"
	time poetry run python -c "from langchain_core._api import beta"
	time poetry run python -c "from langchain_core.callbacks import BaseCallbackHandler"
	time poetry run python -c "from langchain_core.document_loaders import BaseLoader"
	time poetry run python -c "from langchain_core.embeddings import Embeddings"
	time poetry run python -c "from langchain_core.language_models import BaseChatModel"
	time poetry run python -c "from langchain_core.language_models import BaseLLM"
	time poetry run python -c "from langchain_core.load import loads"
	time poetry run python -c "from langchain_core.output_parsers import BaseOutputParser"
	time poetry run python -c "from langchain_core.prompts import ChatPromptTemplate"
	time poetry run python -c "from langchain_core.runnables import RunnableSerializable"
	time poetry run python -c "from langchain_core.tools import BaseTool"
	time poetry run python -c "from langchain_core.tracers import BaseTracer"
	time poetry run python -c "from langchain_core.vectorstores import VectorStore"
