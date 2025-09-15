
######################
# LINTING AND FORMATTING
######################

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/cli --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_cli
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --group typing --group lint mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group typing --group lint ruff check --fix $(PYTHON_FILES)

test tests: _test _e2e_test

PYTHON = .venv/bin/python

_test:
	uv run --group test pytest tests

# custom integration testing for cli integration flow
# currently ignores vectorstores test because lacks implementation
_e2e_test:
	rm -rf .integration_test
	mkdir .integration_test
	cd .integration_test && \
		python3 -m venv .venv && \
		$(PYTHON) -m pip install --upgrade poetry && \
		$(PYTHON) -m pip install -e .. && \
		$(PYTHON) -m langchain_cli.cli integration new --name parrot-link --name-class ParrotLink && \
		$(PYTHON) -m langchain_cli.cli integration new --name parrot-link --name-class ParrotLinkB --src=integration_template/chat_models.py --dst=langchain-parrot-link/langchain_parrot_link/chat_models_b.py && \
		$(PYTHON) -m langchain_cli.cli integration create-doc --name parrot-link --name-class ParrotLinkB --component-type ChatModel --destination-dir langchain-parrot-link/docs && \
		cd langchain-parrot-link && \
			poetry install --with lint,typing,test && \
			poetry run pip install -e ../../../standard-tests && \
			make format lint tests && \
			poetry install --with test_integration && \
			poetry run pip install -e ../../../core && \
			make integration_test 
