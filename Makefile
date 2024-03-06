.PHONY: all clean docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck

# Default target executed when no arguments are given to make.
all: help


######################
# DOCUMENTATION
######################

clean: docs_clean api_docs_clean


docs_build:
	docs/.local_build.sh

docs_clean:
	@if [ -d _dist ]; then \
			rm -r _dist; \
			echo "Directory _dist has been cleaned."; \
	else \
			echo "Nothing to clean."; \
	fi

docs_linkcheck:
	poetry run linkchecker _dist/docs/ --ignore-url node_modules

api_docs_build:
	poetry run python docs/api_reference/create_api_rst.py
	cd docs/api_reference && poetry run make html

api_docs_clean:
	rm -f docs/api_reference/api_reference.rst
	cd docs/api_reference && poetry run make clean

api_docs_linkcheck:
	poetry run linkchecker docs/api_reference/_build/html/index.html

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

######################
# LINTING AND FORMATTING
######################

lint lint_package lint_tests:
	poetry run ruff docs templates cookbook
	poetry run ruff format docs templates cookbook --diff
	poetry run ruff --select I docs templates cookbook
	git grep 'from langchain import' {docs/docs,templates,cookbook} | grep -vE 'from langchain import (hub)' && exit 1 || exit 0

format format_diff:
	poetry run ruff format docs templates cookbook
	poetry run ruff --select I --fix docs templates cookbook


######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- DOCUMENTATION --'
	@echo 'clean                        - run docs_clean and api_docs_clean'
	@echo 'docs_build                   - build the documentation'
	@echo 'docs_clean                   - clean the documentation build artifacts'
	@echo 'docs_linkcheck               - run linkchecker on the documentation'
	@echo 'api_docs_build               - build the API Reference documentation'
	@echo 'api_docs_clean               - clean the API Reference documentation build artifacts'
	@echo 'api_docs_linkcheck           - run linkchecker on the API Reference documentation'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'
	@echo '-- TEST and LINT tasks are within libs/*/ per-package --'
