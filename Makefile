.PHONY: all clean help docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck spell_check spell_fix lint lint_package lint_tests format format_diff

## help: Show this help info.
help: Makefile
	@printf "\n\033[1mUsage: make <TARGETS> ...\033[0m\n\n\033[1mTargets:\033[0m\n\n"
	@sed -n 's/^## //p' $< | awk -F':' '{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort | sed -e 's/^/  /'

## all: Default target, shows help.
all: help

## clean: Clean documentation and API documentation artifacts.
clean: docs_clean api_docs_clean

######################
# DOCUMENTATION
######################

## docs_build: Build the documentation.
docs_build:
	cd docs && make build

## docs_clean: Clean the documentation build artifacts.
docs_clean:
	cd docs && make clean

## docs_linkcheck: Run linkchecker on the documentation.
docs_linkcheck:
	poetry run linkchecker _dist/docs/ --ignore-url node_modules

## api_docs_build: Build the API Reference documentation.
api_docs_build:
	poetry run python docs/api_reference/create_api_rst.py
	cd docs/api_reference && poetry run make html
	poetry run python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/

API_PKG ?= text-splitters

api_docs_quick_preview:
	poetry run pip install "pydantic<2"
	poetry run python docs/api_reference/create_api_rst.py $(API_PKG)
	cd docs/api_reference && poetry run make html
	poetry run python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/
	open docs/api_reference/_build/html/reference.html

## api_docs_clean: Clean the API Reference documentation build artifacts.
api_docs_clean:
	find ./docs/api_reference -name '*_api_reference.rst' -delete
	git clean -fdX ./docs/api_reference
	rm docs/api_reference/index.md
	

## api_docs_linkcheck: Run linkchecker on the API Reference documentation.
api_docs_linkcheck:
	poetry run linkchecker docs/api_reference/_build/html/index.html

## spell_check: Run codespell on the project.
spell_check:
	poetry run codespell --toml pyproject.toml

## spell_fix: Run codespell on the project and fix the errors.
spell_fix:
	poetry run codespell --toml pyproject.toml -w

######################
# LINTING AND FORMATTING
######################

## lint: Run linting on the project.
lint lint_package lint_tests:
	poetry run ruff check docs templates cookbook
	poetry run ruff format docs templates cookbook --diff
	poetry run ruff check --select I docs templates cookbook
	git grep 'from langchain import' docs/docs templates cookbook | grep -vE 'from langchain import (hub)' && exit 1 || exit 0

## format: Format the project files.
format format_diff:
	poetry run ruff format docs templates cookbook
	poetry run ruff check --select I --fix docs templates cookbook
