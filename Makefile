.PHONY: all clean help docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck spell_check spell_fix lint lint_package lint_tests format format_diff

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

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
	uv run --no-group test linkchecker _dist/docs/ --ignore-url node_modules

## api_docs_build: Build the API Reference documentation.
api_docs_build:
	uv run --no-group test python docs/api_reference/create_api_rst.py
	cd docs/api_reference && uv run --no-group test make html
	uv run --no-group test python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/

API_PKG ?= text-splitters

api_docs_quick_preview:
	uv run --no-group test python docs/api_reference/create_api_rst.py $(API_PKG)
	cd docs/api_reference && uv run make html
	uv run --no-group test python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/
	open docs/api_reference/_build/html/reference.html

## api_docs_clean: Clean the API Reference documentation build artifacts.
api_docs_clean:
	find ./docs/api_reference -name '*_api_reference.rst' -delete
	git clean -fdX ./docs/api_reference
	rm -f docs/api_reference/index.md
	

## api_docs_linkcheck: Run linkchecker on the API Reference documentation.
api_docs_linkcheck:
	uv run --no-group test linkchecker docs/api_reference/_build/html/index.html

## spell_check: Run codespell on the project.
spell_check:
	uv run --no-group test codespell --toml pyproject.toml

## spell_fix: Run codespell on the project and fix the errors.
spell_fix:
	uv run --no-group test codespell --toml pyproject.toml -w

######################
# LINTING AND FORMATTING
######################

## lint: Run linting on the project.
lint lint_package lint_tests:
	uv run --group lint ruff check docs cookbook
	uv run --group lint ruff format docs cookbook cookbook --diff
	uv run --group lint ruff check --select I docs cookbook
	git --no-pager grep 'from langchain import' docs cookbook | grep -vE 'from langchain import (hub)' && echo "Error: no importing langchain from root in docs, except for hub" && exit 1 || exit 0
	
	git --no-pager grep 'api.python.langchain.com' -- docs/docs ':!docs/docs/additional_resources/arxiv_references.mdx' ':!docs/docs/integrations/document_loaders/sitemap.ipynb' || exit 0 && \
	echo "Error: you should link python.langchain.com/api_reference, not api.python.langchain.com in the docs" && \
	exit 1

## format: Format the project files.
format format_diff:
	uv run --group lint ruff format docs cookbook
	uv run --group lint ruff check --select I --fix docs cookbook

update-package-downloads:
	uv run python docs/scripts/packages_yml_get_downloads.py
