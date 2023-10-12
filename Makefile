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
	rm -r _dist

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
