.PHONY: all clean help docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck spell_check spell_fix lint lint_package lint_tests format format_diff

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

## help: Show this help info.
help: Makefile
	@printf "\n\033[1mUsage: make <TARGETS> ...\033[0m\n\n\033[1mTargets:\033[0m\n\n"
	@sed -n 's/^## //p' $< | awk -F':' '{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort | sed -e 's/^/  /'

## clean: Clean documentation and API documentation artifacts.
clean: docs_clean api_docs_clean

######################
# DOCUMENTATION
######################

## docs_build: Build the documentation.
docs_build: docs_clean
	@echo "📚 Building LangChain documentation..."
	cd docs && make build
	@echo "✅ Documentation build complete!"

## docs_clean: Clean the documentation build artifacts.
docs_clean:
	@echo "🧹 Cleaning documentation artifacts..."
	cd docs && make clean
	@echo "✅ LangChain documentation cleaned"

## docs_linkcheck: Run linkchecker on the documentation.
docs_linkcheck:
	@echo "🔗 Checking documentation links..."
	@if [ -d _dist/docs ]; then \
		uv run --group test linkchecker _dist/docs/ --ignore-url node_modules; \
	else \
		echo "⚠️  Documentation not built. Run 'make docs_build' first."; \
		exit 1; \
	fi
	@echo "✅ Link check complete"

## api_docs_build: Build the API Reference documentation.
api_docs_build: clean
	@echo "📖 Building API Reference documentation..."
	uv pip install -e libs/cli
	uv run --group docs python docs/api_reference/create_api_rst.py
	cd docs/api_reference && uv run --group docs make html
	uv run --group docs python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/
	@echo "✅ API documentation built"
	@echo "🌐 Opening documentation in browser..."
	open docs/api_reference/_build/html/reference.html

API_PKG ?= text-splitters

api_docs_quick_preview: clean
	@echo "⚡ Building quick API preview for $(API_PKG)..."
	uv run --group docs python docs/api_reference/create_api_rst.py $(API_PKG)
	cd docs/api_reference && uv run --group docs make html
	uv run --group docs python docs/api_reference/scripts/custom_formatter.py docs/api_reference/_build/html/
	@echo "🌐 Opening preview in browser..."
	open docs/api_reference/_build/html/reference.html

## api_docs_clean: Clean the API Reference documentation build artifacts.
api_docs_clean:
	@echo "🧹 Cleaning API documentation artifacts..."
	find ./docs/api_reference -name '*_api_reference.rst' -delete
	git clean -fdX ./docs/api_reference
	rm -f docs/api_reference/index.md
	@echo "✅ API documentation cleaned"

## api_docs_linkcheck: Run linkchecker on the API Reference documentation.
api_docs_linkcheck:
	@echo "🔗 Checking API documentation links..."
	@if [ -f docs/api_reference/_build/html/index.html ]; then \
		uv run --group test linkchecker docs/api_reference/_build/html/index.html; \
	else \
		echo "⚠️  API documentation not built. Run 'make api_docs_build' first."; \
		exit 1; \
	fi
	@echo "✅ API link check complete"

## spell_check: Run codespell on the project.
spell_check:
	@echo "✏️ Checking spelling across project..."
	uv run --group codespell codespell --toml pyproject.toml
	@echo "✅ Spell check complete"

## spell_fix: Run codespell on the project and fix the errors.
spell_fix:
	@echo "✏️ Fixing spelling errors across project..."
	uv run --group codespell codespell --toml pyproject.toml -w
	@echo "✅ Spelling errors fixed"

######################
# LINTING AND FORMATTING
######################

## lint: Run linting on the project.
lint lint_package lint_tests:
	@echo "🔍 Running code linting and checks..."
	uv run --group lint ruff check docs cookbook
	uv run --group lint ruff format docs cookbook cookbook --diff
	git --no-pager grep 'from langchain import' docs cookbook | grep -vE 'from langchain import (hub)' && echo "Error: no importing langchain from root in docs, except for hub" && exit 1 || exit 0
	
	git --no-pager grep 'api.python.langchain.com' -- docs/docs ':!docs/docs/additional_resources/arxiv_references.mdx' ':!docs/docs/integrations/document_loaders/sitemap.ipynb' || exit 0 && \
	echo "Error: you should link python.langchain.com/api_reference, not api.python.langchain.com in the docs" && \
	exit 1
	@echo "✅ Linting complete"

## format: Format the project files.
format format_diff:
	@echo "🎨 Formatting project files..."
	uv run --group lint ruff format docs cookbook
	uv run --group lint ruff check --fix docs cookbook
	@echo "✅ Formatting complete"

update-package-downloads:
	@echo "📊 Updating package download statistics..."
	uv run python docs/scripts/packages_yml_get_downloads.py
	@echo "✅ Package downloads updated"
