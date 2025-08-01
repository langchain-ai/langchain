# Makefile for LangChain-Coherence Integration
.PHONY: all format lint fix check lint_package lint_tests lint_diff format_diff integration_tests check_imports test tests install update-dev update-core lock sync

# ======================
# Environment Paths
# ======================
PYTHON := uv run --active python
PIP := uv pip
RUFF := uv run --active ruff
MYPY := uv run --active mypy
PYTEST := uv run --active pytest

# ======================
# Default
# ======================
all: help

# ======================
# Core Lint / Type Check
# ======================
lint: check

check:
	@echo "🔍 Running linter and type checker..."
	$(RUFF) check langchain_coherence tests
	$(MYPY) --explicit-package-bases langchain_coherence

format:
	@echo "🎨 Formatting Python code and docs notebooks..."
	$(RUFF) format langchain_coherence tests ../../../docs/docs/integrations/vectorstores/coherence.ipynb

fix:
	@echo "🔧 Fixing lint issues..."
	$(MAKE) format
	$(RUFF) check langchain_coherence tests ../../../docs/docs/integrations/vectorstores/coherence.ipynb --fix

# ======================
# New CI-Expected Lint Targets
# ======================
lint_package:
	@echo "🔍 Linting package..."
	$(RUFF) check langchain_coherence
	$(MYPY) --explicit-package-bases langchain_coherence

lint_tests:
	@echo "🔍 Linting tests..."
	$(RUFF) check tests
	$(MYPY) --explicit-package-bases tests

lint_diff:
	@echo "🔍 Linting changed files..."
	@changed_files=$$(git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' || true); \
	if [ "$$changed_files" ]; then \
		$(RUFF) check $$changed_files; \
	else \
		echo "No changed Python files."; \
	fi

format_diff:
	@echo "🎨 Formatting changed files..."
	@changed_files=$$(git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' || true); \
	if [ "$$changed_files" ]; then \
		$(RUFF) format $$changed_files; \
	else \
		echo "No changed Python files."; \
	fi

# ======================
# Import Checks
# ======================
check_imports:
	@echo "🔍 Checking imports..."
	$(PYTHON) ./scripts/check_imports.py

# ======================
# Testing
# ======================
test tests:
	@if [ -d tests/unit_tests ] && [ "$$(find tests/unit_tests -type f -name 'test_*.py')" ]; then \
		$(PYTEST) --disable-socket --allow-unix-socket tests/unit_tests; \
	else \
		echo "⚠️  No unit tests found — running all tests instead."; \
		$(PYTEST) --disable-socket --allow-unix-socket tests; \
	fi

integration_tests:
	@echo "🧪 Running integration tests..."
	$(PYTEST) tests/integration_tests

# ======================
# Dependency Management
# ======================
install:
	@echo "🔧 Installing all dependencies..."
	$(PIP) install -e . \
		--group lint \
		--group typing \
		--group test \
		--group docs \
		--group publish

update-dev:
	@echo "🔄 Updating development dependencies..."
	$(PIP) install -e . \
		--group lint \
		--group typing \
		--group test \
		--group docs \
		--group publish \
		--upgrade

update-core:
	@echo "🔄 Updating core dependencies..."
	$(PIP) install --upgrade langchain-core coherence-client

lock:
	@echo "🔐 Locking all dependencies to uv.lock..."
	uv pip compile \
		--group lint \
		--group typing \
		--group test \
		--group docs \
		--group publish

sync:
	@echo "📦 Syncing dependencies from uv.lock..."
	uv pip install -r uv.lock

# ======================
# Help
# ======================
help:
	@echo "Available targets:"
	@grep '^[a-zA-Z_-]\+:' Makefile | cut -d: -f1
