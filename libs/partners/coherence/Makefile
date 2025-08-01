# Makefile for LangChain-Coherence Integration
.PHONY: install update-dev update-core lock sync lint format fix check test clean docs-ipynb readme-ipynb create-ipynb help

# Paths to common tools (adjust if using .uv or other env)
PYTHON := $(VIRTUAL_ENV)/bin/python
PIP := $(VIRTUAL_ENV)/bin/pip
RUFF := $(VIRTUAL_ENV)/bin/ruff
MYPY := $(VIRTUAL_ENV)/bin/mypy
PYTEST := $(VIRTUAL_ENV)/bin/pytest

PACKAGE_NAME=langchain-coherence
DIST_DIR=dist

install:
	@echo "🔧 Installing all dependencies..."
	uv pip install -e . \
		--group lint \
		--group typing \
		--group test \
		--group docs \
		--group publish

update-dev:
	@echo "🔄 Updating development dependencies..."
	uv pip install -e . \
		--group lint \
		--group typing \
		--group test \
		--group docs \
		--group publish \
		--upgrade

update-core:
	@echo "🔄 Updating core dependencies..."
	uv pip install --upgrade langchain-core coherence-client

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

lint: check

# Lint only this package (required by LangChain CI)
lint_package: check
	@echo "✅ Package linting complete."

check:
	@echo "🔍 Running linter and type checker..."
	$(RUFF) check langchain_coherence tests
	$(MYPY) --explicit-package-bases langchain_coherence

format:
	@echo "🎨 Formatting code with Ruff..."
	$(RUFF) format langchain_coherence tests

fix:
	@echo "🔧 Fixing lint issues..."
	$(MAKE) format
	$(RUFF) check langchain_coherence tests --fix

test:
	@echo "🧪 Running tests..."
	$(PYTEST)

clean:
	@echo "🧹 Cleaning build/test artifacts..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ *.egg-info dist build

build:
	@echo "🧱 Building distribution using local virtualenv"
	$(PYTHON) -m build --no-isolation

upload-pypi:
	@echo "🚀 Uploading to PyPI"
	$(PYTHON) -m twine upload dist/*

publish: build upload-pypi

help:
	@echo "🛠 Available Make targets:"
	@echo "  install       - Install all dependencies into .venv"
	@echo "  update-dev    - Upgrade dev dependencies (ruff, pytest, etc.)"
	@echo "  update-core   - Upgrade core runtime deps (langchain-core, coherence-client)"
	@echo "  lock          - Generate uv.lock with pinned versions"
	@echo "  sync          - Install from uv.lock (repeatable builds)"
	@echo "  lint          - Run linter and mypy"
	@echo "  fix           - Autoformat and fix issues"
	@echo "  test          - Run all tests"
	@echo "  build         - Building distribution using local virtualenv"
	@echo "  upload-pypi   - Uploading to PyPI"
	@echo "  publish       - calls build, upload-pypi"
	@echo "  clean         - Remove temp and build files"


