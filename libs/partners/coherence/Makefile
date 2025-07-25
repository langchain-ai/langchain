# Makefile for LangChain-Coherence Integration
.PHONY: install update-dev update-core lock sync lint format fix check test clean docs-ipynb readme-ipynb create-ipynb help

# Paths to common tools (adjust if using .uv or other env)
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest
MYPY := .venv/bin/mypy

install:
	@echo "🔧 Installing all dependencies..."
	uv venv
	uv pip install -e .[lint,typing,test,docs]

update-dev:
	@echo "🔄 Updating development dependencies..."
	uv pip install -e .[lint,typing,test,docs] --upgrade

update-core:
	@echo "🔄 Updating core dependencies..."
	uv pip install --upgrade langchain-core coherence-client

lock:
	@echo "🔐 Locking all dependencies to uv.lock..."
	uv pip compile --all-extras

sync:
	@echo "📦 Syncing dependencies from uv.lock..."
	uv pip install -r uv.lock

lint: check

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
	rm -rf .pytest_cache .mypy_cache .ruff_cache .venv .uv __pycache__ *.egg-info dist build

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
	@echo "  clean         - Remove temp and build files"


