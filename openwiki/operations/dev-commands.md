---
type: "Developer Tools & Commands"
title: "Development Commands and Local Setup"
description: "Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows."
tags: [development, build, testing, linting, typing, uv, make, pre-commit, local-setup]
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-f708a9db48bfcf1b154e4708
    resource: repo://libs/langchain/Makefile
  - id: openwiki-source-f4436232e0451a04247e92e5
    resource: repo://libs/langchain/pyproject.toml
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-a6e669bb11f217c6fbd06670
    resource: repo://libs/partners/anthropic/Makefile
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
---

## Overview

The LangChain Python monorepo uses `uv` for dependency management, `make` for task automation, and `ruff`/`mypy` for code quality. This page provides a quick reference for common development commands and setup workflows.

## Initial Setup

### Install Dependencies Per Package

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`. Before running tests or making changes, set up dependencies:

```bash
# Navigate to package directory
cd libs/core

# Install all dependency groups (lint, typing, test, dev)
uv sync --all-groups

# Or install only specific groups
uv sync --group test
uv sync --group lint
uv sync --group typing
```

The `--all-groups` flag ensures you have tools for linting, type checking, and testing. Common dependency groups defined in `pyproject.toml`:

- **lint**: Ruff linter and formatter
- **typing**: Mypy type checker and stubs
- **test**: Pytest and testing utilities
- **test_integration**: Integration test dependencies (API clients, mock servers)
- **dev**: Development tools (Jupyter, setuptools)

### uv Essentials

`uv` is a fast Python package installer and dependency resolver that replaces `pip`, `poetry`, and `conda`. Key principles:

- **Automatic environment management**: `uv sync` and `uv run` manage virtual environments automatically—do not manually source `.venv/bin/activate`
- **Per-package isolation**: Each package in `libs/` maintains its own environment via `uv.lock`; do not create ad-hoc virtual environments
- **Explicit dependency groups**: Use group selectors for faster setup
- **Frozen lock files**: `UV_FROZEN=true` in Makefiles prevents unintended changes

Common uv commands:

```bash
# Sync dependencies (respects groups and lock file)
uv sync --all-groups

# Run commands in package environment
uv run --group test pytest tests/unit_tests/

# Add a dependency temporarily for testing
uv run --with 'anthropic>=1.0.0' --group test pytest tests/integration_tests/

# Update lock file (when dependencies change)
uv lock

# Check lock file is up-to-date
uv lock --check
```

### Pre-Commit Setup

The repository uses pre-commit hooks to enforce code quality automatically on commit. Install once per repository:

```bash
pre-commit install
```

Pre-commit then runs automatically on staged files before each commit. To manually trigger:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook by ID
pre-commit run core --all-files
pre-commit run lint --all-files
```

#### Bypassing Pre-Commit Hooks

To skip hooks temporarily (use with caution):

```bash
# Skip all hooks for this commit
git commit --no-verify

# Skip specific hooks via SKIP environment variable
SKIP=core,langchain pre-commit run --all-files
```

## Pre-Commit Hooks Details

The `.pre-commit-config.yaml` defines comprehensive hooks that enforce:

### Standard Validation
- **YAML/TOML syntax**: `check-yaml`, `check-toml`
- **File formatting**: Ensures files end with newline, removes trailing whitespace
- **Text normalization**: Fixes curly quotes and non-standard spaces

### Per-Package Format and Lint
Each package in `libs/` has a local hook running:
```
make -C libs/<package> format lint
```

This applies ruff formatting, linting, and mypy type checking. Hook IDs in pre-commit:
- `core`: libs/core
- `langchain`: libs/langchain  
- `langchain_v1`: libs/langchain_v1
- `standard-tests`: libs/standard-tests
- `text-splitters`: libs/text-splitters
- Partner packages: `anthropic`, `chroma`, `exa`, `fireworks`, `groq`, `huggingface`, `mistralai`, `nomic`, `ollama`, `openai`, `qdrant`, etc.

### Version Consistency Checks
Each package with a version file has a hook ensuring `pyproject.toml` version matches source code:
```
make -C libs/<package> check_version
```

Examples: `core-version`, `langchain-v1-version`, `anthropic-version`, etc.

### Branch Protection
Prevents direct commits to `master` branch.

## Linting and Formatting

### Complete Lint Suite

```bash
# From package directory
make lint
```

This runs three checks in sequence:
1. **Import organization**: `./scripts/lint_imports.sh`
2. **Ruff check**: Logical errors, naming conventions, code smells
3. **Ruff format (diff mode)**: Format checking (no modifications)
4. **Mypy**: Static type checking

### Format Code

```bash
make format
```

Applies auto-fixes:
- `ruff format`: Formats code (imports, spacing, etc.)
- `ruff check --fix`: Auto-fixes linting issues (unsorted imports, unused variables)

**Note**: Always run `make format` before `make lint` since formatting can resolve some lint issues.

### Ruff-Only Commands

For faster iteration during development, run ruff separately:

```bash
# Check without fixing
uv run --group lint ruff check .
uv run --group lint ruff format . --diff

# Apply fixes
uv run --group lint ruff check . --fix
uv run --group lint ruff format .
```

Ruff processes Python files and Jupyter notebooks. Run from within a package directory.

### Lint Changes Only

For faster iteration, lint only files changed in your branch:

```bash
make lint_diff
make format_diff
```

These targets apply `git diff` against `master` branch.

### Type Checking

Full type checking with mypy:

```bash
make type
```

Type checking verifies correct argument types, required type hints, and type consistency across code. Can be slow for large packages.

To type-check specific paths:

```bash
# Single file
uv run --group typing mypy libs/core/langchain_core/runnables.py

# Only tests (faster)
cd libs/langchain && make lint_tests
```

## Testing Commands

### Run All Unit Tests

```bash
cd libs/core
make test
```

Unit tests live in `tests/unit_tests/` with no network calls allowed. Tests run with:
- `pytest` for test execution
- `xdist` (-n auto) for parallelization
- `--disable-socket --allow-unix-socket` to prevent accidental network calls

### Run Specific Test File or Directory

```bash
# Using make with TEST_FILE parameter
make test TEST_FILE=tests/unit_tests/agents/test_agent.py

# Or use uv directly
uv run --group test pytest tests/unit_tests/agents/test_agent.py -v

# Specify test class or method
uv run --group test pytest tests/unit_tests/agents/test_agent.py::TestAgent::test_invoke -v
```

The `TEST_FILE` parameter accepts:
- Single file: `tests/unit_tests/agents/test_agent.py`
- Directory: `tests/unit_tests/agents/`
- Test identifier: `tests/unit_tests/agents/test_agent.py::TestClass::test_method`

### Test in Watch Mode

Auto-re-run tests as code changes:

```bash
make test_watch
```

Uses `pytest-watcher` (via `ptw` command) and automatically updates snapshots. Ideal for test-driven development.

### Extended Tests

Some packages mark slower tests with `@pytest.mark.extended`:

```bash
make extended_tests
```

### Coverage Reports

Generate code coverage reports:

```bash
make coverage
```

Produces both XML and term-missing reports (lines not covered), useful for understanding untested code paths.

### Integration Tests

Integration tests live in `tests/integration_tests/` and require network access and API keys:

```bash
make integration_tests
```

Or run with specific SDK versions for compatibility testing:

```bash
cd libs/partners/anthropic
uv run --with 'anthropic>=0.120.0,<1.0.0' --group test --group test_integration pytest tests/integration_tests/
uv run --with 'anthropic>=1.0.0,<2.0.0' --group test --group test_integration pytest tests/integration_tests/
```

## Debugging Tests

### Verbose Output

```bash
# With single verbosity (-v)
uv run --group test pytest tests/unit_tests/agents/test_agent.py -v

# With double verbosity (-vv) shows assertion details and progress
uv run --group test pytest tests/unit_tests/ -vv

# Use make with PYTEST_EXTRA parameter
make test PYTEST_EXTRA="-vv"
```

### Drop Into Debugger

```bash
# Stop at first failure and drop into pdb
uv run --group test pytest tests/unit_tests/agents/test_agent.py --pdb

# Trace execution (runs slower)
uv run --group test pytest tests/unit_tests/agents/test_agent.py --trace

# Drop into pdb on failures
uv run --group test pytest tests/unit_tests/agents/test_agent.py --pdb --tb=short
```

### List Tests Without Running

```bash
# Collect and display all tests
uv run --group test pytest tests/unit_tests/ --collect-only

# Collect tests matching a pattern
uv run --group test pytest --collect-only -q tests/unit_tests/ -k "agent"
```

### Run Tests Matching a Pattern

```bash
# Run all tests with "agent" in name
uv run --group test pytest -k "test_agent" tests/unit_tests/

# Exclude tests
uv run --group test pytest -k "not slow" tests/unit_tests/

# Use make with PYTEST_EXTRA
make test PYTEST_EXTRA="-k test_agent -vv"
```

### Stop on First Failure

```bash
# Exit on first failure
uv run --group test pytest tests/unit_tests/agents/ -x

# Use in make
make test PYTEST_EXTRA="-x"
```

## Per-Package Development Workflow

To develop a single package in the monorepo:

```bash
# Example: work on langchain_v1
cd libs/langchain_v1

# 1. Install all dependencies
uv sync --all-groups

# 2. Run tests on changes
make test_watch  # Terminal 1

# 3. Edit code in Terminal 2
# Changes auto-trigger re-run in Terminal 1

# 4. Before committing
make format
make lint
make test

# 5. Commit (pre-commit hooks run automatically)
git commit -m "feat(langchain_v1): description"
```

Each package follows the same Makefile structure with consistent targets.

## Make Targets Reference

All packages support these standard make targets:

| Target | Purpose |
|--------|---------|
| `make test` | Run all unit tests (parallel with xdist) |
| `make test TEST_FILE=<path>` | Run tests in specific file/directory |
| `make test_watch` | Run tests in watch mode (auto-rerun) |
| `make extended_tests` | Run only `@pytest.mark.extended` tests |
| `make integration_tests` | Run integration tests (requires API keys) |
| `make coverage` | Run tests and generate coverage report |
| `make lint` | Run ruff check + ruff format --diff + mypy |
| `make lint_diff` | Lint only changed files (git diff vs master) |
| `make lint_package` | Lint package code only (skip tests) |
| `make lint_tests` | Lint test code only (faster) |
| `make format` | Apply ruff format + ruff check --fix |
| `make format_diff` | Format only changed files |
| `make type` | Run mypy type checking |
| `make check_version` | Validate version consistency (package-specific) |
| `make help` | Display available targets |

Package-specific targets vary; run `make help` in a package directory to see all available options.

## Environment Variables

The Makefiles use environment variables to control behavior:

| Variable | Purpose | Default |
|----------|---------|---------|
| `UV_FROZEN` | Prevent lock file changes during `uv sync` | `true` (in Makefiles) |
| `TEST_FILE` | Path to test file or directory | `tests/unit_tests/` |
| `PYTEST_EXTRA` | Extra pytest options (e.g., `-vv`, `-x`) | (empty) |
| `PYTHON_FILES` | Files to lint/format | `.` (all files) |
| `MYPY_CACHE` | Mypy cache directory | `.mypy_cache` |

Example: Run fast tests with verbosity:

```bash
make test PYTEST_EXTRA="-vv" TEST_FILE=tests/unit_tests/agents
```

Example: Run lint_tests with custom cache:

```bash
make lint_tests MYPY_CACHE=.mypy_cache_test
```

## Lock File Management

The `uv.lock` file in each package pins exact dependency versions for reproducible builds. Update locks when `pyproject.toml` changes:

```bash
# From a package directory
uv lock

# Regenerate all package locks
cd libs && make lock

# Verify all locks are up-to-date (CI check)
cd libs && make check-lock
```

The Makefiles set `UV_FROZEN = true` to prevent unintended lock changes during regular development. Use the commands above when intentionally updating dependencies.

## Common Workflows

### Before Committing

```bash
# 1. Format code
make format

# 2. Run linting and type checks
make lint

# 3. Run tests
make test

# 4. Commit (pre-commit hooks run automatically)
git commit -m "feat(core): description"
```

Alternatively, let pre-commit hooks handle formatting/linting:

```bash
make test
git add .
git commit  # or pre-commit run --all-files first
```

### Iterative Development

For fast feedback during development:

```bash
# Terminal 1: Watch tests
make test_watch

# Terminal 2: Edit code and format on save
make format
```

Tests in Terminal 1 auto-rerun when files change.

### Fixing Lint Errors

When `make lint` fails:

```bash
# 1. Auto-fix formatting and simple lint issues
make format

# 2. Re-check
make lint

# 3. Fix remaining issues manually (e.g., mypy type errors)
# Edit source files, then:
make type
```

For faster iteration on specific files:

```bash
# Format and lint single file
uv run --group lint ruff check --fix langchain/agents/agent.py
uv run --group lint ruff format langchain/agents/agent.py
uv run --group typing mypy langchain/agents/agent.py
```

## Troubleshooting

### Lock file out of sync

```bash
# Regenerate lock for current package
uv lock

# Or check if lock is up-to-date
uv lock --check

# If conflicts, reset and regenerate
rm uv.lock
uv lock
```

### Dependencies not installed

```bash
# Ensure all groups are installed
uv sync --all-groups

# Or just the test group
uv sync --group test

# Reinstall from scratch
rm uv.lock
uv sync --all-groups
```

### Tests fail with "no network" error

This is intentional—unit tests have socket restrictions to prevent accidental network calls. For integration tests:

```bash
make integration_tests
```

To bypass socket restrictions temporarily (for debugging):

```bash
uv run --group test pytest --disable-socket=false tests/unit_tests/
```

### Ruff or mypy not found

```bash
# Install lint and typing groups
uv sync --group lint --group typing

# Or just all groups
uv sync --all-groups
```

### Pre-commit hook fails locally but passes in CI

Ensure you're using the same Python version and have all dependency groups:

```bash
python --version
uv sync --all-groups
pre-commit run --all-files
```

### Cache clearing

If you encounter stale cache issues:

```bash
# Clear mypy cache
rm -rf .mypy_cache .mypy_cache_test

# Clear pytest cache
rm -rf .pytest_cache

# Clear ruff cache
rm -rf .ruff_cache

# Full clean: reinstall environment
rm uv.lock
uv sync --all-groups
```

### Dependency conflicts

When dependency resolution fails:

```bash
# From package directory
rm uv.lock
uv lock

# Or check for specific conflicts
uv sync --all-groups
```

## Related Documentation

- [Quickstart](/openwiki/quickstart.md): Getting started with LangChain development
- [CI Workflows](/openwiki/operations/ci-workflows.md): GitHub Actions, release processes, and automation
- [Unit Tests](/openwiki/testing/unit-tests.md): Unit test structure, fixtures, and best practices
- [Integration Tests](/openwiki/testing/integration-tests.md): Integration test setup, Docker services, and API mocking
