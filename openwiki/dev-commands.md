---
type: "Developer Tools & Commands"
title: "Development Commands and Local Setup"
description: "Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows."
tags: [development, build, testing, linting, typing, uv, make, pre-commit, local-setup, conventional-commits]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-20T08:25:57.412Z
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-7c96a74af67942d40559bf7d
    resource: repo://libs/langchain_v1/Makefile
  - id: openwiki-source-f708a9db48bfcf1b154e4708
    resource: repo://libs/langchain/Makefile
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-a6e669bb11f217c6fbd06670
    resource: repo://libs/partners/anthropic/Makefile
generated: { by: "openwiki/0.5.0", at: "2026-10-20T08:25:57.412Z" }
---

## Overview

The LangChain Python monorepo uses `uv` for dependency management, `make` for task automation, and `ruff`/`mypy` for code quality. This page provides a quick reference for common development commands and setup workflows.

## Initial Setup

### Install Dependencies

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`. Before running tests or making changes, set up dependencies:

```bash
# Install all dependency groups (lint, typing, test, dev)
uv sync --all-groups

# Or install only a specific group
uv sync --group test
uv sync --group lint
```

The `--all-groups` flag ensures you have tools for linting, type checking, and testing. See the [Contributing Guide in AGENTS.md](repo://AGENTS.md) for detailed development conventions and PR guidelines.

### uv Basics

`uv` is a fast Python package installer and dependency resolver that replaces `pip`, `poetry`, and `conda`. Key points:

- **No manual activation needed**: `uv sync` and `uv run` manage virtual environments automatically — do not source `.venv/bin/activate`.
- **Per-package environments**: Each package in `libs/` maintains its own environment via `uv.lock`. Do not create ad-hoc virtual environments outside package directories.
- **Dependency groups**: Install specific groups for faster setup: `--group test` (testing), `--group lint` (linting), `--group typing` (type checking), or all with `--all-groups`.
- **No implicit installation**: Use explicit `uv sync` calls with group selectors; never let dependencies install implicitly.

Common uv commands:

```bash
# Install/sync dependencies (respects groups and lock file)
uv sync --all-groups

# Run a command in the package environment
uv run --group test pytest tests/unit_tests/

# Update lock file (see Lock File Management below)
uv lock

# Run a command with an additional dependency (temporary)
uv run --with 'package>=1.0' --group test pytest tests/integration_tests/
```

### Pre-Commit Setup

The repository uses pre-commit hooks to enforce code quality on commit. Install and configure them once:

```bash
pre-commit install
```

Pre-commit runs automatically on staged files before each commit. To manually trigger hooks:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
```

#### Bypassing Pre-Commit Hooks

To skip pre-commit hooks temporarily (use with caution):

```bash
# Skip all hooks for this commit
git commit --no-verify
```

**SKIP environment variables**: Some hooks accept `SKIP` to bypass specific checks:

```bash
# Skip specific hooks (check hook IDs in .pre-commit-config.yaml)
SKIP=core,langchain pre-commit run --all-files
```

## Pre-Commit Hooks

The `.pre-commit-config.yaml` defines hooks that enforce:

- **Standard validation**: YAML/TOML syntax checking, proper file endings, no trailing whitespace
- **Text normalization**: Fix curly quotes and non-standard spaces
- **Per-package format and lint**: Each package in `libs/` (core, langchain, partners/*) runs `make format lint`
- **Version consistency checks**: Ensure `pyproject.toml` versions match source code for `langchain-core`, `langchain-v1`, and partner packages
- **Branch protection**: Prevents direct commits to the `master` branch

These hooks automatically prevent commits that fail linting or have formatting issues. They use the same Makefiles documented below.

## Testing Commands

### Run All Unit Tests

```bash
# From any package directory
make test

# Example: test langchain_v1
cd libs/langchain_v1 && make test
```

Unit tests live in `tests/unit_tests/` (no network calls allowed). The test target uses `pytest` with `xdist` for parallelization and socket restrictions to prevent accidental network calls. Tests are parallelized via the `-n auto` flag.

### Run a Specific Test File

```bash
# Using make
make test TEST_FILE=tests/unit_tests/agents/test_agent.py

# Or using uv directly
uv run --group test pytest tests/unit_tests/agents/test_agent.py
```

The `TEST_FILE` parameter accepts:
- A single file: `tests/unit_tests/agents/test_agent.py`
- A directory: `tests/unit_tests/agents/`
- A test by name: `tests/unit_tests/agents/test_agent.py::TestClass::test_method`

### Integration Tests

Integration tests live in `tests/integration_tests/` and require network access and API keys. Run them separately:

```bash
make integration_tests
```

Some packages (like `langchain_v1`) use Docker services (PostgreSQL, Redis) for integration tests:

```bash
cd libs/langchain_v1
make test  # starts services, runs tests, stops services
```

To run integration tests with specific SDK versions (useful for compatibility testing):

```bash
# Test with multiple versions of an SDK
cd libs/partners/anthropic
uv run --with 'anthropic>=0.120.0,<1.0.0' --group test --group test_integration pytest tests/integration_tests/
uv run --with 'anthropic>=1.0.0,<2.0.0' --group test --group test_integration pytest tests/integration_tests/
```

### Test in Watch Mode

Auto-re-run tests as you edit code:

```bash
make test_watch
```

This uses `pytest-watcher` (via the `ptw` command) and updates snapshots automatically. Useful for test-driven development.

### Extended Tests

Some packages include tests marked with `@pytest.mark.extended` that run slower or under different conditions:

```bash
make extended_tests
```

### Coverage Reports

Generate code coverage reports:

```bash
make coverage
```

This produces `xml` and term-missing reports, useful for understanding untested code paths.

For targeted coverage (e.g., agent middleware):

```bash
cd libs/langchain_v1
make coverage_agents  # generates HTML and term-missing coverage for agents
```

## Linting and Formatting

### Run Full Linting Suite

```bash
make lint
```

This runs three checks in order:
1. **Ruff check**: Linter for logical errors, naming conventions, and code smells
2. **Ruff format (diff)**: Format checking (does not modify files)
3. **Mypy**: Static type checking

### Format Code

```bash
make format
```

This applies `ruff format` and `ruff check --fix` to auto-fix formatting and logical issues (e.g., unsorted imports, unused variables).

### Ruff-Only Commands

Format and linter can be run separately for faster iteration:

```bash
# Check formatting without fixing
ruff check .
ruff format . --diff

# Fix formatting and lint issues
ruff check . --fix
ruff format .
```

Run from within a package directory or from the repo root. Ruff processes Python and Jupyter notebooks.

### Lint Changes Only

For faster iteration during development, lint only files changed in your branch:

```bash
make lint_diff
make format_diff
```

These targets run on `git diff` output against `master` branch.

### Type Checking

Full type checking with mypy:

```bash
mypy .
```

Or use the make target:

```bash
make type
```

This checks all Python files for type errors (e.g., incorrect argument types, missing type hints). Type checking can be slow for large packages; run it with:

```bash
# Type check specific file or directory
mypy libs/core/langchain_core/runnables.py

# Type check only tests (faster)
cd libs/langchain_v1
make lint_tests
```

## Testing a Single Package

To develop and test a single package in the monorepo:

```bash
# Example: work on langchain_v1 core agent system
cd libs/langchain_v1

# Install all dependencies for this package
uv sync --all-groups

# Run all unit tests
make test

# Test specific file or with custom pytest options
make test TEST_FILE=tests/unit_tests/agents/test_create_agent.py

# Format and lint
make format
make lint

# Type check
make type
```

Each package has its own Makefile with consistent targets. The monorepo's `/libs/Makefile` provides package-wide commands like regenerating lock files.

## Lock File Management

The `uv.lock` file in each package pins exact dependency versions for reproducible builds. Update locks when dependencies change:

```bash
# From a package directory
uv lock

# Or regenerate all package locks
cd libs && make lock

# Verify all locks are up-to-date (CI check)
cd libs && make check-lock
```

The Makefiles set `UV_FROZEN = true`, which prevents unexpected lock file changes during regular development. Use the commands above when intentionally updating dependencies.

## Make Commands Reference

All packages follow the same Makefile structure:

| Command | Purpose |
|---------|---------|
| `make test` | Run all unit tests (pytest) |
| `make test TEST_FILE=<path>` | Run tests in a specific file or directory |
| `make test_watch` | Run tests in watch mode (auto-rerun on changes) |
| `make integration_tests` | Run integration tests (requires API keys) |
| `make extended_tests` | Run only tests marked with `@pytest.mark.extended` |
| `make lint` | Run ruff check + ruff format --diff + mypy |
| `make lint_diff` | Lint only changed files (git diff against master) |
| `make format` | Apply ruff format and ruff check --fix |
| `make format_diff` | Format only changed files (git diff against master) |
| `make type` | Run mypy type checking |
| `make coverage` | Run tests and generate coverage report |
| `make help` | Display all available targets |

Package-specific commands (see Makefiles in each directory):

- `langchain_v1`: `make test_fast`, `make coverage_agents`, `make start_services`, `make stop_services`
- `core`: `make check_imports`, `make benchmark`
- `partners/*`: `make test TEST_FILE=tests/integration_tests/`, `make benchmark`

## Environment Variables

The Makefiles use a few environment variables to control behavior:

| Variable | Purpose | Default |
|----------|---------|---------|
| `UV_FROZEN` | Prevent lock file changes during `uv sync` | `true` in Makefiles |
| `TEST_FILE` | Path to test file or directory | `tests/unit_tests/` |
| `PYTEST_EXTRA` | Extra pytest options (e.g., `-vv`, `-x`) | (empty) |
| `LANGGRAPH_TEST_FAST` | Use in-memory services instead of Docker | `1` (fast) or `0` (full) |

Example: Run fast tests with extra pytest verbosity:

```bash
make test PYTEST_EXTRA="-vv" TEST_FILE=tests/unit_tests/agents
```

Example: Run full integration tests with Docker services:

```bash
cd libs/langchain_v1
LANGGRAPH_TEST_FAST=0 make test
```

## Common Workflows

### Before Committing

```bash
# 1. Format code
make format

# 2. Run linting and type checks
make lint

# 3. Run tests
make test

# 4. Commit (pre-commit hooks will run automatically)
git commit
```

Or let pre-commit do the format/lint:

```bash
make test
git add .
pre-commit run --all-files  # or just commit and let hooks run
git commit
```

### Iterative Development

For fast feedback during development:

```bash
# Terminal 1: Watch tests
make test_watch

# Terminal 2: Edit code and format
# Changes auto-trigger re-run in Terminal 1
make format
```

### Fixing Lint Errors

When `make lint` fails:

```bash
# 1. Auto-fix formatting and simple lint issues
make format

# 2. Check what remains
make lint

# 3. Fix remaining issues manually (mypy errors, etc.) and re-check
make lint
```

For faster iteration on specific files:

```bash
# Lint and format specific file
ruff check --fix langchain/agents/agent.py
ruff format langchain/agents/agent.py
mypy langchain/agents/agent.py
```

### Running Specific Tests

```bash
# All tests in a file
make test TEST_FILE=tests/unit_tests/agents/test_agent.py

# Single test class
uv run --group test pytest tests/unit_tests/agents/test_agent.py::TestAgentClass -v

# Single test method
uv run --group test pytest tests/unit_tests/agents/test_agent.py::TestAgentClass::test_method_name -v

# All tests matching a pattern
uv run --group test pytest -k "test_agent" tests/unit_tests/

# With extra verbosity and stopping on first failure
make test TEST_FILE=tests/unit_tests/agents PYTEST_EXTRA="-vv -x"
```

## Conventional Commits and PR Titles

Follow [Conventional Commits](https://www.conventionalcommits.org/) format with required scope:

```txt
<type>(<scope>): <description>
```

**Format rules:**
- `<type>`: One of `feat`, `fix`, `chore`, `docs`, `refactor`, `perf`, `test`
- `<scope>`: Required—one of `core`, `langchain`, partner name (e.g., `anthropic`, `openai`), `infra`, or `docs`
- `<description>`: Lowercase start (unless first word is proper noun like `Azure`, `GitHub`, or a code name)
- Named entities (class, function, parameter names) should be wrapped in backticks

**Examples:**
```txt
feat(langchain): add new chat completion feature
fix(core): resolve type hinting issue in vector store
chore(anthropic): update infrastructure dependencies
feat(langchain): `ls_agent_type` tag on `create_agent` calls
fix(openai): infer Azure chat profiles from model name
```

## Troubleshooting

### Lock file out of sync

```bash
# Regenerate lock
cd libs/<package>
uv lock

# Or check if lock is up-to-date
uv lock --check
```

### Dependencies not installed

```bash
# Ensure all groups are installed
uv sync --all-groups

# Or just the test group
uv sync --group test
```

### Tests fail with "no network" error

This is intentional—unit tests have socket restrictions. For integration tests:

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
```

### Pre-commit hook fails locally but passes in CI

Ensure you're using the same Python version and have all dependency groups installed:

```bash
python --version
uv sync --all-groups
pre-commit run --all-files
```

### Cache clearing issues

If you encounter stale cache issues:

```bash
# Clear mypy cache
rm -rf .mypy_cache

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
# Remove lock and resolve from scratch
cd libs/<package>
rm uv.lock
uv lock

# Or check for conflicts
uv sync --all-groups
```

## Related Documentation

- [Contributing Guide](repo://AGENTS.md): Detailed development conventions, PR templates, and code standards
- [System Architecture](repo:///openwiki/architecture.md): Three-layer design and module responsibilities
- [CI/CD Workflows](repo:///openwiki/ci-workflows.md): GitHub Actions automation and release process
- [Unit Tests](repo:///openwiki/unit-tests.md): Unit test structure, fixtures, and best practices
