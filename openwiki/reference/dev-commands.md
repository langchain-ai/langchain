---
type: "Reference"
title: "Development Commands Reference"
description: "Complete reference for make targets and uv package management commands used in the LangChain Python monorepo for testing, linting, formatting, and type checking."
tags: [development, commands, make, uv, testing, linting, reference]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Make Targets Reference

All packages in `libs/` include a `Makefile` that defines development tasks. These targets are run through `uv run` within virtual environments managed automatically by uv. The core `Makefile` is located at `repo://libs/core/Makefile`.

### Testing Targets

#### `make test` / `make tests`

Runs unit tests using pytest with automatic parallelization and socket restrictions enabled. Environment variables for tracing are unset to prevent interference.

**Command:**
```bash
make test
make tests  # alias
```

**Behavior:**
- Disables network sockets except Unix sockets (prevents accidental external connections)
- Runs tests in parallel with `pytest -n auto` (requires pytest-xdist)
- Disables benchmark mode with `--benchmark-disable`
- Unsets tracing environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGCHAIN_PROJECT`)
- Default test path: `tests/unit_tests/`

**Custom test file:**
```bash
make test TEST_FILE=tests/unit_tests/test_specific.py
```

**Additional pytest options:**
```bash
make test PYTEST_EXTRA="-vv --pdb"
```

**Environment variable set in Makefile:**
- `UV_FROZEN=true` — Prevents unintended changes to lock files

#### `make test_watch`

Runs unit tests in watch mode using pytest-watch (`ptw`). Re-runs tests whenever files change.

**Command:**
```bash
make test_watch
```

**Behavior:**
- Watches for file changes and automatically re-runs affected tests
- Enables snapshot updates with `--snapshot-update`
- Runs with `--now` (starts immediately without waiting for file changes)
- Includes `-vv` (very verbose output)
- Socket restrictions enabled
- Default path: current directory `.`

#### `make test_profile`

Generates performance profile information as SVG files for analyzing test execution performance.

**Command:**
```bash
make test_profile
```

**Behavior:**
- Runs pytest on `tests/unit_tests/` with profiling enabled
- Generates `.svg` files showing call stacks and timing
- Useful for identifying performance bottlenecks in tests

#### `make extended_tests`

Runs only tests marked with `@pytest.mark.extended_tests` or tests requiring special configuration flags.

**Command:**
```bash
make extended_tests
```

**Behavior:**
- Uses `--only-extended` flag to filter to extended test markers
- Socket restrictions enabled
- Allows customizing test path with `TEST_FILE` parameter

### Code Quality Targets

#### `make format`

Runs code formatters on the entire package to ensure consistent style.

**Command:**
```bash
make format
```

**Behavior:**
- Runs ruff formatter: `ruff format .`
- Runs ruff fixer (auto-fixes lint violations): `ruff check --fix .`
- Formats all Python and Jupyter notebook files

**Format only changed files (compared to master):**
```bash
make format_diff
```

#### `make lint`

Runs all linting checks: import validation, ruff linter, format checker, and type checking.

**Command:**
```bash
make lint
```

**Behavior:**
- Runs `./scripts/lint_imports.sh` to validate import structure
- Runs `ruff check .` for linting violations
- Runs `ruff format . --diff` to check formatting without modifying files
- Runs `mypy` for type checking with cache
- Creates `.mypy_cache` directory for incremental type checking

**Lint only the package code (not tests):**
```bash
make lint_package
```

**Lint only tests:**
```bash
make lint_tests
```

**Lint only changed files (compared to master):**
```bash
make lint_diff
```

**Note:** `lint_package` and `lint_tests` use a more restrictive `--group lint` dependency set instead of `--all-groups`.

#### `make type`

Runs mypy type checking on all Python files.

**Command:**
```bash
make type
```

**Behavior:**
- Runs mypy type checker with incremental cache
- Creates/uses `.mypy_cache` directory
- Respects mypy configuration in `pyproject.toml` or `setup.cfg`

#### `make check_version`

Validates version consistency in the package.

**Command:**
```bash
make check_version
```

**Behavior:**
- Runs `./scripts/check_version.py`
- Typically verifies that `__version__` in code matches package metadata version

#### `make check_imports`

Validates that imports are properly structured and not creating circular dependencies.

**Command:**
```bash
make check_imports
```

**Behavior:**
- Runs `./scripts/check_imports.py` on all `.py` files in `langchain_core` (or equivalent package directory)
- Detects circular imports and other import issues

#### `make benchmark`

Runs benchmark tests using codspeed for performance profiling.

**Command:**
```bash
make benchmark
```

**Behavior:**
- Runs `pytest tests/benchmarks --codspeed`
- Generates performance comparison data
- Useful for catching performance regressions

### Help and Metadata

#### `make help`

Displays all available make targets with brief descriptions.

**Command:**
```bash
make help
```

**Output example:**
```
----
format                       - run code formatters
lint                         - run linters
type                         - run type checking
check_version                - validate version consistency
test                         - run unit tests
tests                        - run unit tests
test TEST_FILE=<test_file>   - run all tests in file
test_watch                   - run unit tests in watch mode
```

#### `make all`

Default target (runs when no target specified). Currently runs `make help`.

**Command:**
```bash
make
# or
make all
```

## Multi-Package Targets (libs/Makefile)

The root `libs/Makefile` at `repo://libs/Makefile` provides cross-package commands that operate on core packages: `core`, `text-splitters`, `langchain`, `langchain_v1`, `model-profiles`.

### `make lock`

Regenerates lock files for all core packages when dependencies change.

**Command (from libs/ directory):**
```bash
cd libs
make lock
```

**Behavior:**
- Iterates over each core package directory
- Runs `uv lock` in each package to regenerate `uv.lock`
- Used after modifying `pyproject.toml` dependencies

### `make check-lock`

Verifies that all core package lock files are up-to-date with their `pyproject.toml`.

**Command (from libs/ directory):**
```bash
cd libs
make check-lock
```

**Behavior:**
- Runs `uv lock --check` in each core package
- Fails if any lock file is out-of-sync
- Used in CI to ensure lock files are committed when dependencies change

## UV Package Management Commands

`uv` is a fast Python dependency resolver and installer. It replaces pip, poetry, and poetry-like workflows in this monorepo. Key principles:

- **Automatic environment management**: `uv run` and `uv sync` manage virtualenvs automatically
- **Group-based dependency organization**: Dependencies are organized into groups (lint, test, typing, dev) in `pyproject.toml`
- **Frozen lock files**: The `UV_FROZEN=true` environment variable prevents lock file changes during builds/tests

### Core uv Commands

#### `uv sync [OPTIONS]`

Synchronizes the local virtual environment with the lock file and `pyproject.toml`.

**Command:**
```bash
# Sync all dependency groups
uv sync --all-groups

# Sync specific groups
uv sync --group test
uv sync --group lint
uv sync --group typing

# Sync only main dependencies (no dev/test groups)
uv sync
```

**Common groups in LangChain packages:**
- `lint` — Ruff linter, formatter, and related tools
- `typing` — Mypy type checker and stub packages
- `test` — Pytest and testing utilities (fixtures, markers, plugins)
- `test_integration` — Integration test dependencies (API clients, mock servers)
- `dev` — Development tools (Jupyter, IPython, setuptools)

#### `uv lock [OPTIONS]`

Creates or updates the lock file (`uv.lock`) based on `pyproject.toml`.

**Command:**
```bash
# Generate/update lock file
uv lock

# Check if lock file is up-to-date (fails if not)
uv lock --check
```

**When to use:**
- After modifying `pyproject.toml` dependencies
- In CI to verify lock files are current

#### `uv run [OPTIONS] <COMMAND>`

Runs a command in the package's virtual environment without explicitly activating it.

**Command:**
```bash
# Run a single command
uv run python --version

# Run pytest with specific test group
uv run --group test pytest tests/unit_tests/

# Run with multiple groups
uv run --all-groups mypy .

# Run with temporary extra dependency
uv run --with 'anthropic>=1.0.0' --group test pytest tests/integration/
```

**Common patterns in Makefiles:**
```bash
uv run --group test pytest ...        # Test with test dependencies
uv run --all-groups ruff check .      # Linting with all tools
uv run --group lint ruff format .     # Formatting with lint group
```

**Environment variable control:**
- `UV_FROZEN=true` — Fail if lock file would be modified (used in CI/tests)
- `UV_OFFLINE=true` — Fail if network access needed (requires pre-synced env)

### Advanced uv Options

#### Adding Temporary Dependencies

When you need a dependency for a single run (e.g., testing an integration):

```bash
# Run with temporary extra dependency
uv run --with 'anthropic>=1.0.0' python -c "import anthropic; ..."

# Multiple temporary dependencies
uv run --with 'anthropic>=1.0.0' --with 'openai>=1.0.0' pytest tests/integration/
```

#### Workspace and Per-Package Isolation

Each package in `libs/` maintains its own `pyproject.toml`, `uv.lock`, and virtual environment:

```bash
# Work in specific package
cd libs/core
uv sync --all-groups       # Sets up environment for core only
uv run pytest tests/

cd ../langchain
uv sync --all-groups       # Separate environment for langchain
uv run pytest tests/
```

## Pre-Commit Hooks

Pre-commit hooks are defined in `repo://.pre-commit-config.yaml` and run automatically before commits. Install once per repository:

```bash
pre-commit install
```

### Hook Categories

**Standard pre-commit hooks** (from pre-commit/pre-commit-hooks):
- `no-commit-to-branch` — Prevents direct commits to protected branches (master)
- `check-yaml` — Validates YAML syntax
- `check-toml` — Validates TOML syntax
- `end-of-file-fixer` — Ensures files end with a newline
- `trailing-whitespace` — Removes trailing whitespace

**Text normalization** (from sirosen/texthooks):
- `fix-smartquotes` — Replaces curly quotes with straight quotes
- `fix-spaces` — Normalizes spaces (removes non-breaking spaces, etc.)

**Per-package format and lint hooks** (local):
For each package (`core`, `langchain`, `standard-tests`, `text-splitters`, partners):
- Runs `make -C <package_dir> format lint`
- Triggered for files in that package directory

**Manual pre-commit execution:**
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run core --all-files

# Update hook definitions
pre-commit autoupdate
```

## Environment Variables in Development

### Tracing Environment Variables (Unset During Tests)

These variables are explicitly unset when running tests to prevent LangSmith tracing interference:

```
LANGCHAIN_TRACING_V2      # Enable LangSmith tracing
LANGCHAIN_API_KEY          # LangSmith API key
LANGSMITH_API_KEY          # Alternative API key name
LANGSMITH_TRACING          # Enable tracing flag
LANGCHAIN_PROJECT          # Project name for tracing
```

### UV Control Variables

```
UV_FROZEN=true             # Fail if lock file would be modified (prevents accidental changes)
UV_OFFLINE=true            # Fail if network access needed (requires pre-synced env)
UV_CACHE_DIR=<path>        # Custom cache directory for package index
UV_NO_SYNC=true            # Skip automatic sync before running
```

## Common Development Workflows

### Setup New Package

```bash
cd libs/<package_name>
uv sync --all-groups
pre-commit install
```

### Run Tests

```bash
# Run all unit tests
make test

# Run specific test file
make test TEST_FILE=tests/unit_tests/test_specific.py

# Run tests in watch mode (re-run on file changes)
make test_watch

# Run with extra pytest options
make test PYTEST_EXTRA="-vv -k test_name"
```

### Check Code Quality Before Commit

```bash
# Run all checks locally
make lint
make type
make format

# Or let pre-commit hooks run automatically
git commit -m "..."  # pre-commit hooks run automatically
```

### Update Lock Files After Modifying pyproject.toml

```bash
cd libs/<package_name>
uv lock
cd ..
make check-lock  # Verify all packages' locks are current
```

### Run Integration Tests

```bash
# Sync with integration test dependencies
cd libs/<package_name>
uv sync --group test_integration

# Run integration tests
uv run --group test_integration pytest tests/integration_tests/
```

## Troubleshooting

### "Lock file is out-of-date"

The `pyproject.toml` was modified but `uv.lock` wasn't regenerated:

```bash
cd libs/<package_name>
uv lock
git add uv.lock pyproject.toml
git commit
```

### "Virtual environment not found or corrupted"

Remove and resync:

```bash
cd libs/<package_name>
rm -rf .venv
uv sync --all-groups
```

### Pre-commit hooks fail with permission errors

Pre-commit needs to invoke `make` targets. Ensure `make` is installed and accessible:

```bash
which make  # Verify make is installed
pre-commit run --all-files  # Try again
```

### Tests fail with "socket not allowed"

The test runner has socket restrictions enabled (`--disable-socket`). To allow specific sockets:

```bash
# Allow Unix sockets (already enabled by default in Makefiles)
# For custom socket needs, modify PYTEST_EXTRA in test invocation
make test PYTEST_EXTRA="--allow-unix-socket --allow-tcp-localhost"
```

### Mypy cache corruption

Clear the mypy cache:

```bash
cd libs/<package_name>
rm -rf .mypy_cache .mypy_cache_test
make type
```
