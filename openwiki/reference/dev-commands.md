---
type: "Developer Tools Reference"
title: "Development Commands Reference"
description: "Complete reference for make targets, uv commands, and development utilities across the LangChain Python monorepo."
tags: [development, reference, make, uv, testing, linting, pre-commit]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-7c96a74af67942d40559bf7d
    resource: repo://libs/langchain_v1/Makefile
  - id: openwiki-source-f708a9db48bfcf1b154e4708
    resource: repo://libs/langchain/Makefile
  - id: openwiki-source-a6e669bb11f217c6fbd06670
    resource: repo://libs/partners/anthropic/Makefile
  - id: openwiki-source-2632c84304ed07e4f6683de6
    resource: repo://libs/text-splitters/Makefile
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

This is a comprehensive reference for development commands and tools used in the LangChain Python monorepo. The monorepo uses `uv` for package management, `make` for task automation, and standard Python tools (`ruff`, `mypy`, `pytest`) for code quality and testing.

**Related pages:** See [Development Commands and Local Setup](/openwiki/dev-commands.md) for operational guides and setup workflows.

## Monorepo Structure and Make Targets

The repository consists of multiple independently packaged modules, each with its own `Makefile` and dependency management:

### Core Packages

Located under `libs/`:

- **`libs/core/`** – `langchain-core` package with foundational abstractions
- **`libs/langchain/`** – `langchain-classic` package (legacy, no new features)
- **`libs/langchain_v1/`** – Actively maintained `langchain` package with agent infrastructure
- **`libs/text-splitters/`** – Document chunking and text splitting utilities
- **`libs/model-profiles/`** – Model configuration and profile management
- **`libs/partners/`** – Third-party service integrations (OpenAI, Anthropic, Ollama, etc.)
- **`libs/standard-tests/`** – Shared test suite for integration testing

### Root-Level Make Targets

The `libs/Makefile` and `libs/partners/Makefile` provide cross-package operations:

```bash
# libs/Makefile
make lock          # Regenerate lockfiles for all core packages
make check-lock    # Verify all lockfiles are up-to-date

# libs/partners/Makefile
make lock          # Regenerate lockfiles for all partner packages
make check-lock    # Verify all partner lockfiles are up-to-date
```

## Testing Targets

### Standard Test Targets

All packages implement these core test targets (invoked as `make -C libs/<package> <target>`):

```bash
make test              # Run unit tests with parallel execution and socket disabled
make tests             # Alias for 'make test'
make test TEST_FILE=<path>  # Run tests in a specific directory
make test_watch        # Run unit tests in watch mode with auto-reload
make integration_tests # Run integration tests (requires test_integration group)
make extended_tests    # Run only tests marked with @pytest.mark.extended
make coverage          # Generate coverage report with HTML and XML outputs
```

**Test Configuration:**

- Tests run with `pytest -n auto` for parallel execution
- Socket access is disabled (`--disable-socket`) except for Unix sockets
- Environment variables are cleared to prevent tracing/API leakage:
  - `LANGCHAIN_TRACING_V2`
  - `LANGCHAIN_API_KEY`
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING`
  - `LANGCHAIN_PROJECT`

### Package-Specific Test Variations

#### `libs/langchain_v1/`

Includes specialized test management for agent infrastructure with Docker services:

```bash
make start_services    # Start PostgreSQL and Redis containers for tests
make stop_services     # Stop and remove test containers
make test              # Run tests with all services started/stopped
make test_fast         # Run tests with in-memory services only (LANGGRAPH_TEST_FAST=1)
make coverage_agents   # Run middleware and agent tests with coverage
make test_watch        # Run tests in watch mode with services
```

#### `libs/partners/anthropic/`

Includes multi-version SDK testing:

```bash
make integration_tests # Test against both anthropic 0.x and 1.x SDK versions
```

The target uses `uv run --with` to temporarily pin specific SDK versions for smoke testing:
- Anthropic 0.x (backed by `httpx`)
- Anthropic 1.x (backed by `httpx2`)

#### `libs/model-profiles/`

Includes profile refresh capability:

```bash
make refresh-profiles  # Refresh model profiles from models.dev for all partners
                       # Requires network access; UV_FROZEN=false override
```

### Test File Parameters

Most packages support custom test file targets:

```bash
make test              # Runs tests/unit_tests/ (default)
make test TEST_FILE=tests/integration_tests/  # Custom directory
make test PYTEST_EXTRA="-vvv"  # Additional pytest flags
```

### Special Test Flags

Packages define `TEST_FILE` and `PYTEST_EXTRA` variables:

- `TEST_FILE` – Target directory (defaults to `tests/unit_tests/`)
- `PYTEST_EXTRA` – Additional pytest arguments (e.g., `-vvv`, `-k pattern`)

## Linting and Formatting Targets

### Common Lint/Format Targets

All packages implement these targets:

```bash
make lint              # Run full linting (ruff check + mypy)
make lint_diff         # Lint only files changed since master
make lint_package      # Lint only the main package (not tests)
make lint_tests        # Lint only test files

make format            # Format code with ruff (modifies files)
make format_diff       # Format only changed files since master
```

### Type Checking

```bash
make type              # Run mypy static type checking on PYTHON_FILES
```

Type checking includes:
- Main package code
- Test code (when lint_tests is run)
- Cache directory: `.mypy_cache` or `.mypy_cache_test`

### Linting Details

The `lint` target runs three checks in sequence:

1. **Import checking** – Custom shell script (`./scripts/lint_imports.sh`) validates import organization
2. **Ruff linting** – Fast Python linter checking code style and common errors
3. **Type checking** – mypy static type checking with cache management

When running `lint` or `format`, `PYTHON_FILES` is set to:

- `.` (entire directory) for the base `lint` and `format` targets
- Only changed files (since master) for `lint_diff` and `format_diff`
- Package directory (e.g., `langchain_core`) for `lint_package`
- `tests/` or `tests/unit_tests` for `lint_tests`

### Format Command

The `format` target applies fixes:

```bash
make format            # Format with ruff, then apply --fix for lint issues
```

This runs:
1. `ruff format` (applies code formatting)
2. `ruff check --fix` (applies auto-fixable lint issues)

**Note:** `ruff format --diff` shows what would change without modifying files.

## uv Command Integration

All make targets use `uv run` to execute commands within package-managed virtual environments. Key patterns:

### Running Commands with Groups

```bash
# Run with specific dependency groups
uv run --group test pytest ...
uv run --group lint ruff ...
uv run --all-groups mypy ...
```

**Group Names:**
- `test` – Testing dependencies (pytest, pytest-xdist, etc.)
- `lint` – Linting tools (ruff)
- `typing` – Type checking (mypy and stubs)
- `dev` – Development utilities (jupyter, etc.)
- `test_integration` – Integration testing dependencies

### Temporary Dependencies

Partner packages sometimes test with temporary SDK versions:

```bash
# Test with anthropic 0.x in addition to default version
uv run --with 'anthropic>=0.120.0,<1.0.0' --group test pytest ...
```

### Lock File Management

```bash
# From within a package directory
uv lock              # Regenerate uv.lock (respects pyproject.toml)
uv lock --check      # Verify lock file matches pyproject.toml
```

The `UV_FROZEN = true` environment variable is exported in all package Makefiles to prevent automatic lock file updates during normal development.

## Pre-Commit Hooks

The repository uses pre-commit to enforce code quality on every commit. Hooks are defined in `.pre-commit-config.yaml`.

### Hook Categories

**Global validation hooks:**
- `no-commit-to-branch` – Prevent direct commits to the `master` branch
- `check-yaml` – Validate YAML file syntax (allows custom tags with `--unsafe`)
- `check-toml` – Validate TOML file syntax
- `end-of-file-fixer` – Ensure files end with a newline
- `trailing-whitespace` – Remove trailing whitespace (excludes `.ambr` files)
- `fix-smartquotes` – Replace curly quotes with straight quotes
- `fix-spaces` – Normalize non-standard whitespace

**Per-package hooks:**
- Each major package (core, langchain, langchain_v1, text-splitters, standard-tests, and partner packages like anthropic, chroma, etc.) has a local pre-commit hook that runs `make format lint` on changed files

### Running Pre-Commit Manually

```bash
# Install hooks (required once per development environment)
pre-commit install

# Run all hooks on staged files (automatic before commit)
# Hooks run automatically; no manual action needed

# Run hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files

# Skip pre-commit for a single commit (use with caution)
git commit --no-verify
```

## Common Command Patterns

### Setting Up a Package for Development

```bash
cd libs/langchain_v1
uv sync --all-groups      # Install all dependency groups
pre-commit install        # Configure pre-commit hooks
make test                 # Verify setup with a test run
```

### Running Tests with Custom Options

```bash
# Run a specific test file
make test TEST_FILE=tests/unit_tests/agents/test_executor.py

# Run tests matching a pattern
make test PYTEST_EXTRA="-k agent"

# Run with verbose output
make test PYTEST_EXTRA="-vvv"

# Run with extra arguments
make test PYTEST_EXTRA="--tb=short --maxfail=3"
```

### Linting and Formatting Workflow

```bash
# Check what would be formatted
make format_diff

# Format all files
make format

# Run full linting (includes type checking)
make lint

# Quick check of changed files
make lint_diff

# Lint only the main package (not tests)
make lint_package
```

### Type Checking Workflow

```bash
# Type check the entire package
make type

# Type check only tests
make lint_tests  # includes type checking
```

### Debugging Failed Tests

```bash
# Run a single failing test with full output
make test TEST_FILE=tests/unit_tests/agents/test_executor.py PYTEST_EXTRA="-vvv --tb=short"

# Run with pytest debugging (drop into pdb on failure)
make test PYTEST_EXTRA="--pdb"

# Run with profiling
make test_profile  # libs/core and libs/text-splitters only
```

### Coverage Reports

```bash
cd libs/langchain_v1
make coverage              # Generate coverage.xml and HTML report
# HTML report: htmlcov/index.html
```

## Environment Variables and Configuration

### Build Environment

All package Makefiles export `UV_FROZEN = true` to prevent automatic lock file updates during development. This ensures reproducible builds.

### Test Environment Isolation

The `test` target explicitly clears environment variables to prevent external service pollution:

```makefile
env \
  -u LANGCHAIN_TRACING_V2 \
  -u LANGCHAIN_API_KEY \
  -u LANGSMITH_API_KEY \
  -u LANGSMITH_TRACING \
  -u LANGCHAIN_PROJECT \
  uv run --group test pytest ...
```

### Special Variables

**`TEST_FILE`** – Default: `tests/unit_tests/`
- Overridden for `integration_tests` target to `tests/integration_tests/`
- User-overridable: `make test TEST_FILE=tests/my_tests/`

**`PYTEST_EXTRA`** – Default: empty
- Additional arguments passed to pytest: `make test PYTEST_EXTRA="-k agent"`

**`LANGGRAPH_TEST_FAST`** – Set by `test_fast` in `libs/langchain_v1/`
- Controls whether tests use in-memory or full services

**`UV_FROZEN`** – Set to `true` in all package Makefiles
- Prevents automatic lock file updates; use `uv lock` explicitly to update

## Package-Specific Notes

### `libs/core/` (langchain-core)

The foundational package with base abstractions. Includes:
- `check_imports` target – Validates import consistency
- `check_version` target – Confirms version consistency
- `benchmark` target – Runs performance benchmarks

### `libs/langchain_v1/` (langchain)

The main implementation package with agents and advanced features. Notable differences:
- Requires Docker services for full test suite (`make start_services`)
- `test_fast` target for rapid feedback without services
- Agent-specific coverage reporting

### `libs/partners/*` (Integration Packages)

Each partner integration (Anthropic, OpenAI, Ollama, etc.) includes:
- Standard test/lint/format targets
- Integration tests via `integration_tests` target
- `check_version` target for version consistency

Anthropic package specifically tests multiple SDK major versions.

### `libs/model-profiles/`

Profile management package with:
- `refresh-profiles` target – Updates model profiles from models.dev
- Integration test discovery from `libs/partners/`

### `libs/standard-tests/`

Shared test suite for integration validation. Supports:
- Standard test/lint targets
- Used by partner packages to validate integration behavior

## Troubleshooting

### Lock File Issues

```bash
# Check if lock file is out of date
make check-lock

# Regenerate lock file
make lock

# Verify all lockfiles
make check-lock
```

### Test Socket Errors

Tests run with `--disable-socket` by default. If a test needs network access:
- Add to integration tests (`tests/integration_tests/`) which don't disable sockets
- Use `--allow-unix-socket` for Unix domain socket access (automatically allowed)

### Pre-Commit Hook Failures

If pre-commit hooks fail:
1. Hooks run `make format lint` automatically
2. Format failures are auto-fixed; review and stage changes
3. Lint failures may require manual fixes
4. Use `git commit --no-verify` only as a last resort (not recommended)

### Virtual Environment Issues

```bash
# Clean and re-sync dependencies
rm -rf .venv
uv sync --all-groups
```

### Type Checking Cache

```bash
# Clear mypy cache if type checking is inconsistent
rm -rf .mypy_cache .mypy_cache_test
make type
```

## Architecture and Make Integration

The make-based command system provides:

1. **Consistency** – Same targets across all packages (test, lint, format, etc.)
2. **Convenience** – Single `make` invocation vs. multiple `uv` commands
3. **Environment management** – Automatic virtual environment handling via `uv run`
4. **Integration** – Pre-commit hooks invoke `make format lint` for committed files
5. **Isolation** – Clear environment variables during testing to prevent pollution

Each package's `Makefile` defines targets specific to its structure, but follows common patterns for discoverability and usability.

## See Also

- [Development Commands and Local Setup](/openwiki/dev-commands.md) – Operational guide for setup and workflows
- [AGENTS.md](repo://AGENTS.md) – Project architecture and development context
- [Pre-commit configuration](repo://.pre-commit-config.yaml) – Global hook definitions
- Individual package Makefiles – `libs/<package>/Makefile` for package-specific targets
