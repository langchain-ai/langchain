---
type: "Reference"
title: "CI/CD Workflows: GitHub Actions and Release Process"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-34e57b5a3a0c875639ab72a7
    resource: repo://.github/scripts/check_diff.py
  - id: openwiki-source-f35e7c44cc1805709393a581
    resource: repo://.github/workflows/_lint.yml
  - id: openwiki-source-c92cc62695c6def991956428
    resource: repo://.github/workflows/_release.yml
  - id: openwiki-source-c9c292f4ecabe180cdae27ce
    resource: repo://.github/workflows/_test_pydantic.yml
  - id: openwiki-source-d8a8900818f4abab719bd1b7
    resource: repo://.github/workflows/_test_vcr.yml
  - id: openwiki-source-4d9cccca7700db7220ec055e
    resource: repo://.github/workflows/_test.yml
  - id: openwiki-source-7330cb37457ccdb62d7c41c7
    resource: repo://.github/workflows/auto-label-by-package.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-6d4b4e707b8d60b6ccfa3425
    resource: repo://.github/workflows/openwiki-update.yml
  - id: openwiki-source-f8781d847f6481a966a44a68
    resource: repo://.github/workflows/pr_labeler.yml
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


# CI/CD Workflows: GitHub Actions and Release Process

LangChain employs a sophisticated CI/CD system built on GitHub Actions that automates testing, linting, quality checks, and release management across a monorepo structure. The system emphasizes efficiency through intelligent change detection, parallel matrix testing, and strict release gates.

## Architecture Overview

The CI/CD system consists of three layers:

1. **Pull request / push CI (`check_diffs.yml`)**: Detects changed packages and runs targeted tests, linting, and compatibility checks
2. **Scheduled integration testing (`integration_tests.yml`)**: Daily remote API testing with live credentials against partner libraries
3. **Manual release workflow (`_release.yml`)**: Comprehensive pre-release validation, PyPI publishing, and dependent package testing

## Primary CI Workflow (Pull Requests & Master Pushes)

The main entry point is `.github/workflows/check_diffs.yml`, which runs on every pull request, push to master, and merge group event.

### Change Detection & Matrix Generation

The workflow begins with a change detection phase:

1. A Python script (`.github/scripts/check_diff.py`) analyzes which files changed
2. Maps changes to package directories (`libs/core`, `libs/partners/*`, etc.)
3. Builds a dependency graph to include dependent packages when core components change
4. Generates separate test matrices for linting, unit tests, Pydantic compatibility tests, integration test compilation, VCR cassette tests, and extended test suites
5. Outputs are passed as JSON to downstream jobs via matrix strategy

This detection ensures only affected packages are tested, optimizing CI runtime.

### Linting Pipeline (`_lint.yml`)

Runs on affected packages with Python 3.11 (configurable):

- **Ruff analysis**: Code style, import sorting, and rule enforcement with inline GitHub annotations
- **MyPy type checking**: Static type verification
- **Markdown linting**: Documentation quality checks (via `.markdownlint.json`)

Tools are sourced from dependency groups: `lint` and `typing`. The workflow installs both package code and test code dependencies, running `make lint_package` and `make lint_tests` targets.

### Unit Testing (`_test.yml`)

Runs matrix tests across Python versions with dependency constraint verification:

**Matrix dimensions**:
- Python 3.10 through 3.14 (per-package configuration)
- Current locked dependencies (from `uv.lock`)
- Minimum supported dependency versions

**Two-phase testing**:

1. **Current dependencies**: Runs full test suite against versions in `uv.lock`
2. **Minimum dependencies**: Calculates minimum versions from `pyproject.toml` constraints, downgrades via pip, and reruns tests to ensure compatibility

The workflow uses `make test PYTEST_EXTRA=-q` and `make tests PYTEST_EXTRA=-q` targets, and verifies the working directory remains clean (no untracked generated files).

### Pydantic Compatibility Testing (`_test_pydantic.yml`)

Tests affected packages against multiple Pydantic versions (e.g., v1 and v2 compatibility):

- Triggered when Pydantic version constraints or dependent code changes
- Configurable per-package via `pyproject.toml`
- Runs matrix over specified Pydantic versions

### VCR Cassette Tests (`_test_vcr.yml`)

Validates integration tests backed by recorded HTTP cassettes:

- Runs in playback-only mode (no API credentials required)
- Detects stale cassettes from test input changes without re-recording
- Enables fast, repeatable integration test feedback

Only triggered for packages with VCR cassettes (currently `libs/partners/openai`).

### Integration Test Compilation (`_compile_integration_test.yml`)

Performs shallow integration test validation:

- Compiles test modules without executing them
- Catches import errors and obvious syntax issues
- Provides quick feedback loop without running expensive external API calls

### Extended Test Suites

For packages defining `extended_testing_deps.txt`, runs additional tests:

- Installs extra dependencies beyond standard test group
- Executes `make extended_tests` target
- Allows performance benchmarks, stress tests, or heavy-weight validations

### Release Option Validation

The workflow includes a `check-release-options` job:

- Verifies `.github/workflows/_release.yml` dropdown options stay synchronized with actual package directories
- Prevents stale release options from blocking valid releases

## Release Workflow (`_release.yml`)

The release workflow is manually triggered via GitHub Actions UI (or can be called as a reusable workflow). It handles versioning, building, testing, and publishing to PyPI.

### Release Modes & Invocation

**Manual dispatch** (`workflow_dispatch`):
- Dropdown selection of package to release (core, langchain, langchain_v1, text-splitters, standard-tests, model-profiles, or partner packages)
- Manual version entry (default `0.1.0`)
- Optional override to full path (e.g., `libs/partners/partner-xyz`)
- Dangerous flags: `dangerous-nonmaster-release`, `allow-prereleases`, `skip-prior-published-package-checks`

**Reusable workflow** (`workflow_call`):
- Accepts `working-directory`, `release-version`, and safety bypass flags
- Used internally for multi-package release orchestration

### Release Gate: Build & Version Check

**Job: `build`** (isolated permissions for security):

1. **Version verification**: Checks `pyproject.toml` version against input, fails if mismatch
2. **PyPI availability check**: Queries PyPI to ensure version not already published (PEP 440 normalization applied)
3. **Build**: Runs `uv build` to create wheel and sdist distributions
4. **Artifact upload**: Stores `dist/` directory for downstream jobs

Security rationale: Separates build (no credentials) from publishing (trusted publishing token) to prevent compromised dependencies from accessing PyPI credentials.

### Release Notes Generation

**Job: `release-notes`**:

1. **Tag detection**: Finds previous release tag via git history
   - For pre-releases: Matches base version; falls back to latest release
   - For stable releases: Searches for previous patch version; falls back to latest
2. **Changelog extraction**: Runs `git log --format="%s" <prev-tag>..HEAD -- <working-dir>` to collect commit messages
3. **First release handling**: Explicitly marks initial releases, uses full commit history

### Pre-Release Checks

**Job: `pre-release-checks`** (no caching to catch missing dependencies):

1. **Direct wheel installation**: Installs built wheel directly (validates metadata)
2. **Package import test**: Verifies main module imports successfully
3. **Unit tests**: Runs full `make tests` against the wheel
4. **Minimum version testing**: Recalculates and tests minimum dependencies (skips serdes tests for speed)
5. **Prerelease dependency detection**: Fails if any dependencies use prerelease constraints (unless release itself is prerelease)
6. **Integration tests**: For partner packages only, runs `make integration_tests` with live API credentials

### PyPI Publishing

**Job: `test-pypi-publish`** (TestPyPI):
- Uses GitHub OpenID Connect (trusted publishing)
- Publishes to test.pypi.org for staging validation
- Tolerates duplicate versions (CI safety only)

**Job: `publish`** (Production PyPI):
- Uses trusted publishing to production PyPI
- Only runs if all prior checks pass
- Creates GitHub Release with generated release notes

### Compatibility Testing

**Job: `test-prior-published-packages-against-new-core`**:
- Only runs for `libs/core` releases
- Tests previously-published partner packages (e.g., langchain-openai, langchain-anthropic) against new core
- Fetches latest partner tag from git, installs new core wheel, runs tests
- Can skip per-partner via `skip-prior-published-package-checks` input

**Job: `test-dependents`**:
- Only runs for `libs/core` or `libs/langchain_v1` releases
- Checks external dependent packages (e.g., deepagents)
- Tests Python 3.11 and 3.13
- Ensures breaking changes are caught before publish

## Integration Testing (`integration_tests.yml`)

Scheduled daily (1 PM UTC) with manual dispatch override capability.

### Test Matrix Generation

**Job: `compute-matrix`**:

- **Default scope**: Tests 9 partner libraries (OpenAI, Anthropic, Fireworks, Groq, MistralAI, XAI, Google VertexAI, Google GenAI, AWS)
- **Python versions**: 3.10 and 3.14 by default; overridable via input
- **Selective testing**: Can select single library, exclude libraries, or override Python versions
- **Scope security**: Only runs on main repository; manual dispatch allowed from forks

### Integration Test Execution

**Job: `integration-tests`**:

- Checks out primary monorepo plus external google-genai, google-vertexai, and langchain-aws repositories
- Reorganizes external repos into local partner directories for unified testing
- Authenticates to Google Cloud and AWS
- Runs per-package `make integration_tests` with all live API credentials injected
- Uses concurrency locks per (package, python-version) to serialize same-package runs and prevent credential conflicts

**Credentials**: Receives 30+ environment variables covering OpenAI, Anthropic, Google, AWS, Azure, Groq, MistralAI, HuggingFace, and more.

## Auto-Labeling Workflows

### Issue Auto-Labeling (`auto-label-by-package.yml`)

Fires when issues are opened or edited:

1. Parses issue body for `## Package` section
2. Maps package name (e.g., "langchain-openai") to label (e.g., "openai")
3. Adds/removes labels to match selected package(s)
4. Supports both dropdown (single) and checkbox (multi-select) formats

### PR Labeling (`pr_labeler.yml`)

Unified PR labeler applying size, file-based, title-based, and contributor classification:

- File-based labels: Maps changed file paths to package labels
- Size labels: Computes PR size (small, medium, large) from diff statistics
- Title-based labels: Detects certain patterns in PR title
- Contributor classification: Checks org membership to tag external contributions
- Uses GitHub App for organization membership verification

Consolidates multiple prior workflows into single sequential run to eliminate race conditions.

## OpenWiki Auto-Update (`openwiki-update.yml`)

Runs on schedule (8 AM UTC daily) or manual dispatch:

1. Checks out full repository history (required for diff-against-HEAD)
2. Installs Node.js and OpenWiki CLI
3. Runs `openwiki code --update --print` to regenerate documentation
4. Removes transient state file
5. Creates/updates pull request with changes
6. Preserves partial progress on failure for baseline establishment

Uses LangSmith tracing for observability.

## Dependency Pinning & Version Management

### Frozen Dependency Locks

All CI jobs set `UV_FROZEN=true` and `UV_NO_SYNC=true` (when applicable):

- Ensures reproducible builds against locked versions in `uv.lock`
- Prevents transitive dependency surprises in CI
- Each job explicitly pins Python version and dependency revisions

### Minimum Version Testing

The `get_min_versions.py` script extracts version constraints from `pyproject.toml` and queries PyPI for minimum published versions satisfying those constraints.

Example: If constraint is `langchain-core>=0.3.0,<1.0`, the script finds and installs the earliest 0.3.* release.

Two modes:
- `pull_request`: Tests against minimum with some leniency (used in PR CI)
- `release`: Stricter testing with prerelease rejection (used in release validation)

## Release Policy

### Semantic Versioning

**Core** (`libs/core`) follows strict semantic versioning:
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

**Partner packages** and other libraries align with core releases:
- LangChain follows core versioning for tight integration
- Partners maintain independent versioning but coordinate with core releases

### Release Branching

- Releases only proceed from `master` branch (default) or explicitly via `dangerous-nonmaster-release` flag (hotfixes only)
- Version must match `pyproject.toml` or operator provides override
- PyPI availability double-checked to prevent accidental re-publishes

### Pre-Release Support

- Supports alpha/beta/rc versions (e.g., `0.1.0-rc1`, `0.1.0a1`)
- Pre-release detection normalizes hyphen/underscore variants per PEP 440
- Optional `allow-prereleases` flag permits transitive prerelease dependencies during alpha cycles
- Final releases block prerelease dependencies unless explicitly allowed

## Configuration & Operations

### Environment Variables

**Frozen dependency control**:
- `UV_FROZEN`: Prevents automatic dependency resolution
- `UV_NO_SYNC`: Skips uv sync in build steps (manual sync used instead)

**Linting & formatting**:
- `RUFF_OUTPUT_FORMAT: github`: Inline GitHub annotations for linter violations

**LangSmith tracing** (optional):
- `LANGSMITH_API_KEY`: Optional tracing of CI workflows themselves
- `LANGCHAIN_TRACING_V2: true`: Enable tracing
- `LANGCHAIN_PROJECT: openwiki`: LangSmith project name

### GitHub Actions Permissions

Workflows follow principle of least privilege:

- **Default**: `contents: read` (read-only)
- **PR labeler**: `pull-requests: write`, `issues: write`
- **Release**: `id-token: write` (trusted publishing), `contents: write` (GitHub Release creation)
- **OpenWiki update**: `contents: write`, `pull-requests: write`

Isolated jobs (build, testing) receive no write permissions; publishing jobs run in separate jobs with restricted scope.

### Custom Actions

**`uv_setup`** (`.github/actions/uv_setup`):
- Sets up Python via official `setup-python` action
- Configures `uv` tool with optional caching
- Supports per-package cache suffixes to avoid cross-contamination
- Parameters: `python-version`, `cache-suffix`, `working-directory`, `enable-cache`

## Important Invariants & Failure Modes

1. **No caching in release pre-checks**: Missing dependencies would be masked by cached venvs, allowing broken releases to publish
2. **Minimum version downgrade isolation**: Minimum version tests reinstall packages in fresh virtual environment context, not via constraint relaxation alone
3. **Separate build/publish jobs**: Build job has no PyPI credentials; publishing job has no build tools, preventing supply-chain attacks
4. **Change detection scope**: VCR and extended test matrices only include packages with appropriate markers; adding test files without markers won't trigger corresponding test suites
5. **Prerelease blocking**: Stable releases reject any prerelease dependencies, preventing version resolution issues in downstream users
6. **Tag/version synchronization**: Release workflow validates git tags match expected version format before publishing, catching manual tag drift

## Extension Points

1. **Adding new package types**: Update `check_diff.py` to recognize new directories and map them to appropriate test matrices
2. **Adding partners to release testing**: Update `test-prior-published-packages-against-new-core` matrix and `skip-prior-published-package-checks` input options (keep in sync)
3. **Adding new linting/type checkers**: Extend `_lint.yml` job steps and dependency groups; ensure `make lint_package` target exists
4. **Adding integration test credentials**: Add environment variable to `integration_tests.yml` job and ensure `make integration_tests` target handles optional credentials
5. **Custom test suites**: Create `extended_testing_deps.txt` in package directory and define `make extended_tests` target
6. **OpenWiki pages**: Add to `openwiki/` directory; auto-updated on each scheduled run
