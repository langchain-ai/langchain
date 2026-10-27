---
type: "Reference"
title: "Ci workflows"
openwiki_generated: true
sources:
  - id: openwiki-source-34e57b5a3a0c875639ab72a7
    resource: repo://.github/scripts/check_diff.py
  - id: openwiki-source-c92cc62695c6def991956428
    resource: repo://.github/workflows/_release.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-6d4b4e707b8d60b6ccfa3425
    resource: repo://.github/workflows/openwiki-update.yml
  - id: openwiki-source-f8781d847f6481a966a44a68
    resource: repo://.github/workflows/pr_labeler.yml
generated: { by: "openwiki/0.5.0", at: "2026-10-26T08:33:32.568Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-26T08:33:32.568Z
---


## Overview

LangChain's CI/CD infrastructure centers on three interconnected GitHub Actions workflows that enforce code quality, compatibility, and release safety. The system uses **intelligent change detection** to run only tests affected by code changes, **matrix testing** across Python versions and Pydantic variants, and **strict release gates** that validate packages before publishing to PyPI.

### Core Components

The system comprises:

1. **PR CI (`check_diffs.yml`)** — Runs on every PR/push to main, detects changed files, and dynamically selects affected test suites
2. **Scheduled Integration Tests (`integration_tests.yml`)** — Daily remote API testing with live credentials to catch breaking changes in external dependencies
3. **Release Workflow (`_release.yml`)** — Publishes to PyPI, validates versions, tests against prior published partners, and requires environment approval

---

## PR CI: Intelligent Change Detection & Matrix Testing

### Workflow: `check_diffs.yml`

The primary PR CI workflow runs on every pull request and push to master. Its core responsibility is **detecting which packages changed and mapping those changes to appropriate test configurations**.

**Trigger:** Pull requests, merges to master, merge queue participation
**Key Features:**
- Cancels redundant runs (concurrency with `cancel-in-progress: true`)
- Reads-only permission level for security
- Runs on `ubuntu-latest` with frozen/read-only UV cache

### Change Detection: `check_diff.py`

The script analyzes git diffs to determine which directories need testing. It:

1. **Maps file changes to package directories**
   - Treats `libs/core`, `libs/langchain`, `libs/text-splitters`, `libs/langchain_v1`, and `libs/model-profiles` as core packages
   - Partner packages live in `libs/partners/{name}/`
   - Maps each changed file to one or more affected directories

2. **Builds a dependency graph**
   - Reads `pyproject.toml` from all packages to track dependencies
   - Loads extended dependencies from `extended_testing_deps.txt` for partner packages
   - Identifies which packages depend on changed packages (e.g., when `libs/core` changes, all dependents must also be tested)
   - Removes ignored partners (e.g., huggingface, due to CI instability) from dependent cascades

3. **Handles infrastructure changes conservatively**
   - Changes to `.github/workflows/`, `.github/tools/`, `.github/actions/`, or `.github/scripts/check_diff.py` trigger extended tests on all core packages (`libs/core`, `libs/text-splitters`, `libs/langchain`, `libs/langchain_v1`, `libs/model-profiles`)
   - This ensures CI/CD infrastructure changes don't break package testing

4. **Handles large diffs**
   - If ≥300 files changed (diff truncation), runs all tests to ensure no test skips occur due to incomplete information

### Output: Dynamic Test Matrix

The `build` job outputs JSON test matrices for seven parallel jobs:

- **lint** — Linting configurations for affected packages
- **test** — Unit test configurations with Python version matrix
- **test-pydantic** — Pydantic compatibility tests (2.0 through max compatible version)
- **extended-tests** — Extended test suites requiring additional dependencies
- **compile-integration-tests** — Compile-time checks for integration tests (no actual execution)
- **vcr-tests** — VCR cassette-backed integration tests in playback-only mode
- **dependencies** — Dependency compatibility checks

Each configuration includes `working-directory` and `python-version` (or `pydantic-version` for Pydantic tests).

### Python Version Matrix

The script generates different Python version matrices per package type:

- **libs/core**: Tests on Python 3.10, 3.11, 3.12, 3.13, 3.14
- **openai and perplexity packages**: Tests on Python 3.11, 3.14 for lint and compile-integration-tests jobs (special SDK compatibility handling)
- **Other packages**: Tests on Python 3.10, 3.14 (reduced to save CI time)

### Pydantic Compatibility Testing

For packages using Pydantic, the script:

1. Reads `uv.lock` files to determine the maximum Pydantic version currently locked
2. Reads `pyproject.toml` to determine the minimum supported Pydantic version
3. Generates a test configuration for each compatible minor version (e.g., `2.0.0`, `2.1.0`, ..., `2.10.0`)
4. Only includes minor versions compatible with both the package and `libs/core`

This ensures the package works across its entire declared Pydantic version range.

### Parallel Jobs Triggered

Once the `build` job outputs the matrices, seven jobs run in parallel (if not empty):

#### 1. Lint Job
- Runs `_lint.yml` reusable workflow
- Per package: Python version from matrix
- Checks code style, type hints, and formatting

#### 2. Unit Test Job
- Runs `_test.yml` reusable workflow
- Matrix: `working-directory` × `python-version`
- Executes `make tests` with unit test dependencies

#### 3. Pydantic Compatibility Tests
- Runs `_test_pydantic.yml` reusable workflow
- Matrix: `working-directory` × `pydantic-version`
- Installs pinned Pydantic version and runs `make tests`

#### 4. Extended Tests
- Runs in same workflow (not reusable)
- Installs extended test dependencies from `extended_testing_deps.txt`
- Executes `make extended_tests`
- Verifies working directory is clean (no uncommitted changes)

#### 5. Compile Integration Tests
- Runs `_compile_integration_test.yml` reusable workflow
- Compiles integration tests without executing them (faster feedback)
- Command: `pytest -m compile tests/integration_tests`

#### 6. VCR Cassette Tests
- Runs `_test_vcr.yml` reusable workflow
- Only for packages with VCR cassettes: `libs/partners/openai`
- Runs integration tests in playback-only mode (uses cached HTTP responses, no API keys)
- Detects stale cassettes that don't match actual API responses
- Note: While only openai has VCR cassettes, perplexity still receives special Python version handling (3.11, 3.14) in integration test jobs due to SDK compatibility requirements

#### 7. Release Option Validation
- `check-release-options` job validates that `_release.yml` dropdown options match actual package directories
- Ensures release workflow stays in sync with repository structure

---

## PR Labeling System

### Workflow: `pr_labeler.yml`

Automatically applies labels to PRs based on contributor type, file changes, PR size, and title patterns.

**Trigger:** PR opened, synchronized, reopened, or edited
**Key Security:**
- Uses `pull_request_target` to avoid checking out untrusted code
- Generates GitHub App token only for membership checks
- Uses standard `GITHUB_TOKEN` for label mutations

### Labeling Rules

The workflow applies labels in this order:

1. **Contributor Classification** (on `opened` only)
   - Checks org membership using GitHub App token
   - Applies `internal` label for org members
   - Applies `external` label for outside contributors
   - `external` label triggers downstream workflows (e.g., `require_issue_link.yml`)
   - Contributor tier labels determined by GitHub org role

2. **File-Based Labels** (skipped on `edited`)
   - Maps changed files to labels via configuration in `.github/scripts/pr-labeler-config.json`
   - Examples: `libs/partners/openai` → `partner: openai`, `docs/` → `area: documentation`
   - Supports both exact path matching and directory prefixes

3. **Size Labels** (skipped on `edited`)
   - Computes total changed lines (additions + deletions)
   - Assigns mutually exclusive size labels:
     - `size: XS` (≤10 lines)
     - `size: S` (11–50 lines)
     - `size: M` (51–250 lines)
     - `size: L` (251–500 lines)
     - `size: XL` (500+ lines)
   - Removes old size label when size changes

4. **Title-Based Labels** (on all events)
   - Parses PR title for type keywords: `fix:`, `feat:`, `docs:`, `chore:`, `refactor:`, etc.
   - Applies corresponding type labels (e.g., `type: feature`, `type: bug`, `type: documentation`)
   - Removes stale type labels when title changes

### Label Concurrency

Uses concurrency group per PR number with separate handling:
- `opened` events use a dedicated concurrency key to prevent losing tier labels (full serialization)
- Other events (`synchronized`, `reopened`, `edited`) allow cancellation

---

## Scheduled Integration Tests

### Workflow: `integration_tests.yml`

Runs daily integration tests against partner libraries using live API credentials. This workflow tests the full LangChain ecosystem—not just unit tests but real API interactions.

**Trigger:** 
- Daily at 1:00 PM UTC (scheduled via cron)
- Manual trigger via `workflow_dispatch` with flexible library selection

**Key Features:**
- Connects to real APIs: OpenAI, Anthropic, Fireworks, Groq, Mistral, XAI, Google Vertex AI, Google GenAI, AWS
- Uses live repository secrets for API keys
- Only runs on main repo (forks can trigger manually but don't run on schedule)
- Supports exclude lists and Python version overrides

### Matrix Configuration

Default libraries tested daily:
- `libs/partners/openai`
- `libs/partners/anthropic`
- `libs/partners/fireworks`
- `libs/partners/groq`
- `libs/partners/mistralai`
- `libs/partners/xai`
- `libs/partners/google-vertexai`
- `libs/partners/google-genai`
- `libs/partners/aws`

**Python versions (default):** 3.10, 3.14

**Special Python version handling:**
- **openai and perplexity packages**: Use Python 3.11 and 3.14 instead of 3.10 and 3.14 due to SDK compatibility requirements
- Other packages: Use default 3.10 and 3.14

**Manual run options:**
- Run all libraries or select one
- Exclude specific libraries from "all" runs
- Override Python version matrix

### Integration Test Execution

Per library:
1. Check out repository
2. Set up Python and UV
3. Sync dependencies with `uv sync --group test --group test_integration`
4. Execute `make integration_tests` (defined in each package's Makefile)
5. Live API keys injected from repository secrets

---

## Release Workflow: Publishing to PyPI

### Workflow: `_release.yml`

Manually triggered workflow that builds, tests, and publishes LangChain packages to PyPI with multiple safety gates.

**Trigger:** Manual `workflow_dispatch` with inputs for package, version, and optional bypasses

**Approval:** Requires Release environment approval (prevents accidental releases)

**Dropdown Packages:**
- Core: `core`, `langchain`, `langchain_v1`, `text-splitters`, `standard-tests`, `model-profiles`
- Partners: `anthropic`, `chroma`, `deepseek`, `exa`, `fireworks`, `groq`, `huggingface`, `mistralai`, `nomic`, `ollama`, `openai`, `openrouter`, `perplexity`, `qdrant`, `typesafe`, `xai`

### Release Jobs

#### 1. Build Job (`build`)

**Role:** Construct distribution package and validate version

**Security:** Runs in isolated environment with read-only permissions; release job has separate permissions to prevent credential sharing

**Steps:**
1. Check Python: must be master branch (or `dangerous-nonmaster-release` bypassed)
2. Validate version against `pyproject.toml`
3. Check PyPI: version must not already exist (fails if duplicate)
4. Build distribution with `uv build` (creates `.whl` and `.tar.gz`)
5. Upload artifact for downstream jobs

**Outputs:** `pkg-name` (e.g., `langchain-openai`), `version` (e.g., `0.1.5`)

**Safety Gates:**
- Fails if requested release version doesn't match pyproject.toml
- Fails if version already published on PyPI (prevents re-publishing)
- Fails if PyPI is unreachable (fail-closed: assumes version might exist)

#### 2. Release Notes Job (`release-notes`)

**Role:** Generate changelog and validate tag state

**Dependency:** Must run after `build` (validates version/tag state)

**Steps:**
1. Check out langchain repository with full history
2. Validate git tag doesn't already exist for this version
3. For pre-releases: find latest matching base version tag; fall back to latest tag
4. For regular releases: construct tag from `pkg-name==version`
5. Generate release notes: `git log --format="%s" PREV_TAG..HEAD -- WORKING_DIR`
6. Handle first-ever releases (no previous tag)

**Outputs:** `release-body` (formatted git log for GitHub release)

#### 3. Pre-Release Checks Job (`pre-release-checks`)

**Role:** Validate package works before publishing

**Dependency:** Requires build, release-notes

**Timeout:** 20 minutes

**No Caching:** Intentionally disables dependency cache to catch missing dependencies

**Steps:**

1. **Install package from wheel**
   - Download built wheel from artifact
   - Install with optional `--prerelease=allow` flag (for alpha releases with prerelease deps)
   - Import package: `import <package>` and verify exports

2. **Run unit tests against installed package**
   - `make tests`

3. **Check prerelease dependencies**
   - Runs script `check_prerelease_dependencies.py`
   - Blocks release if any dependencies allow prerelease versions (unless this release is also a prerelease)

4. **Test minimum dependency versions**
   - Extracts minimum versions from `pyproject.toml`
   - Uses script `get_min_versions.py`
   - Installs minimum versions: `pip install --force-reinstall <min-versions>`
   - Runs `make tests` with `--prerelease=allow` if enabled
   - Excludes `test_serdes` tests (serialization tests may differ across versions)

5. **Run integration tests** (partner packages only)
   - Only if `EFFECTIVE_WORKING_DIR` starts with `libs/partners/`
   - Installs integration test dependencies: `uv sync --group test --group test_integration`
   - Injects all partner API secrets
   - Executes `make integration_tests`

**API Keys Injected (for partner integration tests):**
- LangSmith credentials
- OpenAI, Azure OpenAI (multiple deployments)
- Anthropic
- Google Cloud (Vertex AI, Search)
- Mistral, Groq, Huggingface, EXA, Nomic, WatsonX, Astra DB, Elasticsearch, MongoDB Atlas, Upstage, Fireworks, XAI, DeepSeek, Ollama, OpenRouter

#### 4. TestPyPI Publish Job (`test-pypi-publish`)

**Role:** Publish to staging PyPI for final validation before real release

**Dependency:** Requires build, release-notes, pre-release-checks

**Security:** Uses trusted publishing (OIDC token, no API keys)

**Steps:**
1. Download built wheel from artifact
2. Publish to `test.pypi.org` with `--skip-existing` (allows re-runs without failure)
3. Prints hash and verbose output for audit trail
4. Disables attestations (workaround for gh-action-pypi-publish v1.11.0+)

#### 5. Test Prior Published Packages Against New Core

**Role:** Validates new core doesn't break existing partner packages

**Trigger:** Only runs for `libs/core` releases

**Dependency:** Requires build, release-notes, test-pypi-publish, pre-release-checks

**Matrix:** Tests `anthropic` and `openai` partners (partners with most integration surface)

**Process per partner:**
1. Fetch latest non-yanked published version from PyPI (e.g., `langchain-openai==0.1.4`)
2. Verify source tag exists in git: `git ls-remote --exit-code --tags origin <TAG>`
3. Fetch that tag with shallow clone
4. Check out published partner package at that tag
5. Install new unreleased core wheel: `pip install ../../core/dist/*.whl`
6. Run unit tests against old partner code + new core
7. Run integration tests (with live API keys)

**Skip options:**
- `skip-prior-published-package-checks` input allows skipping specific partners:
  - `none` (default) — run all checks
  - `anthropic` or `openai` — skip that partner
  - `all` — skip all checks

#### 6. Test Dependents Job (`test-dependents`)

**Role:** Validates new core/langchain_v1 doesn't break external projects

**Trigger:** Only runs for `libs/core` and `libs/langchain_v1` releases

**Dependency:** Requires build, release-notes, test-pypi-publish, pre-release-checks

**Current External Projects:**
- `deepagents` (from `langchain-ai/deepagents` repo)

**Matrix:** Python 3.11, 3.13

**Process:**
1. Check out both LangChain repo and external project repo
2. Install new unreleased core/langchain_v1 wheel
3. Run tests in external project: `make test`

**Extensible:** Add more external projects to matrix when dependencies are identified

#### 7. Publish to PyPI Job (`publish`)

**Role:** Final publication to production PyPI

**Dependency:** Requires all prior jobs (build, release-notes, test-pypi-publish, pre-release-checks, and others)

**Security:** 
- Uses trusted publishing (OIDC token, no API keys in workflow)
- PyPI trusts GitHub OIDC identity to verify package ownership
- No credentials stored or transmitted

**Steps:**
1. Download wheel artifact
2. Publish to `pypi.org` using `pypa/gh-action-pypi-publish` action
3. Prints hash and verbose output

#### 8. Mark Release Job (`mark-release`)

**Role:** Create git tag and GitHub release

**Dependency:** Requires publish job completion

**Git Operations:**
1. Create annotated git tag: `<package-name>==<version>`
2. Push tag to repository

**GitHub Release:**
1. Create GitHub release from tag
2. Include changelog body from release-notes job
3. Attach distribution files (wheel, source tarball)

---

## Release Workflow Inputs & Bypasses

### Primary Inputs

- **working-directory** (choice): Short package name (dropdown) or full path (workflow_call)
- **working-directory-override** (string): Full path override (takes precedence)
- **release-version** (string): Version to release; must match pyproject.toml
- **dangerous-nonmaster-release** (boolean): Allow release from non-master branch (use only for hotfixes)
- **allow-prereleases** (boolean): Allow prerelease dependencies during testing
- **skip-prior-published-package-checks** (choice): Skip backward-compatibility checks for specific partners

### Safety Model

The release workflow uses **layered safety gates**:

1. **Version verification** (build job)
   - Validates version matches pyproject.toml
   - Prevents duplicate publishes

2. **Pre-release checks** (pre-release-checks job)
   - Runs unit tests, integration tests
   - Checks for prerelease dependencies (stable releases must use stable deps)
   - Tests minimum version support

3. **TestPyPI validation** (test-pypi-publish job)
   - Publishes to staging PyPI
   - Allows manual verification before production release

4. **Backward compatibility** (test-prior-published-packages-against-new-core)
   - For core releases: validates against latest published partners
   - Catches breaking changes to public APIs

5. **External project testing** (test-dependents)
   - For core/langchain_v1 releases: validates external projects still work
   - Catches API breaks that unit tests might miss

6. **Manual approval** (Release environment)
   - Release job requires environment approval
   - Adds human gate before production publish

---

## Matrix Testing & Performance Benchmarks

### CodSpeed Benchmarks

For performance-critical packages, the `build` job includes CodSpeed benchmark configurations:

- **libs/core**: Tested with walltime mode (default CPU simulation too slow)
- **libs/partners/fireworks**: Walltime mode (~328s under CPU simulation)
- **libs/partners/openai**: Walltime mode (6 benchmarks, ~6 min under CPU simulation)
- Other packages: CPU simulation mode

Each package runs benchmarks on Python 3.13 with `uv run pytest --benchmark-compare ...`.

---

## Implementation Details & Key Files

### Core Scripts

- **`check_diff.py`** — Analyzes file changes, builds dependency graph, outputs test matrices
- **`get_min_versions.py`** — Extracts minimum supported versions from `pyproject.toml`
- **`check_prerelease_dependencies.py`** — Verifies stable releases don't depend on prereleases
- **`pr-labeler.js`** — Implements file-based, size-based, and contributor-tier labels
- **`pr-labeler-config.json`** — Configuration for file-to-label mappings

### Reusable Workflows

- **`_lint.yml`** — Linting job (imported by check_diffs.yml)
- **`_test.yml`** — Unit test job (imported by check_diffs.yml)
- **`_test_pydantic.yml`** — Pydantic version compatibility (imported by check_diffs.yml)
- **`_compile_integration_test.yml`** — Integration test compilation (imported by check_diffs.yml)
- **`_test_vcr.yml`** — VCR cassette playback tests (imported by check_diffs.yml)
- **`_release.yml`** — Release workflow (manually triggered or called by other workflows)

### Configuration

- **`.github/scripts/pr-labeler-config.json`** — Maps file patterns to PR labels
- **Extended test dependencies:** `<package>/extended_testing_deps.txt` — Additional dependencies for extended tests
- **Minimum versions:** Declared in `pyproject.toml` under `[dependency-groups]` with PEP 440 constraints

---

## Failure Modes & Recovery

### PR CI Failures

| Failure | Cause | Recovery |
|---------|-------|----------|
| `check_diff.py` crashes | New package type not handled | Update `check_diff.py` to recognize new `libs/*` structure |
| Test matrix empty | No files changed (infrastructure-only PR) | Expected; CI passes without running tests |
| Flaky test fails | Intermittent test failure | Re-run workflow or fix test (mark with `@pytest.mark.flaky`) |
| Dependency conflict | New dependency incompatible with existing | Update `pyproject.toml` constraints |

### Release Failures

| Failure | Cause | Recovery |
|---------|-------|----------|
| Version already published | Version not bumped before release | Increment version in `pyproject.toml`, retry |
| Prerelease dependency | Stable release depends on alpha package | Update dependency constraints or use `--prerelease=allow` (if deliberate) |
| Min version tests fail | Minimum version doesn't satisfy constraints | Fix test, update constraints, or bump minimum version |
| Partner integration tests fail | Partner API changed | Update partner integration code, or skip check with `skip-prior-published-package-checks` |
| External project fails | External project broken by API change | Fix API or coordinate release with external project |

---

## Best Practices

### For Contributors

1. **Limit PR scope** — Smaller PRs run faster and have clearer CI output
2. **Update version before release** — Ensure `pyproject.toml` version matches release version
3. **Test locally** — Run `make tests` and `make lint` before pushing
4. **Check extended test status** — Extended tests catch integration issues early
5. **Review integration test results** — Scheduled integration tests run daily; check results before merging

### For CI Maintainers

1. **Keep `check_diff.py` in sync** — Add new packages/directories to `LANGCHAIN_DIRS` or partners detection
2. **Monitor flaky tests** — Use `@pytest.mark.flaky` for known intermittent failures
3. **Update release options** — When adding a new package, update `_release.yml` dropdown options
4. **Watch secret rotation** — API keys in partner integration tests rotate; update repository secrets when credentials expire
5. **Review PR labeling rules** — Keep `pr-labeler-config.json` in sync with repository structure
6. **Document release bypasses** — When using `dangerous-nonmaster-release` or `skip-prior-published-package-checks`, document why in PR description

---

## Troubleshooting

### Why didn't my changes trigger tests?

**Cause:** Change detection determined no packages were affected.

**Debug:** Check `check_diff.py` logic:
- Is your changed file under `libs/core`, `libs/langchain`, `libs/partners/XYZ`, etc.?
- Are you editing `.github/` (should trigger core packages)?
- Did you exceed 300 changed files (triggers all tests)?

**Fix:** Edit `check_diff.py` if a new directory structure isn't recognized.

### Why is my package missing from the release dropdown?

**Cause:** Package added to repo but not registered in `_release.yml`.

**Fix:** Add package name to `working-directory` options in `_release.yml` workflow_dispatch inputs.

### Why did integration tests pass locally but fail in CI?

**Cause:** 
- Missing API key injected in CI (check secrets are defined)
- Different Python/dependency versions
- Network-dependent test with unstable external service

**Debug:** Re-run integration tests manually with `workflow_dispatch` or check external service status.

### Can I release from a branch other than master?

**Yes, but dangerous:** Use `dangerous-nonmaster-release=true`. This should only be used for critical hotfixes. Always create a PR to master afterward to merge your hotfix back to main.
