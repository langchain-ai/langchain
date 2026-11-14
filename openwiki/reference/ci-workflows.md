---
type: "Reference"
title: "GitHub Actions CI Workflows Reference"
description: "Complete technical reference for LangChain's GitHub Actions workflows, job configurations, triggers, matrix testing, and environment variables."
tags: [ci, workflows, github-actions, reference, automation]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-c92cc62695c6def991956428
    resource: repo://.github/workflows/_release.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Overview

LangChain's CI/CD infrastructure consists of interconnected GitHub Actions workflows that enforce code quality, dependency compatibility, and release safety. This reference documents all workflow files, their triggers, job configurations, matrix structures, and environment variables.

**Primary workflows:**
- **`check_diffs.yml`** — PR CI with intelligent change detection and dynamic test matrix
- **`integration_tests.yml`** — Scheduled daily integration tests against live APIs
- **`_release.yml`** — Multi-stage release workflow with validation gates
- **Supporting workflows** — Linting, testing, validation, PR labeling, and infrastructure checks

---

## PR CI Workflow: `check_diffs.yml`

### Trigger Configuration

```yaml
on:
  push:
    branches: [master]
  pull_request:
  merge_group:
```

Runs on:
- Every pull request (opened, synchronized, reopened)
- Pushes to `master` branch
- Merge queue participation

### Concurrency & Optimization

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

Cancels redundant runs within the same branch/PR to save CI resources. If another push arrives while the workflow is running, the earlier run is cancelled.

### Environment Variables

```yaml
env:
  UV_FROZEN: "true"      # Use frozen uv.lock (no updates)
  UV_NO_SYNC: "true"     # Skip sync for faster setup in build job
```

### Permissions

```yaml
permissions:
  contents: read
```

Minimal read-only access; no permission to write to repository.

---

## Job: `build` — Detect Changes & Set Matrix

**Purpose:** Analyze changed files and output dynamic test matrices for downstream jobs.

**Triggers:** If PR does not have `ci-ignore` label

**Environment:** `ubuntu-latest`

### Steps

1. **Checkout Code** — `actions/checkout@v6`
2. **Setup Python 3.11** — `actions/setup-python@v7.1.0`
3. **Get Changed Files** — `Ana06/get-changed-files@v2.3.0` with JSON format
4. **Run Change Detection Script** — `python .github/scripts/check_diff.py`
   - Installs: `packaging`, `requests`
   - Runs unit tests: `python -m unittest discover -s .github/scripts -p test_check_diff.py`
   - Generates matrix: `python .github/scripts/check_diff.py "$ALL_CHANGED_FILES" >> $GITHUB_OUTPUT`

### Outputs (JSON matrices)

Each output is a JSON array of job configurations:

- **`lint`** — Lint job configurations `[{working-directory, python-version}, ...]`
- **`test`** — Unit test configurations
- **`test-pydantic`** — Pydantic version matrix `[{working-directory, pydantic-version}, ...]`
- **`extended-tests`** — Extended test configurations
- **`compile-integration-tests`** — Integration test compile checks
- **`vcr-tests`** — VCR cassette-backed integration tests
- **`dependencies`** — Dependency compatibility checks

### Change Detection Logic (check_diff.py)

**Core packages identified:**
- `libs/core`
- `libs/langchain`
- `libs/text-splitters`
- `libs/langchain_v1`
- `libs/model-profiles`

**Partner packages:** `libs/partners/{name}/`

**Detection flow:**
1. Maps file changes → affected directories
2. Builds dependency graph from `pyproject.toml` and `extended_testing_deps.txt`
3. Cascades changes through dependents
4. Ignores partners: `huggingface` (CI instability)
5. Infrastructure changes (`.github/workflows/`, `.github/actions/`, `.github/scripts/check_diff.py`) trigger extended tests on all core packages
6. Diffs ≥300 files → run all tests (fallback for truncated diffs)

### Python Version Matrices

Generated per-package-type:

| Package Type | Python Versions |
|---|---|
| `libs/core` | 3.11, 3.12, 3.13, 3.14 |
| `openai`, `perplexity` (lint/compile) | 3.11, 3.14 |
| Other packages | 3.11, 3.14 |

### Pydantic Compatibility Testing

For packages using Pydantic:
1. Read `uv.lock` for current max version
2. Read `pyproject.toml` for minimum version
3. Generate test for each compatible minor version (e.g., `2.0.0`, `2.1.0`, ..., `2.10.0`)
4. Filter to versions compatible with both package and `libs/core`

---

## Parallel Jobs Triggered by `build` Output

### Job: `lint`

**Reusable workflow:** `./.github/workflows/_lint.yml`

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.lint) }}`

**Skips if:** `needs.build.outputs.lint == '[]'`

**Parallelization:** One job per `(working-directory, python-version)` pair

**Execution:**
```yaml
with:
  working-directory: ${{ matrix.job-configs.working-directory }}
  python-version: ${{ matrix.job-configs.python-version }}
```

---

### Job: `test`

**Reusable workflow:** `./.github/workflows/_test.yml`

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.test) }}`

**Skips if:** `needs.build.outputs.test == '[]'`

**Execution:**
```yaml
with:
  working-directory: ${{ matrix.job-configs.working-directory }}
  python-version: ${{ matrix.job-configs.python-version }}
```

**Runs:** `make tests` in package directory

---

### Job: `test-pydantic`

**Reusable workflow:** `./.github/workflows/_test_pydantic.yml`

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.test-pydantic) }}`

**Skips if:** `needs.build.outputs.test-pydantic == '[]'`

**Execution:**
```yaml
with:
  working-directory: ${{ matrix.job-configs.working-directory }}
  pydantic-version: ${{ matrix.job-configs.pydantic-version }}
```

**Runs:** Unit tests with pinned Pydantic version

---

### Job: `compile-integration-tests`

**Reusable workflow:** `./.github/workflows/_compile_integration_test.yml`

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.compile-integration-tests) }}`

**Skips if:** `needs.build.outputs.compile-integration-tests == '[]'`

**Execution:**
```yaml
with:
  working-directory: ${{ matrix.job-configs.working-directory }}
  python-version: ${{ matrix.job-configs.python-version }}
```

**Command:** Compiles integration tests without execution (detects import errors early)

---

### Job: `vcr-tests`

**Reusable workflow:** `./.github/workflows/_test_vcr.yml`

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.vcr-tests) }}`

**Skips if:** `needs.build.outputs.vcr-tests == '[]'`

**Packages with VCR cassettes:** `libs/partners/openai`

**Execution:**
```yaml
with:
  working-directory: ${{ matrix.job-configs.working-directory }}
  python-version: ${{ matrix.job-configs.python-version }}
```

**Mode:** Playback-only (uses recorded HTTP responses; no API keys needed)

---

### Job: `extended-tests`

**Environment:** `ubuntu-latest`, timeout 20 minutes

**Matrix:** `matrix.job-configs: ${{ fromJson(needs.build.outputs.extended-tests) }}`

**Skips if:** `needs.build.outputs.extended-tests == '[]'`

**Execution:**
```bash
uv venv
uv sync --group test
VIRTUAL_ENV=.venv uv pip install -r extended_testing_deps.txt
VIRTUAL_ENV=.venv make extended_tests
```

**Verification:** Ensures working directory remains clean after tests (no uncommitted changes)

---

### Job: `check-release-options`

**Environment:** `ubuntu-latest`

**Condition:** `github.repository_owner == 'langchain-ai'` (main repo only)

**Purpose:** Validates that `_release.yml` dropdown options match actual package directories

**Execution:**
```bash
python -m pip install pyyaml pytest
python -m pytest .github/scripts/test_release_options.py -v
```

---

### Job: `ci_success` — Final Status Check

**Environment:** `ubuntu-latest`

**Depends on:** All job results (always runs, even if some fail)

**Purpose:** Aggregate check status for merge protection

**Outputs JSON:** All job results and exit code

**Condition:**
```yaml
if: |
  always()  # Run regardless of upstream failures
```

**Merge blocking:** Uses exit code 0 (all pass) or 1 (any failure/cancellation)

---

## Integration Tests Workflow: `integration_tests.yml`

### Trigger Configuration

```yaml
on:
  workflow_dispatch:
    inputs:
      working-directory:
        type: choice
        description: "Library to test (select from dropdown)"
        default: "all"
        options: [all, core, langchain, langchain_v1, text-splitters, standard-tests, 
                  model-profiles, anthropic, aws, chroma, deepseek, exa, fireworks, 
                  google-genai, google-vertexai, groq, huggingface, mistralai, nomic, 
                  ollama, openai, openrouter, perplexity, qdrant, typesafe, xai]
      working-directory-override:
        type: string
        description: "Manual override — takes precedence over dropdown (e.g. libs/partners/partner-xyz)"
      exclude:
        type: string
        description: "Comma-separated short names to drop from the 'all' run (e.g. openai,anthropic)"
      python-version-override:
        type: string
        description: "Python version override — defaults to 3.11 and 3.14 (e.g. 3.12)"
  schedule:
    - cron: "0 13 * * *"  # Daily at 1:00 PM UTC (9 AM EDT, 6 AM PDT)
```

### Environment Variables

```yaml
env:
  UV_FROZEN: "true"
  DEFAULT_LIBS: >-
    ["libs/partners/openai",
    "libs/partners/anthropic",
    "libs/partners/fireworks",
    "libs/partners/groq",
    "libs/partners/mistralai",
    "libs/partners/xai",
    "libs/partners/google-vertexai",
    "libs/partners/google-genai",
    "libs/partners/aws"]
```

### Job: `compute-matrix`

**Condition:** `github.repository_owner == 'langchain-ai' || github.event_name != 'schedule'`

**Purpose:** Generate test matrix based on inputs or defaults

**Output:**
```yaml
outputs:
  matrix: ${{ steps.set-matrix.outputs.matrix }}
  python-versions: ${{ steps.set-matrix.outputs.python-versions }}
```

**Matrix computation:**
- Priority: override string > dropdown choice > DEFAULT_LIBS
- Maps short names to full paths (case statement for core vs. partners)
- Excludes specified packages from all-run
- Generates include list: `[{working-directory, python-version}, ...]`

**Python version handling:**
- Default: `[3.11, 3.14]`
- If override specified: uses override value for single-version test
- Example: `python-version-override: "3.12"` → `[3.12]`

---

### Job: `integration-tests`

**Environment:** `ubuntu-latest`, 30 minute timeout

**Environment approval:** "Scheduled testing" (restricts secret access)

**Concurrency:** Per `(working-directory, python-version)` pair, **no cancellation** (prevents live API credential conflicts between manual and scheduled runs)

**Matrix:** `include: ${{ fromJSON(needs.compute-matrix.outputs.matrix).include }}`

**Repository checkouts:** Monorepo + external repos
- `langchain-ai/langchain` → `./langchain` (main monorepo)
- `langchain-ai/langchain-google` → `./langchain-google` (external Google packages)
- `langchain-ai/langchain-aws` → `./langchain-aws` (external AWS packages)

**External package setup:**
```bash
rm -rf langchain/libs/partners/google-genai langchain/libs/partners/google-vertexai
mv langchain-google/libs/genai langchain/libs/partners/google-genai
mv langchain-google/libs/vertexai langchain/libs/partners/google-vertexai
mv langchain-aws/libs/aws langchain/libs/partners/aws
```

**Dependency installation strategy:**
- Partner packages use `[tool.uv.sources]` in `pyproject.toml` to resolve `langchain-core`/`langchain` to local editable installs
- External Google/AWS packages don't declare `[tool.uv.sources]`, so `uv sync` pulls from PyPI
- Overlay local editable installs after sync for external packages to test against current branch

**Installation steps:**
```bash
cd langchain/${{ matrix.working-directory }}
uv sync --group test --group test_integration

case "${{ matrix.working-directory }}" in
  libs/partners/google-genai)
    uv pip install -e $GITHUB_WORKSPACE/langchain/libs/core \
                   -e $GITHUB_WORKSPACE/langchain/libs/standard-tests
    ;;
  libs/partners/google-vertexai)
    uv pip install -e $GITHUB_WORKSPACE/langchain/libs/core \
                   -e $GITHUB_WORKSPACE/langchain/libs/langchain_v1 \
                   -e $GITHUB_WORKSPACE/langchain/libs/standard-tests
    ;;
  libs/partners/aws)
    uv pip install -e $GITHUB_WORKSPACE/langchain/libs/core \
                   -e $GITHUB_WORKSPACE/langchain/libs/langchain_v1 \
                   -e $GITHUB_WORKSPACE/langchain/libs/langchain \
                   -e $GITHUB_WORKSPACE/langchain/libs/standard-tests \
                   -e $GITHUB_WORKSPACE/langchain/libs/partners/anthropic
    ;;
esac
```

**Execution:**
```bash
cd langchain/${{ matrix.working-directory }}
make integration_tests
```

### LangSmith Metadata Injection

Each test execution injects GitHub metadata for tracing and observability:

```bash
metadata=$(jq -nc \
  --arg github_sha "$GITHUB_SHA" \
  --arg github_run_id "$GITHUB_RUN_ID" \
  --arg github_run_attempt "$GITHUB_RUN_ATTEMPT" \
  '{...}')
echo "LANGSMITH_METADATA=$metadata" >> "$GITHUB_ENV"
```

**Environment variables injected:**
```yaml
env:
  LANGSMITH_TRACING: "true"
  LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
  LANGSMITH_PROJECT: "scheduled-testing-py"
  LANGSMITH_TAGS: "github-actions,${{ matrix.working-directory }},python-${{ matrix.python-version }}"
  # Plus API keys for 20+ services (OpenAI, Anthropic, Google, AWS, etc.)
```

**Live API credentials provided** (when running with scheduled testing environment):
- LangSmith (Tracing, Gateway)
- OpenAI (+ Azure endpoints)
- Anthropic
- Google (Vertex AI, GenAI, Search)
- Mistral, Groq, Huggingface, EXA, Nomic, WatsonX, Astra DB, Elasticsearch, MongoDB Atlas, Upstage, Fireworks, XAI, DeepSeek, Ollama, OpenRouter, and others

---

## Release Workflow: `_release.yml`

### Trigger & Input Configuration

```yaml
on:
  workflow_call:
    inputs:
      working-directory:
        type: string
        required: true
      release-version:
        type: string
        default: ""
      allow-prereleases:
        type: boolean
        default: false
      skip-prior-published-package-checks:
        type: string
        default: "none"
        # Values: none, anthropic, openai, all
  workflow_dispatch:
    inputs:
      working-directory:
        type: choice
        options: [core, langchain, langchain_v1, text-splitters, standard-tests, 
                  model-profiles, anthropic, chroma, deepseek, exa, fireworks, 
                  groq, huggingface, mistralai, nomic, ollama, openai, openrouter, 
                  perplexity, qdrant, typesafe, xai]
      release-version:
        type: string
        required: true
      dangerous-nonmaster-release:
        type: boolean
        default: false
      allow-prereleases:
        type: boolean
        default: false
      skip-prior-published-package-checks:
        type: choice
        options: [none, anthropic, openai, all]
        default: "none"
```

### Environment Variables

```yaml
env:
  PYTHON_VERSION: "3.11"
  UV_FROZEN: "true"
  UV_NO_SYNC: "true"
  EFFECTIVE_WORKING_DIR: >-
    ${{ inputs.working-directory-override || ... }}  # Resolves to libs/{...}
```

### Job: `build` — Build Distribution

**Environment:** Release (approval required)

**Condition:** `github.repository_owner == 'langchain-ai' && (master || dangerous-nonmaster-release)`

**Outputs:**
```yaml
outputs:
  pkg-name: ${{ steps.check-version.outputs.pkg-name }}
  version: ${{ steps.check-version.outputs.version }}
  dist-sha256sums: ${{ steps.dist-checksums.outputs.sha256sums }}
```

**Steps:**
1. Checkout (no credentials)
2. Setup Python + uv (cache disabled for security)
3. Log release bypasses if enabled
4. **Check version** (Python step)
   - Read `pyproject.toml`
   - Validate against input `release-version` (if provided)
   - Query PyPI: version must not exist
   - Fail closed if PyPI unreachable
5. Build with `uv build` → `.whl` + `.tar.gz`
6. Compute SHA256 checksums
7. Upload artifact for downstream jobs

---

### Job: `release-notes` — Generate Changelog

**Condition:** Depends on `build`

**Outputs:**
```yaml
outputs:
  release-body: ${{ steps.generate-release-body.outputs.release-body }}
```

**Steps:**
1. Checkout with full history (`fetch-depth: 0`)
2. **Check tags** (shell script)
   - For pre-releases: find latest base version tag or latest tag
   - For releases: compute expected previous tag
   - Validate tag exists in git
3. **Generate release body**
   - Use git log between tags: `git log --format="%s" PREV_TAG..HEAD -- WORKING_DIR`
   - For first release: `git rev-list --max-parents=0 HEAD`

---

### Job: `pre-release-checks` — Validate Package

**Environment:** Release (approval required)

**Timeout:** 20 minutes

**Depends on:** build, release-notes

**Cache:** Disabled (catch missing dependencies)

**Steps:**
1. Checkout (no credentials)
2. Setup Python + uv (no cache)
3. Download built wheel artifact
4. **Import package**
   - Create venv, install wheel
   - Run: `import <pkg>; print(dir(<pkg>))`
5. Sync test dependencies: `uv sync --group test`
6. Re-install wheel (overwrite local version)
7. **Check prerelease dependencies**
   - Run: `check_prerelease_dependencies.py pyproject.toml`
   - Blocks if stable package depends on prerelease versions
8. **Run unit tests**: `make tests`
9. **Get minimum versions**: Parse `pyproject.toml`, run `get_min_versions.py`
10. **Test minimum versions** (if available)
    - Force-reinstall dependencies at minimums
    - Run: `make tests PYTEST_EXTRA="-q -k 'not test_serdes'"`
11. **Import integration test dependencies** (if partner): `uv sync --group test --group test_integration`
12. **Run integration tests** (if partner only): `make integration_tests`

**API credentials injected** (partner packages):
- LangSmith (Tracing, Gateway)
- OpenAI (+ Azure endpoints)
- Anthropic (Files API)
- Google (Vertex AI, GenAI, Search)
- Mistral, Groq, Huggingface, EXA, Nomic, WatsonX, Astra DB, Elasticsearch, MongoDB Atlas, Upstage, Fireworks, XAI, DeepSeek, Ollama, OpenRouter

---

### Job: `test-pypi-publish` — Publish to Test PyPI

**Environment:** Release (approval required)

**Depends on:** build, release-notes, pre-release-checks

**Permissions:** `id-token: write` (trusted publishing via OIDC)

**Steps:**
1. Checkout (no credentials)
2. Download wheel artifact
3. **Verify checksums** (fail if mismatch)
4. **Publish to test.pypi.org**
   - Repository: `https://test.pypi.org/legacy/`
   - `skip-existing: true` (idempotent re-runs)
   - `attestations: false` (workaround for v1.11.0+)

---

### Job: `test-prior-published-packages-against-new-core`

**Condition:** Only for `libs/core` releases

**Depends on:** build, release-notes, test-pypi-publish, pre-release-checks

**Matrix:** Partners: `anthropic`, `openai`

**Per-partner steps:**
1. Checkout LangChain (no credentials)
2. **Check if libs/core** (exit early if not)
3. Setup Python + uv (no cache)
4. Download artifact
5. **Check skip condition** (exit if skip requested)
6. **Query PyPI for latest published partner**
   - Find latest non-yanked final release (PEP 440)
   - Fail closed if PyPI unreachable
7. **Verify git tag exists**: `git ls-remote --exit-code --tags origin`
8. **Fetch and checkout published code**
   - Shallow fetch tag
   - Remove current partner code, restore from tag
9. **Install and test against new core**
   - `uv sync --group test --group test_integration`
   - `uv pip install ../../core/dist/*.whl`
   - `make test`
   - `make integration_tests` (with live API keys)

---

### Job: `test-dependents` — External Project Testing

**Condition:** Only for `libs/core` and `libs/langchain_v1` releases

**Depends on:** build, release-notes, test-pypi-publish, pre-release-checks

**Matrix:**
- Python: `[3.11, 3.13]`
- Packages:
  - `deepagents` (from `langchain-ai/deepagents`)

**Steps:**
1. Checkout LangChain (no credentials)
2. Checkout external project
3. Setup Python + uv (no cache)
4. Download artifact
5. **Install with overrides**
   - `uv sync --group test`
   - `uv pip install dist/*.whl` (override external's pinned versions)
6. **Run tests**: `make test`

---

### Job: `publish` — Publish to PyPI

**Condition:** All upstream pass or skip: `!cancelled() && !failure()`

**Environment:** Release (approval required)

**Depends on:** build, release-notes, test-pypi-publish, pre-release-checks, test-dependents, test-prior-published-packages-against-new-core

**Permissions:** `id-token: write` (trusted publishing)

**Steps:**
1. Checkout (no credentials)
2. Setup Python + uv (no cache)
3. Download artifact
4. **Verify checksums**
5. **Publish to pypi.org**
   - Uses `pypa/gh-action-pypi-publish@v1`
   - Verbose output and hash logging

---

### Job: `mark-release` — Create Git Tag & GitHub Release

**Condition:** `!cancelled() && !failure()`

**Environment:** Release (approval required)

**Permissions:** `contents: write` (git tag creation)

**Depends on:** build, release-notes, test-pypi-publish, pre-release-checks, publish

**Steps:**
1. Checkout (no credentials)
2. Setup Python + uv (no cache)
3. Download artifact
4. **Create release** (via `ncipollo/release-action@v1`)
   - Tag: `<pkg-name>==<version>` (e.g., `langchain-core==0.1.5`)
   - Body: Release notes from `release-notes` job
   - Artifacts: Attach `.whl` and `.tar.gz` from artifact
   - `makeLatest: true` only for `langchain-core` (primary package)
