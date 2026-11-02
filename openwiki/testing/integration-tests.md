---
type: "Reference"
title: "Integration Testing: System-Level Validation"
description: "Integration testing patterns for multi-component workflows, API interactions, and provider integrations. Covers VCR recording, live API testing, and cross-package validation."
tags: ["integration-testing", "vcr", "api-testing", "standard-tests", "mocking"]
sources:
  - id: openwiki-source-d8a8900818f4abab719bd1b7
    resource: repo://.github/workflows/_test_vcr.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-c2966498bbc4d18ea148ee58
    resource: repo://libs/core/extended_testing_deps.txt
  - id: openwiki-source-9a9f29414e5bd045f28c0081
    resource: repo://libs/partners/openai/Makefile
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-ed8ac34fd0216d557729038e
    resource: repo://libs/partners/openai/tests/conftest.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
  - id: openwiki-source-db02c1dda8563ab005cd9d62
    resource: repo://libs/standard-tests/langchain_tests/conftest.py
  - id: openwiki-source-025cad4ae99967890152b7e0
    resource: repo://libs/standard-tests/README.md
generated: { by: "openwiki/0.5.0", at: "2026-10-26T08:33:32.568Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
---

## Overview

Integration testing in LangChain validates multi-component workflows, external API interactions, and provider integrations at the system level. The strategy separates concerns across three testing layers:

1. **VCR cassette-backed testing** — Records and replays HTTP interactions to catch stale test inputs without API credentials
2. **Live API testing** — Daily scheduled runs with real credentials against partner providers
3. **Standard tests conformance** — Partner integration conformance via inherited test suites from `langchain-tests`

This layered approach balances test speed (VCR replays run in seconds), cost (no daily API calls for cassette tests), and confidence (live credentials verify against production endpoints).

---

## VCR Recording and Playback: Cassette-Based Integration Tests

### What VCR Does

VCR (via `vcrpy`) records HTTP request/response pairs ("cassettes") and replays them in subsequent test runs. Tests marked with `@pytest.mark.vcr` automatically use cassettes instead of hitting live APIs.

**Key Workflow:**
- **First run**: Test executes with real API credentials; HTTP interactions are recorded to a `.yaml.gz` cassette file
- **Subsequent runs**: VCR loads the cassette, matches incoming requests, and replays stored responses — no API call happens
- **Stale cassettes**: If test code changes the request (e.g., different parameters), VCR's request matcher fails and alerts developers to update the cassette

### Cassette Storage and Format

Cassettes live alongside test files:

```
tests/
  integration_tests/
    chat_models/
      test_base.py
      cassettes/
        test_invoke.yaml.gz         # One cassette per marked test
        test_stream.yaml.gz
        TestChatModel.test_abatch.yaml.gz  # Class+method naming
```

Cassettes use **YAML + gzip compression** for storage:

- **YAML format**: Human-readable, inspectable in text editors (after decompression)
- **Gzip compression**: Reduces file size for large API responses (images, audio, PDFs)
- **Custom serializer**: `langchain_tests.conftest.CustomSerializer` handles YAML ↔ gzip conversion and securely parses YAML (uses `yaml.safe_load`, not `yaml.load`)

### Security: Credential Redaction

Cassettes **redact sensitive data** before saving to version control. The `base_vcr_config()` from `langchain_tests.conftest` applies default filtering:

**Base redactions** (`CustomSerializer` registers globally):
- Authorization headers → `PLACEHOLDER`
- API keys (`x-api-key`, `api-key`) → `PLACEHOLDER`

**Partner-specific redactions** (e.g., OpenAI's `conftest.py`):
- OAuth tokens, refresh tokens, JWT strings → `**REDACTED**` or `**REDACTED-JWT**`
- All request/response headers → `**REDACTED**` (defense-in-depth; OAuth tokens, User-Agent, etc.)
- Request URI → `**REDACTED**`
- OAuth secrets in JSON bodies (recursively: `access_token`, `refresh_token`, `id_token`, `code`, `device_code`, `client_secret`)
- Binary magic byte prefixes detected: PNG, JPEG, GIF, WAV, Ogg, MP3, PDF, ZIP — skipped (no secrets in binary)

**Cassette modification is visible** in git diffs, allowing post-recording leak scans before merge.

### VCR Configuration: Matching and Recording Modes

VCR configuration is set per-package in a `vcr_config` pytest fixture (e.g., `/libs/partners/openai/tests/conftest.py`):

| Setting | Default | Purpose |
|---------|---------|---------|
| `record_mode` | `"once"` | Record only if cassette missing; reuse existing |
| `match_on` | `["method", "uri", "body"]` | Match requests by HTTP method, URI, and request body (for JSON: normalized, ignoring key order) |
| `allow_playback_repeats` | `True` | Replay the same cassette response for duplicate requests |
| `decode_compressed_response` | `True` | Decompress gzipped API responses before matching |
| `cassette_library_dir` | `tests/cassettes` | Where cassettes are stored |
| `path_transformer` | `VCR.ensure_suffix(".yaml.gz")` | Auto-append filename suffix to cassette name |

**Custom matchers** (e.g., OpenAI's `_json_body_matcher`) normalize JSON bodies to ignore key order and docstring formatting changes across Python versions.

### Recording Cassettes: When and How

**Recording triggers:**
- First time a test runs (cassette doesn't exist)
- On demand: Run with `--record-mode=new_episodes` or `--record-mode=all` to force re-record
- Manual: Delete existing cassette and re-run

**Recording flow (in CI):**
```bash
# Run with live credentials; VCR records responses
uv run --group test --group test_integration pytest tests/integration_tests/
```

**In playback-only mode** (VCR cassette tests, `_test_vcr.yml`):
```bash
# Fake API key; VCR must replay from cassette or test fails
OPENAI_API_KEY=sk-fake pytest --record-mode=none -m vcr tests/integration_tests/
```

This catches **stale cassettes** (test code changed but cassette wasn't updated) without needing real credentials.

---

## Live API Testing: Daily Scheduled Integration Tests

### Workflow: `integration_tests.yml`

Daily (1 PM UTC) and on-demand integration tests run against **live partner APIs** with real credentials stored as GitHub Secrets.

**Key characteristics:**
- **Scope**: Partner packages (`libs/partners/openai`, `libs/partners/anthropic`, etc.) and external dependents
- **Frequency**: Scheduled daily; manually triggerable per package or subset
- **Credentials**: GitHub organization secrets (scoped to "Scheduled testing" environment)
- **Tracing**: Routes traces to LangSmith project with GitHub run metadata for debugging
- **Concurrency**: Serialized per package to avoid racing credential usage
- **Python versions**: 3.10, 3.14 (OpenAI and Perplexity packages require 3.11, 3.14)

### Python Version Matrix: Special Handling for OpenAI and Perplexity

The integration test matrix generation applies different Python version sets based on package type:

- **Default packages** (Anthropic, Groq, Google, etc.): Python 3.10, 3.14
- **OpenAI and Perplexity packages**: Python 3.11, 3.14 (due to SDK compatibility constraints)

This distinction is implemented in the `compute-matrix` job's matrix generation logic:

```bash
# Default versions for most packages
python_version='["3.10", "3.14"]'
# Minimum 3.11 versions for packages with SDK constraints
python_version_min_3_11='["3.11", "3.14"]'

# Apply version set based on package directory
matrix="$(jq -nc ... \
  '{include: [$directories[] as $dir |
    (if ($dir == "libs/partners/openai" or $dir == "libs/partners/perplexity")
     then $integration_versions else $versions end)[] |
    {"working-directory": $dir, "python-version": .}]}')"
```

This ensures OpenAI and Perplexity SDKs are tested only against Python versions they officially support, while other packages test a broader version range.

### Test Dependency Groups: `test_integration`

Live API tests require additional dependencies beyond unit tests. Packages declare these in `pyproject.toml`:

```toml
[dependency-groups]
test = [
    "vcrpy>=8.2.0",          # Cassettes for other tests
    "pytest>=9.0.0",
    # ... core unit test deps
]
test_integration = [
    "httpx>=0.27.0,<1.0.0",  # Extra HTTP client (if needed)
    "pillow>=12.1.1",        # Image processing (vision tests)
]
```

The CI workflow installs both groups:
```bash
uv sync --group test --group test_integration
```

This ensures integration tests have specialized dependencies without bloating unit test installations.

### Credential Management: GitHub Secrets

Credentials are GitHub organization secrets, injected at runtime:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
  # ... 40+ partner and service credentials
```

**Security:**
- Secrets are scoped to the "Scheduled testing" environment (requires approval)
- Never logged or printed (GitHub masks secrets in output)
- Only available to trusted code (typically partner-owned packages in the monorepo)

### LangSmith Tracing

Integration tests emit traces to LangSmith's `scheduled-testing-py` project (configurable) with GitHub context metadata:

```yaml
LANGSMITH_TRACING: "true"
LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
LANGSMITH_TAGS: "github-actions,libs/partners/openai,python-3.10,sha-abc123"
```

Traces link back to the originating Actions run, enabling developers to debug failures by examining LLM calls and chain execution.

### External Packages

Some partner packages live in separate repositories and are checked out separately during the workflow:

- `langchain-google` (Google Gen AI, Vertex AI)
- `langchain-aws` (AWS Bedrock)

The workflow reorganizes them into `libs/partners/` structure, then overlays local langchain packages:

```bash
uv pip install \
  -e langchain/libs/core \
  -e langchain/libs/standard-tests \
  -e langchain/libs/partners/anthropic  # Local editable installs
```

This ensures external packages test against the current branch's langchain code, not published releases.

---

## Standard Tests: Partner Conformance Framework

### Purpose and Architecture

`langchain-tests` (published package) provides base test classes that all partner integrations must inherit from. This enforces **conformance contracts** — e.g., all chat model providers must behave identically for critical operations.

**Base classes** (in `langchain_tests` package):
- `ChatModelUnitTests` / `ChatModelIntegrationTests` — Chat model providers
- `EmbeddingsUnitTests` / `EmbeddingsIntegrationTests` — Embedding providers
- `ToolsUnitTests` / `ToolsIntegrationTests` — Tool/function-calling providers
- `VectorStoreIntegrationTests` — Vector store providers
- `RetrieversIntegrationTests` — Retriever providers

**Partner usage pattern** (e.g., OpenAI):

```python
# tests/unit_tests/test_standard.py
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from langchain_openai import ChatOpenAI

class TestOpenAIStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI
```

The test framework automatically generates ~30 test methods covering invoke, stream, batch, async variants, tool calling, structured output, and edge cases.

### Standard Test Override Rules

Partners may **skip or xfail** tests with justification, but **cannot modify test logic**. The `test_no_overrides_DO_NOT_OVERRIDE` check enforces this:

```python
# Allowed: mark as xfail with a reason
@pytest.mark.xfail(reason="Provider doesn't support streaming")
def test_stream(self): ...

# Disallowed: change the test body
def test_stream(self):
    # custom logic here  ← Fails the conformance check
```

If a test cannot pass, the partner must use `@pytest.mark.xfail(reason="...")` to document why.

### Integration vs. Unit Tests

| Test Type | Location | Requirements | Network | Credentials | Speed |
|-----------|----------|--------------|---------|-------------|-------|
| **Unit** | `tests/unit_tests/test_standard.py` | Core only; no API | No | No (mocked) | ~1s per test |
| **Integration** | `tests/integration_tests/test_standard.py` | Full package | Yes (VCR) | Optional (cassettes work) | ~2-5s per test |

Both inherit from the same standard test class hierarchy; integrations add API calls and assert realistic behavior.

---

## Debugging Integration Test Failures

### Issue: Stale Cassette

**Symptom:** `CassetteNotFoundError` or request doesn't match any cassette entry

**Cause:** Test code changed (e.g., different parameters, new assertion) but cassette wasn't re-recorded

**Fix:**
```bash
cd libs/partners/openai
# Delete the cassette
rm tests/cassettes/test_invoke.yaml.gz

# Re-run with live credentials (local dev or CI re-run with real API key)
uv run --group test --group test_integration pytest tests/integration_tests/test_base.py::TestChat::test_invoke

# Cassette is re-recorded; verify diffs before committing
git diff tests/cassettes/test_invoke.yaml.gz
```

### Issue: Request Mismatch in VCR Playback

**Symptom:** `vcr.errors.CannotOverwriteExistingCassetteException` or "No match found"

**Cause:** Request in test differs from what's in the cassette (e.g., parameter order in JSON, Python version differences in docstrings)

**Debug:**
1. Enable verbose logging: `pytest -vv --log-cli-level=debug`
2. Check `vcrpy` matcher: does it compare JSON bodies structurally or by byte string? (Check `conftest.py` for custom matchers)
3. For OpenAI: `_json_body_matcher` normalizes docstring indentation and key order, so Python 3.13+ docstring changes don't break cassettes

**Fix:**
- Update cassette if test change is intentional: `--record-mode=all`
- Adjust matcher if legitimate variance: e.g., ignore whitespace in JSON

### Issue: Live API Failure (integration_tests.yml)

**Symptom:** Test passes locally with cassettes but fails in scheduled run with live API

**Cause:** API behavior changed, provider endpoint down, or credential permissions insufficient

**Debug:**
1. Check LangSmith traces (linked in Actions run output) to see exact API request/response
2. Review provider's changelog for breaking changes
3. Verify secrets are up-to-date (tokens expire, keys rotate)
4. Check rate limits or quotas

**Fix:**
- Update test expectations if API contract changed
- Regenerate cassettes with new API response
- Fix credentials or contact provider

### Issue: Cassette Contains Secrets

**Symptom:** Pre-merge scanning detected `OPENAI_API_KEY` or JWT in cassette file

**Cause:** Credential redaction didn't trigger (custom cassette format, binary body, or new secret type not in redaction list)

**Fix:**
1. Delete cassette and re-record (forces redaction to run)
2. Or manually edit cassette YAML (decompress with `gzip -d`, redact, re-compress)
3. Add missing secret patterns to `conftest.py` redaction functions if it's a new provider format

---

## Running Integration Tests Locally

### With VCR (Replay Mode)

**Use pre-recorded cassettes, no API credentials needed:**

```bash
cd libs/partners/openai

# VCR playback-only mode (same as CI)
OPENAI_API_KEY=sk-fake make test_vcr

# Or manually:
uv run --group test pytest --record-mode=none -m vcr tests/integration_tests/
```

### With Live API (Record Mode)

**Hit real API and record cassettes (requires credentials):**

```bash
cd libs/partners/openai

# Set your credentials
export OPENAI_API_KEY=sk-...

# Run integration tests; VCR will record new cassettes
make integration_tests

# Or:
uv run --group test --group test_integration pytest tests/integration_tests/
```

**For external packages** (google-vertexai, aws):
```bash
# Authenticate first
gcloud auth application-default login  # For Google
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Then run
uv sync --group test --group test_integration
uv run pytest tests/integration_tests/
```

### Standard Tests Only

```bash
cd libs/partners/openai

# Run only conformance tests (unit + integration)
uv run --group test --group test_integration pytest -k "Standard"
```

---

## Extended Testing: Complex Workflows and Multi-Package Validation

### Extended Test Groups

Some packages declare extended test dependencies in `extended_testing_deps.txt`. These are conditional dependencies needed only for comprehensive testing:

```
# libs/extended_testing_deps.txt
langchain-community>=0.2.0          # Docstore examples
langchain-text-splitters>=0.2.0    # Splitting workflows
```

**When extended tests run:**
- Automatically triggered in `check_diffs.yml` for affected packages
- Manually triggered via `make extended_tests`
- Installs extra dependencies, then runs pytest with `--only-extended` marker

**Packages skip extended tests by default** to keep CI fast; only developers testing complex workflows explicitly enable them.

### Cross-Package Integration Testing

The `test-dependents` job in `integration_tests.yml` tests external packages (like `deepagents`) against the current branch's langchain code:

```yaml
test-dependents:
  # Runs against separate repo's tests with local langchain overrides
  uv sync --group test  # Install dependents' test dependencies
  uv pip install -e ./langchain/libs/core  # Override with local version
  make test
```

This catches **breaking changes** that don't fail partner tests but break downstream users.

---

## CI Workflow: VCR Cassette Tests in check_diffs.yml

The `_test_vcr.yml` reusable workflow runs in PR CI (called from `check_diffs.yml`) to validate cassettes without credentials:

**Key steps:**
1. Checkout repository
2. Set up Python + UV
3. Install test dependencies: `uv sync --group test`
4. Run VCR-only tests: `make test_vcr` (env: `OPENAI_API_KEY=sk-fake`)
5. Verify no file modifications (cassettes shouldn't change on playback)

**Why it matters:**
- Catches stale cassettes in PR (test code changed, cassette wasn't updated)
- No API credentials required (saves keys from exposure)
- Fast feedback (cassettes replay in seconds)
- Ensures cassettes are checked in and reproducible

---

## Configuration Reference

### VCR Configuration Locations

| File | Purpose |
|------|---------|
| `langchain_tests.conftest.base_vcr_config()` | Default config (authorization, x-api-key redaction) |
| `libs/partners/<name>/tests/conftest.py:vcr_config()` | Package-specific overrides (extra headers, serializer, matchers) |
| `libs/partners/<name>/tests/integration_tests/conftest.py` | Test-level fixtures (auth bypasses, env cleanup) |

### Pytest Marks for Integration Tests

| Mark | Usage | Scope |
|------|-------|-------|
| `@pytest.mark.vcr` | Enables cassette recording/playback for this test | Test function |
| `@pytest.mark.scheduled` | Runs only in daily scheduled tests, skip in check_diffs.yml | Test function |
| `@pytest.mark.flaky(retries=3, delay=1)` | Retry on transient failures (network, rate limits) | Test function |
| `@pytest.mark.default_cassette("file.yaml.gz")` | Override cassette name (for parameterized tests) | Test function |

---

## Related Operations

- **CI Workflows**: `/openwiki/operations/ci-workflows.md` — Details on `check_diffs.yml`, `integration_tests.yml`, change detection
- **Dev Commands**: `/openwiki/operations/dev-commands.md` — Local `make` targets (`make test_vcr`, `make integration_tests`)
