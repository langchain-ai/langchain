---
type: "Testing & QA"
title: "Integration Testing: Live API Tests and VCR Cassettes"
description: "How to write integration tests that call real model APIs with VCR cassette recording for CI compatibility, including environment setup, cassette management, and parameterization patterns."
tags: [integration-tests, vcr, cassettes, api-testing, pytest, ci-cd, model-testing]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-bcf7be66f36f862f639f3c7a
    resource: repo://libs/langchain_v1/tests/integration_tests/conftest.py
  - id: openwiki-source-bae620f3bb8d2668c69ac079
    resource: repo://libs/langchain/tests/integration_tests/.env.example
  - id: openwiki-source-9a9f29414e5bd045f28c0081
    resource: repo://libs/partners/openai/Makefile
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-ed8ac34fd0216d557729038e
    resource: repo://libs/partners/openai/tests/conftest.py
  - id: openwiki-source-ce605960b642234837131282
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/conftest.py
  - id: openwiki-source-711811fb8df6b32329091d23
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
  - id: openwiki-source-d619d2c7cfe653fb9b712740
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base.py
  - id: openwiki-source-e758f7cc743f83c081224f08
    resource: repo://libs/partners/openai/tests/integration_tests/embeddings/test_base.py
  - id: openwiki-source-db02c1dda8563ab005cd9d62
    resource: repo://libs/standard-tests/langchain_tests/conftest.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

Integration tests in LangChain differ fundamentally from unit tests: they call real model APIs (OpenAI, Anthropic, etc.) and require network access and valid credentials. To make these tests reproducible and CI-friendly without exposing API keys or relying on external services, LangChain uses **VCR cassettes**—recorded HTTP interactions that are replayed during test execution.

This page covers the full lifecycle of integration testing: setting up environments, understanding the VCR pattern, organizing cassettes, writing tests that work with both live and recorded modes, handling parameterization, and managing cassette refresh workflows.

## Environment Setup: API Keys and .env

Integration tests require valid API credentials to record cassettes (once) and for running live integration tests in scheduled or on-demand scenarios. Store credentials in a `.env` file at the integration test directory root.

### .env File Location and Format

For most packages (e.g., OpenAI partner), the structure is:

```
libs/partners/openai/tests/.env
```

Example `.env.example` (visible in repo):

```bash
# openai
# your api key from https://platform.openai.com/account/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# searchapi
SEARCHAPI_API_KEY=your_searchapi_api_key_here
```

**Important**: `.env` files are `.gitignore`d and never committed. Copy `.env.example` to `.env` locally and populate with valid keys.

### Automatic Skipping When Keys Are Missing

Integration tests don't explicitly skip when API keys are absent. Instead:

1. Tests marked `@pytest.mark.scheduled` (for live API calls) run **only** in scheduled CI or when explicitly selected
2. Tests marked `@pytest.mark.vcr` (for cassette playback) run in CI via `make test_vcr` without credentials
3. Tests without markers but that consume API keys will fail at runtime if no `.env` is present—this is intentional for development environments

When running `make integration_tests` locally, you must have a valid `OPENAI_API_KEY` in `.env`. If you don't, the test will error, alerting you that credentials are needed.

### Environment Loading

The test conftest automatically loads `.env` on import:

```python
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

def _load_env() -> None:
    dotenv_path = PROJECT_DIR / "tests" / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

_load_env()
```

Credentials are then accessed via `os.environ["OPENAI_API_KEY"]` or similar in tests.

## The VCR Pattern: Recording and Playback

**VCR** (Video Cassette Recorder, implemented by the `vcrpy` library) records HTTP interactions—requests and responses—the first time a test runs with a live API. On subsequent runs, VCR replays the recorded cassette instead of making live network calls.

### Recording Phase (Once, by Developers)

When a new integration test is written or an existing test's behavior changes:

1. Run the test with `--record-mode=new_episodes` (or the default `once`) and a valid API key in `.env`
2. VCR intercepts all HTTP calls your test makes
3. **Before recording**, sensitive headers and tokens in request/response bodies are **scrubbed** (redacted to `PLACEHOLDER` or `**REDACTED**`)
4. The cassette is saved as a YAML file (optionally compressed as `.yaml.gz`)
5. The cassette is committed to the repository

### Playback Phase (Always, in CI and Local Development)

When running tests:

1. VCR loads the cassette file
2. Instead of making real API calls, VCR intercepts your HTTP client and returns the pre-recorded response
3. No API key is needed; no credentials are exposed
4. Tests are fast (no network latency) and deterministic (same response every time)

### Cassette Location

Cassettes are stored relative to test modules in a `cassettes/` subdirectory:

```
libs/partners/openai/tests/
  integration_tests/
    chat_models/
      test_base.py
      cassettes/
        test_base/
          TestChatOpenAICodexStandard.test_invoke.yaml.gz
          test_chat_openai.yaml.gz
```

Or at the test root:

```
libs/partners/openai/tests/cassettes/
  test_langchain_openai_embeddings_equivalent_to_raw.yaml.gz
  test_streaming_tool_call_v1_v2_parity.yaml.gz
```

The fixture `vcr_cassette_dir` (from `conftest.py`) computes the correct directory per test module:

```python
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = Path(request.module.__file__)
    return str(module.parent / "cassettes" / module.stem)
```

## Security: Scrubbing Sensitive Data

VCR cassettes are **committed to git** and **visible to the world** (in public repositories). To prevent accidental exposure of API keys, JWTs, OAuth tokens, and other secrets, LangChain uses a multi-layer scrubbing pipeline.

### What Gets Redacted

**Headers** (configured in `conftest.py`):

```python
_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("authorization", "PLACEHOLDER"),
    ("cookie", "PLACEHOLDER"),
]
```

**Request and Response Bodies** (OAuth secret fields):

```python
_OAUTH_SECRET_FIELDS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "code",
    "device_code",
    "client_secret",
})
```

Redaction is applied with specialized handlers:

- **JSON bodies**: Recursively walks the parsed JSON tree and redacts any field matching `_OAUTH_SECRET_FIELDS`
- **Form-encoded bodies**: Splits on `&` and redacts matching keys
- **JWT patterns**: Uses regex to detect and redact JWT-shaped strings anywhere in the body

**Binary payloads** (PNG, JPEG, PDF, audio, etc.) are **skipped**—their magic bytes are detected and the scrubbing stack is bypassed for performance (JWTs and OAuth secrets are ASCII, so binary bodies can't carry them).

### Scrubbing Configuration in conftest.py

```python
def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers and OAuth secrets from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    request.body = _scrub_oauth_secrets(request.body)
    return request

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["match_on"] = ["json_body"]  # Don't match on URI (it's redacted)
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"  # Compress cassettes
    return config
```

## Writing Integration Tests

### Basic Structure

Integration tests live in `tests/integration_tests/` (mirroring the layout of unit tests):

```
libs/partners/openai/tests/
  unit_tests/
  integration_tests/
    chat_models/
      __init__.py
      conftest.py
      test_base.py
      cassettes/
```

A minimal integration test using pytest-recording (which auto-hooks VCR):

```python
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@pytest.mark.vcr  # Mark test to use VCR
def test_chat_openai_invoke():
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert response.content  # Assert the response is non-empty
```

The `@pytest.mark.vcr` marker tells pytest-recording to automatically:

1. Look for a cassette file named `test_chat_openai_invoke.yaml.gz` in the module's `cassettes/` directory
2. Use VCR to intercept HTTP calls
3. Replay the cassette if it exists; record a new one if it doesn't (or if `--record-mode=new_episodes` is passed)

### Markers for Test Classification

**`@pytest.mark.scheduled`**: Marks tests for **live API calls** (no cassette). These run in scheduled CI workflows with real credentials.

```python
@pytest.mark.scheduled
def test_chat_openai_streaming_live():
    """Test streaming with a live API call."""
    chat = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    response = chat.invoke("Hello")
    assert response.content
```

These tests are **skipped in CI** unless explicitly selected or running in a scheduled job. Locally, they require a valid API key.

**`@pytest.mark.vcr`**: Marks tests for **cassette playback**. VCR is enabled; the cassette is replayed. No API key needed in CI.

```python
@pytest.mark.vcr
def test_chat_openai_with_tools():
    """Test tool calling with a cassette."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    tools = [...]
    response = chat.invoke(..., tools=tools)
    assert response.tool_calls
```

**No marker**: Tests without a marker run as **plain unit-like tests**, often for initialization, validation, or offline scenarios. They don't require an API key.

```python
def test_chat_openai_model_name():
    """Test model name is set correctly."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.model_name == "gpt-4o-mini"
```

### Parameterized Integration Tests

When testing the same scenario against multiple models or configurations, use `pytest.mark.parametrize`:

```python
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MODELS = ["gpt-4o-mini", "gpt-4o"]

@pytest.mark.vcr
@pytest.mark.parametrize("model_name", MODELS)
def test_chat_models_invoke(model_name: str):
    """Test invoke on different models."""
    chat = ChatOpenAI(model=model_name)
    response = chat.invoke([HumanMessage(content="Hello")])
    assert response.content
```

VCR automatically generates separate cassette files for each parameter combination:

```
cassettes/
  test_chat_models_invoke[gpt-4o-mini].yaml.gz
  test_chat_models_invoke[gpt-4o].yaml.gz
```

When the test runs with `model_name="gpt-4o-mini"`, VCR loads the cassette matching `[gpt-4o-mini]`. This allows testing multiple models with independent recorded interactions.

### Testing Streaming

Stream tests benefit from cassettes because they capture the full stream sequence:

```python
@pytest.mark.vcr
def test_chat_openai_streaming():
    """Test streaming behavior."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    chunks = []
    for chunk in chat.stream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[-1].content  # Final chunk has content
```

The cassette records all the streaming HTTP chunks, and VCR replays them in order during playback.

### Testing Error Paths

To test error handling (rate limits, malformed inputs, API errors), you can either:

1. **Record once with a real error** (e.g., using an invalid API key or endpoint), then replay the error response
2. **Create a cassette manually** with a canned error response for testing

Example: testing a `RateLimitError`:

```python
@pytest.mark.vcr
def test_chat_openai_handles_rate_limit():
    """Test that rate limit errors are handled gracefully."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    try:
        response = chat.invoke("Hello")
    except Exception as e:
        assert "rate" in str(e).lower()
```

The cassette for this test should contain a recorded 429 (Too Many Requests) response from the API.

## Cassette Management

### Recording New Cassettes

To record a new cassette or re-record an existing one:

```bash
cd libs/partners/openai

# Record new cassettes (default --record-mode=once)
uv run --group test --group test_integration pytest \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke

# Or explicitly with --record-mode=new_episodes to overwrite
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

**Prerequisites**:
- A valid `OPENAI_API_KEY` in `tests/.env`
- Network access (obviously)

**What happens**:
1. pytest-recording starts VCR in record mode
2. Your test runs and makes real HTTP calls to the API
3. VCR intercepts all requests and responses
4. Scrubbing functions redact sensitive headers and secrets
5. The cassette is serialized to YAML, compressed to `.yaml.gz`, and saved

**Commit the cassette**:

```bash
git add libs/partners/openai/tests/cassettes/test_chat_openai_invoke.yaml.gz
git commit -m "Add cassette for chat_openai_invoke integration test"
```

### Refreshing Stale Cassettes

When test code changes (e.g., you change the prompt, the model, or the parameters), the cassette may become stale. VCR will attempt to match the new request body against the old cassette; if there's a mismatch, it will fail with a `CannotOverwriteExistingCassetteException` error (in playback mode) or suggest re-recording.

To refresh:

```bash
cd libs/partners/openai
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

This overwrites the cassette with the new interaction. Re-verify the cassette was scrubbed correctly, then commit.

### Cassette Format

Cassettes are YAML files (optionally gzip-compressed). A typical cassette structure:

```yaml
interactions:
  - request:
      body: null
      headers: {}
      method: POST
      uri: https://api.openai.com/v1/chat/completions
    response:
      body:
        string: '{"choices": [{"message": {"content": "Hello!", "role": "assistant"}}]}'
      headers:
        content-type:
          - application/json
      status:
        code: 200
        message: OK
version: 1
```

Headers and secrets are replaced with `PLACEHOLDER` or `**REDACTED**`:

```yaml
request:
  headers:
    authorization:
      - "**REDACTED**"
    openai-organization:
      - "PLACEHOLDER"
```

### Inspecting Cassettes

To inspect a cassette, decompress and read:

```bash
gunzip -c libs/partners/openai/tests/cassettes/test_invoke.yaml.gz | head -50
```

Or, for a permanent view:

```bash
gunzip libs/partners/openai/tests/cassettes/test_invoke.yaml.gz
# Now test_invoke.yaml is readable
```

(Remember to re-compress or re-record before committing.)

## Running Integration Tests

### Full Integration Test Suite

Run all integration tests against live APIs (requires credentials):

```bash
cd libs/partners/openai
make integration_tests  # Runs full suite with live API calls
```

This invokes:

```bash
uv run --with 'openai>=2.45.0,<3.0.0' --group test --group test_integration pytest \
  -v --tb=short tests/integration_tests/

uv run --with 'openai>=3.0.0,<4.0.0' --group test --group test_integration pytest \
  -v --tb=short tests/integration_tests/
```

It also smoke-tests a single live request on each supported OpenAI SDK major version (2.x and 3.x).

### VCR Cassette Tests (CI Mode)

Run integration tests using only cassettes (no live API calls, no credentials):

```bash
cd libs/partners/openai
make test_vcr
```

This invokes:

```bash
uv run --group test pytest --record-mode=none -m vcr tests/integration_tests/
```

The `--record-mode=none` flag tells VCR to **only replay** cassettes and fail if a cassette is missing (never attempt a live call or record a new one).

This is what CI runs; it ensures cassettes are up-to-date and tests are repeatable without exposing secrets.

### Scheduled Integration Tests

Scheduled CI workflows (daily or on-demand) run live integration tests with real credentials:

```bash
cd libs/partners/openai
make integration_tests
```

Only tests marked `@pytest.mark.scheduled` run in these workflows. See [CI/CD Workflows](repo:///openwiki/ci-workflows.md#integration-test-compilation) for details.

## Streaming and Async Integration Tests

### Async Tests

Async integration tests work the same as sync tests:

```python
@pytest.mark.vcr
async def test_chat_openai_ainvoke():
    """Test async invoke."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = await chat.ainvoke("Hello")
    assert response.content
```

VCR intercepts the underlying `httpx` (or `requests`) library, so both sync and async HTTP calls are recorded and replayed identically.

### Streaming with `stream_events`

Tests that use `stream_events` (the LangChain event streaming API) also work with cassettes:

```python
@pytest.mark.vcr
def test_chat_openai_stream_events():
    """Test streaming events."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    events = []
    for event in chat.stream_events("Hello", version="v2"):
        events.append(event)
    assert len(events) > 0
```

The underlying HTTP calls are recorded in the cassette; the event structure is replayed exactly.

## VCR Configuration Details

### conftest.py Setup

Each package's `conftest.py` configures VCR globally:

```python
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    # Base config:
    # - record_mode: "once"
    # - filter_headers: ["authorization", "x-api-key", "api-key"]
    # - match_on: ["method", "uri", "body"]
    # - cassette_library_dir: "tests/cassettes"
    
    # Extend for OpenAI:
    config["match_on"] = ["json_body"]  # Custom JSON matching
    config["serializer"] = "yaml.gz"     # Compress cassettes
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    return config

def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    """Register custom serializer, persister, and matchers."""
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("json_body", _json_body_matcher)
```

### Record Modes

VCR supports several record modes:

- **`once`** (default): Record new cassettes; replay existing ones. Use for initial development.
- **`new_episodes`**: Record/overwrite cassettes even if they exist. Use when refreshing stale cassettes.
- **`none`**: Never record; replay only. Use in CI. Fails if cassette is missing.
- **`all`**: Always record, even for existing cassettes. Use sparingly.

### Request Matching

By default, VCR matches requests by:

```python
match_on: ["method", "uri", "body"]
```

This means: "A request matches a cassette if the HTTP method, URI, and body are identical."

For APIs with non-deterministic bodies (e.g., timestamps in the request), you can customize matchers or use `allow_playback_repeats` to reuse the same response for multiple request variants.

OpenAI tests use a custom `json_body` matcher:

```python
config["match_on"] = ["json_body"]  # Only match on JSON-parsed body (ignores whitespace/key order)
```

This allows cassettes to match even if JSON key order differs.

## Integration with CI/CD

### CI Workflows

The LangChain CI system runs integration tests at multiple points:

1. **Pull request checks** (`check_diffs.yml`): Compiles integration tests without running them
2. **VCR cassette tests** (`_test_vcr.yml`): Runs cassette-backed integration tests in playback-only mode
3. **Scheduled integration tests** (`integration_tests.yml`): Daily remote API testing with live credentials against partner libraries

See [CI/CD Workflows: GitHub Actions and Release Process](repo:///openwiki/ci-workflows.md) for full details, including scheduled workflows and record mode management.

### Cassette Validation in CI

The `test_vcr` job runs with `--record-mode=none`, which means:

- Cassettes are **never recorded** or overwritten in CI
- If a cassette is missing, the test **fails immediately**
- If request bodies don't match, the test **fails immediately**

This catches stale cassettes caused by test code changes without attempting live calls.

## Extension Points and Customization

### Custom VCR Matchers

For advanced matching logic (e.g., ignoring certain request fields), register a custom matcher in `conftest.py`:

```python
def _custom_matcher(r1: Any, r2: Any) -> None:
    """Match requests with custom logic."""
    # Example: match method and URI, ignore timestamps in body
    assert r1.method == r2.method
    assert r1.uri == r2.uri
    # Don't check body

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["match_on"] = ["custom"]
    return config

def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_matcher("custom", _custom_matcher)
```

### Custom Scrubbers

To add additional scrubbing for proprietary secrets:

```python
def remove_proprietary_header(request: Any) -> Any:
    """Remove custom header."""
    request.headers.pop("X-Proprietary-Token", None)
    return request

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["before_record_request"] = remove_proprietary_header
    return config
```

## Best Practices

1. **Always scrub cassettes**: Use `before_record_request` and `before_record_response` to redact all sensitive data. Inspect cassettes before committing.

2. **Commit cassettes**: Cassettes are part of your test suite. Commit them to git so others can run tests without live API access.

3. **Keep cassettes small**: Avoid recording huge responses (e.g., embeddings of long documents). If needed, use manual cassette creation with canned data.

4. **Use markers correctly**:
   - `@pytest.mark.vcr` for cassette-backed tests
   - `@pytest.mark.scheduled` for live-only tests
   - No marker for offline tests

5. **Test against multiple models/versions**: Use `@pytest.mark.parametrize` to test multiple configurations and generate separate cassettes for each.

6. **Document cassette dependencies**: If a cassette depends on specific model behavior (e.g., a reasoning model), document it in the test.

7. **Refresh cassettes when code changes**: If you modify the test (new prompt, new parameters), refresh the cassette with `--record-mode=new_episodes`.

8. **Review cassette diffs**: When committing cassette updates, review the diff in your PR to ensure no secrets leaked.
