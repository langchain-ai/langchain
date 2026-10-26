---
type: "Testing & QA"
title: "Integration Testing: Standard Test Suites & Providers"
description: "How LangChain integration tests work: VCR cassettes for API recording/playback, standard test suite inheritance for providers, CI/CD integration, and test markers for live vs. recorded testing."
tags: [integration-tests, vcr, cassettes, api-testing, pytest, ci-cd, model-testing, standard-tests, test-markers]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-20T08:25:57.412Z
sources:
  - id: openwiki-source-a12f520c2572fe032a111e3b
    resource: repo://.github/workflows/_compile_integration_test.yml
  - id: openwiki-source-d8a8900818f4abab719bd1b7
    resource: repo://.github/workflows/_test_vcr.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-ff76574b014ac8b5c67560a6
    resource: repo://libs/langchain_v1/tests/integration_tests/chat_models/test_base.py
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
  - id: openwiki-source-13cd70e2dd4252d8b05ac9c1
    resource: repo://libs/standard-tests/langchain_tests/integration_tests/__init__.py
  - id: openwiki-source-f7c26a6473517c7b426ef023
    resource: repo://libs/standard-tests/langchain_tests/integration_tests/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-10-20T08:25:57.412Z" }
---

## Overview

Integration tests in LangChain differ fundamentally from unit tests: they call real model APIs (OpenAI, Anthropic, Groq, etc.) and require network access and valid credentials. To make these tests reproducible and CI-friendly without exposing API keys or relying on external services, LangChain uses **VCR cassettes**—recorded HTTP interactions that are replayed during test execution.

All LangChain partner packages inherit a standard suite of integration tests via `ChatModelIntegrationTests` and other base classes. This ensures consistent test coverage across providers while allowing provider-specific customization via property overrides and method extension.

This page covers:
- VCR cassette recording, playback, and security (scrubbing secrets)
- Test markers (`@pytest.mark.vcr`, `@pytest.mark.scheduled`) for classification
- Standard test suite inheritance and customization points
- CI/CD integration: compilation checks, VCR playback in PR checks, scheduled live testing
- Environment setup, cassette management, and best practices

## Test Markers: Classification and Execution

LangChain uses pytest markers to classify integration tests into three categories, each running in different CI contexts:

### `@pytest.mark.vcr` — Cassette-Backed Tests

Tests marked with `@pytest.mark.vcr` use **VCR cassettes** for playback. No API key is needed in CI; VCR intercepts HTTP calls and replays pre-recorded responses.

```python
@pytest.mark.vcr
def test_chat_invoke():
    """Test chat invoke using a cassette."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke("Hello")
    assert response.content
```

**When cassettes are used**:
- **In CI**: Always. VCR runs with `--record-mode=none` (playback-only).
- **Locally**: On first run, if a cassette doesn't exist, `--record-mode=once` will record it (if API key is present). On subsequent runs, VCR replays the cassette.

**Advantages**:
- Tests are **fast** (no network latency)
- Tests are **deterministic** (same response every time)
- **No credentials** needed in CI
- Cassettes are **committed to git** so all developers can run tests

### `@pytest.mark.scheduled` — Live API Tests

Tests marked with `@pytest.mark.scheduled` run **live API calls** without cassettes. These tests are skipped in regular CI but run in scheduled daily workflows with real credentials.

```python
@pytest.mark.scheduled
def test_chat_streaming_live():
    """Test streaming behavior with a live API call."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke("Tell me a joke")
    assert response.content
```

**When these run**:
- **In PR checks**: Skipped
- **In scheduled daily workflows** (`.github/workflows/integration_tests.yml`): Run with real API credentials
- **On-demand**: Can be manually triggered from GitHub Actions UI

**Purpose**: Validates that the model integrations work against live APIs. Catches issues not covered by cassettes (API changes, model behavior changes, breaking SDK changes).

### No Marker — Unit-Like Tests

Tests without `@pytest.mark.vcr` or `@pytest.mark.scheduled` run as offline unit tests. They don't require API keys and don't make network calls.

```python
def test_model_name_set():
    """Test model initialization without API calls."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.model_name == "gpt-4o-mini"
```

## VCR Cassettes: Recording, Playback, and Security

### The VCR Pattern

**VCR** (Video Cassette Recorder, implemented by the `vcrpy` library) records HTTP interactions—requests and responses—the first time a test runs with a live API. On subsequent runs, VCR replays the recorded cassette instead of making live network calls.

#### Recording Phase (Once, by Developers)

When a new integration test is written or an existing test's behavior changes:

1. Run the test with a valid API key in `.env` and `--record-mode=once` (default) or `--record-mode=new_episodes`
2. VCR intercepts all HTTP calls your test makes
3. **Before recording**, sensitive headers and tokens in request/response bodies are **scrubbed** (redacted)
4. The cassette is saved as a YAML file, optionally compressed as `.yaml.gz`
5. The cassette is committed to the repository

```bash
cd libs/partners/openai
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

#### Playback Phase (Always, in CI and Local Development)

When running tests:

1. VCR loads the cassette file
2. Instead of making real API calls, VCR intercepts your HTTP client and returns the pre-recorded response
3. No API key is needed; no credentials are exposed
4. Tests are fast and deterministic

### Cassette Location and Naming

Cassettes are stored relative to test modules in a `cassettes/` subdirectory:

```
libs/partners/openai/tests/
  integration_tests/
    chat_models/
      test_base.py
      test_base_standard.py
      cassettes/
        test_base/
          test_invoke.yaml.gz
          test_chat_openai.yaml.gz
        test_base_standard/
          TestOpenAIStandard.test_invoke.yaml.gz
          TestOpenAIStandard.test_stream[model0].yaml.gz
```

For parameterized tests, cassette names include parameter identifiers:

```
test_invoke[gpt-4o-mini].yaml.gz
test_invoke[gpt-4o].yaml.gz
```

The `vcr_cassette_dir` fixture (from `conftest.py`) computes the correct directory per test module:

```python
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = Path(request.module.__file__)
    return str(module.parent / "cassettes" / module.stem)
```

### Security: Scrubbing Sensitive Data

VCR cassettes are **committed to git** and **visible to the world** (in public repositories). To prevent accidental exposure of API keys, JWTs, OAuth tokens, and other secrets, LangChain uses a multi-layer scrubbing pipeline.

#### What Gets Redacted

**Request and Response Headers** (configured in `conftest.py`):

```python
_EXTRA_HEADERS = [
    ("authorization", "PLACEHOLDER"),
    ("openai-organization", "PLACEHOLDER"),
    ("cookie", "PLACEHOLDER"),
    ("set-cookie", "PLACEHOLDER"),
    ("chatgpt-account-id", "PLACEHOLDER"),
]
```

All headers are replaced with `**REDACTED**` during recording.

**Request and Response Bodies** (OAuth secret fields):

```python
_OAUTH_SECRET_FIELDS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "code",
    "device_code",
    "client_secret",
    "code_verifier",
})
```

Redaction is applied with specialized handlers:

- **JSON bodies**: Recursively walks the parsed JSON tree and redacts any field matching `_OAUTH_SECRET_FIELDS`
- **Form-encoded bodies**: Splits on `&` and redacts matching keys
- **JWT patterns**: Uses regex to detect and redact JWT-shaped strings anywhere in the body

**Binary payloads** (PNG, JPEG, PDF, audio, etc.) are **skipped**—their magic bytes are detected and the scrubbing stack is bypassed for performance. JWTs and OAuth secrets are ASCII, so binary bodies cannot carry them.

#### Scrubbing Configuration in conftest.py

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
    config = base_vcr_config()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    return config
```

## Environment Setup: API Keys and .env

Integration tests require valid API credentials to record cassettes (once) and for running live integration tests in scheduled or on-demand scenarios. Store credentials in a `.env` file at the package's test directory root.

### .env File Location and Format

For partner packages (e.g., OpenAI), the structure is:

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

## Writing Integration Tests

### Basic Structure with VCR

Integration tests live in `tests/integration_tests/` (mirroring the layout of unit tests):

```
libs/partners/openai/tests/
  unit_tests/
  integration_tests/
    chat_models/
      __init__.py
      conftest.py
      test_base.py
      test_base_standard.py
      cassettes/
```

A minimal integration test using `@pytest.mark.vcr`:

```python
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@pytest.mark.vcr
def test_chat_openai_invoke():
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert response.content
```

The `@pytest.mark.vcr` marker tells pytest-recording to automatically:

1. Look for a cassette file named `test_chat_openai_invoke.yaml.gz` in the module's `cassettes/` directory
2. Use VCR to intercept HTTP calls
3. Replay the cassette if it exists; record a new one if it doesn't (or if `--record-mode=new_episodes` is passed)

### Parameterized Integration Tests

When testing the same scenario against multiple models or configurations, use `@pytest.mark.parametrize`:

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
    assert chunks[-1].content
```

The cassette records all the streaming HTTP chunks, and VCR replays them in order during playback. Async tests work identically—VCR intercepts the underlying `httpx`/`requests` library for both sync and async.

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
- Network access

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

When test code changes (e.g., you change the prompt, the model, or the parameters), the cassette may become stale. VCR will attempt to match the new request body against the old cassette; if there's a mismatch, it will fail with a `CannotOverwriteExistingCassetteException` error (in playback mode).

To refresh:

```bash
cd libs/partners/openai
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

This overwrites the cassette with the new interaction. Re-verify the cassette was scrubbed correctly, then commit.

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

## Running Integration Tests

### Full Integration Test Suite (Live)

Run all integration tests against live APIs (requires credentials):

```bash
cd libs/partners/openai
make integration_tests  # Runs full suite with live API calls
```

This invokes:

```bash
uv run --group test --group test_integration pytest -v --tb=short tests/integration_tests/
```

It also smoke-tests a single live request on each supported SDK major version (e.g., OpenAI SDK 2.x and 3.x).

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

Only tests marked `@pytest.mark.scheduled` run in these workflows. See [CI/CD Workflows](repo:///openwiki/ci-workflows.md) for details.

## Standard Test Suite Inheritance

LangChain partner packages inherit a standard suite of integration tests via `ChatModelIntegrationTests` from `langchain_tests.integration_tests`. This allows all chat model providers to run a common set of tests without duplicating test code.

### How It Works

Each partner package defines a test class that inherits from `ChatModelIntegrationTests`:

```python
# libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_openai import ChatOpenAI

class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def has_tool_calling(self) -> bool:
        return True
```

The base class (`ChatModelIntegrationTests`) implements dozens of test methods that exercise standard chat model capabilities: `invoke`, `ainvoke`, `stream`, `astream`, tool calling, structured output, image inputs, and more. Each test method is automatically marked with `@pytest.mark.vcr` so cassettes are recorded and replayed.

### Required Properties

Partner implementations **must** override two properties:

- **`chat_model_class`**: The model class to instantiate (e.g., `ChatOpenAI`)
- **`chat_model_params`**: Initialization parameters for the model (e.g., `{"model": "gpt-4o-mini"}`)

### Feature Declaration Properties

Partner implementations can declare model capabilities via properties. These control which tests run and how they are executed:

- **`has_tool_calling`**: Whether the model supports tool/function calling
- **`has_tool_choice`**: Whether the model supports forcing a tool call via `tool_choice` parameter
- **`has_structured_output`**: Whether the model supports structured output
- **`supports_image_inputs`**: Whether the model accepts image inputs (base64)
- **`supports_image_urls`**: Whether the model accepts image URLs
- **`supports_json_mode`**: Whether the model supports `method='json_mode'` in structured output
- **`supports_pdf_inputs`**: Whether the model accepts PDF file inputs
- **`structured_output_kwargs`**: Dict of additional kwargs for `with_structured_output()` (e.g., `{"method": "json_schema"}`)

### Customization: Overriding Test Methods

Partners can override individual test methods to provide provider-specific behavior:

```python
class TestOpenAIStandard(ChatModelIntegrationTests):
    # ... properties ...
    
    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        """Override to test cache read with specific model features."""
        llm = ChatOpenAI(model="gpt-4o-mini", stream_usage=True)
        return llm.invoke("Cached prompt", ...)
```

### VCR Integration with Standard Tests

Standard test methods inherit VCR integration automatically:

1. Each test method is pre-marked with `@pytest.mark.vcr`
2. Cassettes are generated automatically with names like `TestOpenAIStandard.test_invoke.yaml.gz`, `TestOpenAIStandard.test_invoke[model0].yaml.gz` (for parameterized variants)
3. Tests run in CI via `make test_vcr --record-mode=none` to validate cassettes are up-to-date
4. When a partner updates a test (e.g., adds a new model to test), new cassettes are recorded with `--record-mode=new_episodes`

This approach ensures that all LangChain partners maintain consistent test coverage and that changes to the standard test suite (new test methods, new capabilities) are automatically picked up by all partners.

## VCR Configuration Details

### conftest.py Setup

Each package's `conftest.py` configures VCR globally:

```python
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    # Base config includes:
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

For APIs with non-deterministic bodies (e.g., timestamps in the request), you can customize matchers or use `allow_playback_repeats` to reuse the same response for multiple request variants.

OpenAI tests use a custom `json_body` matcher:

```python
config["match_on"] = ["json_body"]  # Only match on JSON-parsed body (ignores whitespace/key order)
```

This allows cassettes to match even if JSON key order differs. The matcher is registered in `pytest_recording_configure`:

```python
def _json_body_matcher(r1: Any, r2: Any) -> None:
    """Match request bodies as parsed JSON, ignoring key order and tool description whitespace."""
    # Parses both bodies as JSON and compares parsed trees
    # Normalizes tool descriptions to handle Python version differences
    # Fails gracefully for non-JSON bodies
```

## CI/CD Integration

### Workflow Structure

The LangChain CI system runs integration tests at multiple points:

1. **Pull request checks** (`check_diffs.yml`): Compiles integration tests without running them, then runs `make test_vcr` for cassette playback tests
2. **Scheduled integration tests** (`integration_tests.yml`): Daily remote API testing with live credentials against partner libraries

#### Integration Test Compilation (`_compile_integration_test.yml`)

Validates that integration tests compile without syntax or import errors:

```bash
uv run pytest -m compile tests/integration_tests
```

This is a fast check that catches import errors and syntax issues before attempting to run tests. It runs in `check_diffs.yml` for every package with changed files.

#### VCR Cassette Tests (`_test_vcr.yml`)

Runs cassette-backed integration tests in playback-only mode:

```bash
uv run pytest --record-mode=none -m vcr tests/integration_tests/
```

**Key features**:
- No API keys needed (fake key like `sk-fake` is set)
- Cassettes are **never recorded** or overwritten in CI
- If a cassette is missing, the test **fails immediately**
- If request bodies don't match, the test **fails immediately**
- Tests run in parallel for speed (`-n auto`)

This catches stale cassettes caused by test code changes without attempting live calls.

#### Scheduled Integration Tests (`integration_tests.yml`)

Runs with `make integration_tests` daily or on-demand:

```bash
uv run --group test --group test_integration pytest -v --tb=short tests/integration_tests/
```

**Key features**:
- Runs with real API credentials from secrets
- Runs only tests marked `@pytest.mark.scheduled`
- Tests marked `@pytest.mark.vcr` are skipped (use cassettes in PR checks instead)
- Smoke-tests each supported SDK major version

### Cassette Validation in CI

The `test_vcr` job runs with `--record-mode=none`, which means:

- Cassettes are **never recorded** or overwritten in CI
- If a cassette is missing, the test **fails immediately**
- If request bodies don't match, the test **fails immediately**

This catches stale cassettes caused by test code changes without attempting live calls.

## Advanced Configuration

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

### Disabling Specific Tests in CI

Some providers (e.g., Codex, audio models) require special handling in CI. Use `pytest_collection_modifyitems` hooks to skip tests at collection time:

```python
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip Codex tests in CI unless they are replaying VCR cassettes."""
    if not os.getenv("CI") or _vcr_record_mode(config) == "none":
        return
    # Skip Codex tests unless in playback mode
```

And use autouse fixtures to stub dependencies:

```python
@pytest.fixture(autouse=True)
def _fake_codex_oauth_token(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub OAuth token reads for Codex VCR tests."""
    if "codex" not in request.module.__name__:
        return
    # Monkeypatch OAuth token provider with in-memory fake token
```

## Best Practices

1. **Always scrub cassettes**: Use `before_record_request` and `before_record_response` to redact all sensitive data. Inspect cassettes before committing.

2. **Commit cassettes**: Cassettes are part of your test suite. Commit them to git so others can run tests without live API access.

3. **Keep cassettes small**: Avoid recording huge responses (e.g., embeddings of long documents). If needed, use manual cassette creation with canned data.

4. **Use markers correctly**:
   - `@pytest.mark.vcr` for cassette-backed tests (run in PR checks)
   - `@pytest.mark.scheduled` for live-only tests (run in scheduled workflows)
   - No marker for offline tests

5. **Test against multiple models/versions**: Use `@pytest.mark.parametrize` to test multiple configurations and generate separate cassettes for each.

6. **Document cassette dependencies**: If a cassette depends on specific model behavior (e.g., a reasoning model), document it in the test.

7. **Refresh cassettes when code changes**: If you modify the test (new prompt, new parameters), refresh the cassette with `--record-mode=new_episodes`.

8. **Review cassette diffs**: When committing cassette updates, review the diff in your PR to ensure no secrets leaked.

9. **Inherit standard tests**: Use `ChatModelIntegrationTests` and similar base classes from `langchain_tests.integration_tests` to ensure consistent coverage across providers.

10. **Override properties, not methods**: Prefer declaring capabilities via properties (`supports_*`, `has_*`) rather than overriding test methods, unless provider-specific logic is needed.
