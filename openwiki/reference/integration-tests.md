---
type: "Reference"
title: "Edit cassette.yaml, check for redactions"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-11-21T08:14:55.070Z
sources:
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-ed8ac34fd0216d557729038e
    resource: repo://libs/partners/openai/tests/conftest.py
  - id: openwiki-source-ce605960b642234837131282
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/conftest.py
  - id: openwiki-source-711811fb8df6b32329091d23
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
  - id: openwiki-source-db02c1dda8563ab005cd9d62
    resource: repo://libs/standard-tests/langchain_tests/conftest.py
  - id: openwiki-source-f7c26a6473517c7b426ef023
    resource: repo://libs/standard-tests/langchain_tests/integration_tests/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-11-21T08:14:55.070Z" }
---


## Overview

LangChain integration tests use pytest fixtures and configuration helpers to manage VCR cassette recording, classify tests, and standardize chat model implementations across partner libraries. This reference documents the key APIs for configuring and extending integration tests.

## Standard Test Base Classes

### ChatModelIntegrationTests

The `ChatModelIntegrationTests` class (from `langchain_tests.integration_tests`) provides 70+ standard tests for chat model implementations, covering streaming, structured output, tool calling, multimodality, and advanced features.

**Location**: `repo://libs/standard-tests/langchain_tests/integration_tests/chat_models.py`

#### Required Properties

Subclasses **must** implement these properties:

**`chat_model_class`**: Returns the chat model class to test.

```python
@property
def chat_model_class(self) -> type[BaseChatModel]:
    return ChatOpenAI
```

**`chat_model_params`**: Returns initialization parameters as a dict.

```python
@property
def chat_model_params(self) -> dict:
    return {"model": "gpt-4o-mini", "temperature": 0}
```

#### Optional Feature Properties

Subclasses control feature coverage via boolean properties:

| Property | Default | Purpose |
|----------|---------|---------|
| `has_tool_calling` | Auto-detected from `bind_tools` override | Whether model supports function/tool calling |
| `has_tool_choice` | Auto-detected from `bind_tools` signature | Whether model supports forcing specific tools via `tool_choice` parameter |
| `has_structured_output` | Auto-detected from `with_structured_output` override | Whether model supports structured output/JSON schema |
| `supports_json_mode` | `False` | Whether `with_structured_output(method='json_mode')` is supported |
| `supports_image_inputs` | `False` | Whether model accepts image content blocks |
| `supports_image_urls` | `False` | Whether model accepts image URLs in messages |
| `supports_pdf_inputs` | `False` | Whether model accepts PDF file content blocks |
| `supports_pdf_tool_message` | `False` | Whether model accepts `ToolMessage` with PDF content |
| `supports_audio_inputs` | `False` | Whether model accepts audio content blocks |
| `returns_usage_metadata` | `True` | Whether model returns token usage in `AIMessage.usage_metadata` |
| `supports_anthropic_inputs` | `False` | Whether model accepts Anthropic-style content blocks (tool_use, tool_result) |

**Example**:

```python
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

#### VCR Test Configuration

**`enable_vcr_tests`**: Boolean property controlling whether to enable VCR-backed tests (defaults to `False`).

When `True`, enables select tests that use VCR cassette recording, such as benchmarking and deterministic output tests. Partner libraries must configure VCR security (credential redaction) before enabling.

```python
@property
def enable_vcr_tests(self) -> bool:
    return True
```

#### Optional Method Overrides

For feature-specific test customization:

**`invoke_with_cache_read_input(stream: bool = False) -> AIMessage`**: Custom implementation for cache read input tests (for models supporting prompt caching).

**`invoke_with_reasoning_output(stream: bool = False) -> AIMessage`**: Custom implementation for reasoning tests (for reasoning models like OpenAI o1).

**`structured_output_kwargs() -> dict`**: Dict property specifying additional kwargs for `with_structured_output()` calls, such as `method='json_schema'`.

## VCR Cassette System

### Configuration Fixtures

#### `base_vcr_config()`

Function (not a fixture) that returns default VCR configuration applied to all cassettes.

**Location**: `repo://libs/standard-tests/langchain_tests/conftest.py#L130-L143`

**Returns**: Dict with keys:

| Key | Value | Purpose |
|-----|-------|---------|
| `record_mode` | `"once"` | Record only if cassette missing; replay existing |
| `filter_headers` | List of tuples | Headers to redact (base: `authorization`, `x-api-key`, `api-key`) |
| `match_on` | `["method", "uri", "body"]` | Request fields to match for cassette lookup |
| `allow_playback_repeats` | `True` | Allow replaying same response for duplicate requests |
| `decode_compressed_response` | `True` | Decompress gzipped API responses before matching |
| `cassette_library_dir` | `"tests/cassettes"` | Directory for cassette storage |
| `path_transformer` | `VCR.ensure_suffix(".yaml")` | Auto-append filename suffix |

**Usage in partner conftest**:

```python
from langchain_tests.conftest import base_vcr_config

@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    config = base_vcr_config()
    # Extend with partner-specific filtering
    config["filter_headers"].extend([
        ("x-openai-client-user-agent", "PLACEHOLDER"),
    ])
    return config
```

#### `vcr_config` pytest Fixture

Partner libraries implement this session-scoped fixture in `tests/conftest.py` to extend default configuration with security measures and custom matchers.

**Example** (OpenAI):

```python
@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    config = base_vcr_config()
    config["match_on"] = [
        m if m != "body" else "json_body"
        for m in config.get("match_on", [])
        if m != "uri"
    ]
    config.setdefault("filter_headers", []).extend([
        ("openai-organization", "PLACEHOLDER"),
        ("user-agent", "PLACEHOLDER"),
    ])
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config
```

### Serialization and Persistence

#### `CustomSerializer`

YAML + gzip serializer for cassettes. Secure (uses `yaml.safe_load`), handles binary content blocks (audio, PDF, images).

**Location**: `repo://libs/standard-tests/langchain_tests/conftest.py#L47-L82`

**Methods**:

**`serialize(cassette_dict: dict) -> bytes`**: Converts cassette to YAML, compresses with gzip, handles `io.BytesIO` objects.

**`deserialize(data: bytes) -> dict`**: Decompresses gzip, parses YAML, reconstructs `vcr.request.Request` objects.

**Registration**:

```python
def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    vcr.register_serializer("yaml.gz", CustomSerializer())
```

#### `CustomPersister`

Custom file I/O for VCR cassettes using `CustomSerializer`.

**Location**: `repo://libs/standard-tests/langchain_tests/conftest.py#L85-L117`

**Methods**:

**`load_cassette(cassette_path, serializer) -> tuple[list, list]`**: Loads cassette file, deserializes, returns (requests, responses).

**`save_cassette(cassette_path, cassette_dict, serializer) -> None`**: Serializes and writes cassette to file, creating parent directories as needed.

**Registration**:

```python
def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
```

### Request Matching and Normalization

Partner libraries often register custom matchers to handle API-specific request normalization.

**Example** (OpenAI's `_json_body_matcher`):

```python
def _json_body_matcher(r1: Any, r2: Any) -> None:
    """Match request bodies as parsed JSON, ignoring key order."""
    b1 = r1.body or b""
    b2 = r2.body or b""
    # Normalize tool descriptions for Python version differences
    j1 = _normalize_tool_descriptions(json.loads(b1))
    j2 = _normalize_tool_descriptions(json.loads(b2))
    assert j1 == j2

def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    vcr.register_matcher("json_body", _json_body_matcher)
```

**Registration context**: The `pytest_recording_configure(config, vcr)` hook (called by pytest-recording plugin) is the standard entry point for registering custom serializers, persisters, and matchers.

## Security and Credential Redaction

### Base Header Filtering

Applied by default via `base_vcr_config()`:

| Header | Replacement |
|--------|-------------|
| `authorization` | `PLACEHOLDER` |
| `x-api-key` | `PLACEHOLDER` |
| `api-key` | `PLACEHOLDER` |

### Partner-Specific Redaction Functions

Partners implement request/response hooks to redact sensitive data before cassette storage.

**Example** (OpenAI's OAuth and secret scrubbing):

```python
_OAUTH_SECRET_FIELDS = frozenset({
    "access_token", "refresh_token", "id_token", "code",
    "code_verifier", "device_code", "client_secret"
})

def remove_request_headers(request: Any) -> Any:
    """Redact all headers and URI."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    request.body = _scrub_oauth_secrets(request.body)
    return request

def remove_response_headers(response: dict) -> dict:
    """Redact all response headers and body secrets."""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    response["body"]["string"] = _scrub_oauth_secrets(response["body"]["string"])
    return response

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    return config
```

**Binary content detection**: OpenAI's implementation detects binary magic byte prefixes (PNG, JPEG, PDF, WAV, MP3, etc.) and skips regex scanning on confirmed binary bodies for performance.

## Test Classification via Pytest Markers

### `@pytest.mark.vcr`

Marks a test to use VCR cassette playback (recorded HTTP interactions). Tests run with `--record-mode=none` in CI.

```python
@pytest.mark.vcr
def test_invoke() -> None:
    """Uses cassette; no live API call."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke("Hello")
    assert isinstance(result, AIMessage)
```

**CI behavior**:
- **VCR mode** (`make test_vcr`): `--record-mode=none` — cassette must exist, no re-recording
- **Full integration** (`make integration_tests`): `--record-mode=once` — records if missing, replays existing

### `@pytest.mark.scheduled`

Marks a test to run against live APIs in daily GitHub Actions workflows. Requires valid credentials as GitHub Secrets.

```python
@pytest.mark.scheduled
def test_live_api() -> None:
    """Runs only in scheduled CI with credentials."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke("What is 2+2?")
    assert "4" in result.content
```

### Unmarked Tests

Tests without markers execute as offline unit tests. Run with `--disable-socket` to prevent accidental API calls.

```python
def test_parsing() -> None:
    """Offline test; no networking needed."""
    tool = Tool(name="calc", func=lambda x: x+1)
    assert tool.name == "calc"
```

## Integration Test Execution

### Makefile Targets

Partner libraries define three test targets in their Makefile:

**`make test`**: Unit tests with socket disabled (no networking).

```makefile
test:
    uv run --group test pytest --disable-socket
```

**`make test_vcr`**: VCR cassette-only playback tests (no credentials required).

```makefile
test_vcr:
    uv run --group test pytest --record-mode=none -m vcr
```

**`make integration_tests`**: Full integration suite (cassettes + live tests if credentials available).

```makefile
integration_tests:
    uv run --group test --group test_integration pytest \
        tests/integration_tests/
```

### Environment Variables

**For local testing**:

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API credentials | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API credentials | `sk-ant-...` |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) | `ls__...` |
| `LANGCHAIN_TESTS_USER_AGENT` | User-Agent for external resource fetches (audio, etc.) | `MyBot/1.0` |

**For CI**:

GitHub Actions injects all repository secrets into the `integration-tests` job environment (intentional for simplicity). The workflow also sets LangSmith metadata tags automatically:

```bash
LANGSMITH_TRACING: "true"
LANGSMITH_PROJECT: "scheduled-testing-py"
LANGSMITH_TAGS: "github-actions,libs/partners/openai,python-3.11,sha-<commit-hash>"
```

## Fixture Patterns in Partner Conftest

### Environment Cleanup Fixtures

**OpenAI's `_clear_openai_base_url_env`**:

```python
@pytest.fixture(autouse=True)
def _clear_openai_base_url_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strip ambient OpenAI endpoint env vars for VCR tests only."""
    if request.node.get_closest_marker("vcr") is None:
        return
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
```

**Purpose**: Prevents VCR cassette mismatch when developers have custom gateway URLs in their shell environment. Scoped to `@pytest.mark.vcr` tests only; live tests run through the gateway.

### Module-Level Hooks

**OpenAI's `pytest_collection_modifyitems`**:

```python
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip Codex tests in CI unless replaying VCR cassettes."""
    if not os.getenv("CI") or _vcr_record_mode(config) == "none":
        return
    for item in items:
        if "codex" in item.module.__name__:
            item.add_marker(pytest.mark.skip(reason="Codex requires VCR"))
```

**Purpose**: Collection-time skipping ensures cassette file access is deferred; avoids trying to load missing cassettes before fixtures can stub authentication.

### Monkeypatching Authentication

**OpenAI's `_fake_codex_oauth_token`**:

```python
@pytest.fixture(autouse=True)
def _fake_codex_oauth_token(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub OAuth token provider for Codex VCR tests."""
    if "codex" not in request.module.__name__:
        return
    if _vcr_record_mode(request.config) != "none":
        return
    
    fake_token = chatgpt_oauth._ChatGPTToken(
        access_token="vcr-fake-access-token",
        refresh_token="vcr-fake-refresh-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        account_id="vcr-fake-account-id",
    )
    monkeypatch.setattr(
        chatgpt_oauth._FileChatGPTOAuthTokenProvider,
        "get_token",
        lambda self: fake_token,
    )
```

**Purpose**: Allows Codex tests to replay cassettes without reading disk-based OAuth credentials. Enables CI testing of authentication flows without secrets.

## Best Practices

### Cassette Organization

- **One cassette per test method**: Naming is automatic (module + method name)
- **Deterministic output**: Avoid timestamps, UUIDs, or random values in assertions — they break cassette replay
- **Cassette naming**: Use unique test names; avoid generic names that collide across modules

### Recording Cassettes

**Initial recording**:

```bash
cd libs/partners/openai
export OPENAI_API_KEY=sk-...
make integration_tests
```

**Re-recording after test changes**:

```bash
pytest tests/integration_tests/chat_models/test_base.py --record-mode=new_episodes
```

**Manual inspection**:

```bash
gzip -d tests/cassettes/test_invoke.yaml.gz
# Edit cassette.yaml, check for redactions
gzip cassette.yaml
```

### Credential Rotation and Leaks

- **If credentials are rotated**: Re-record cassettes or manually update `PLACEHOLDER` values
- **Pre-commit scanning**: Cassette diffs are visible in git; review for unredacted tokens before merge
- **Fallback scanning**: Partners may implement post-recording leak detection scripts

### Debugging VCR Mismatch

**Symptoms**: "APIConnectionError" despite cassette existing, or "CassetteNotFoundError"

**Causes**:
1. **Request body mismatch**: JSON key order, whitespace, or normalized fields (Python version differences)
2. **URI mismatch**: Ambient `OPENAI_BASE_URL` or `OPENAI_API_BASE` env var points to gateway
3. **Header mismatch**: Missing or extra headers after redaction

**Solutions**:
- Decompress cassette: `gzip -d cassette.yaml.gz`
- Inspect recorded request/response bodies
- Clear ambient endpoint env vars: `unset OPENAI_BASE_URL OPENAI_API_BASE`
- Register custom matcher if JSON normalization needed

## Configuration Examples

### Minimal Partner Integration

```python
# tests/conftest.py
import pytest
from langchain_tests.conftest import base_vcr_config

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    return base_vcr_config()
```

```python
# tests/integration_tests/chat_models/test_standard.py
from langchain_tests.integration_tests import ChatModelIntegrationTests
from my_partner_package import MyChatModel

class TestMyChatModelStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self):
        return MyChatModel
    
    @property
    def chat_model_params(self) -> dict:
        return {"model": "my-model", "api_key": "..."}
```

### Advanced with Custom Security

```python
# tests/conftest.py
import pytest
from langchain_tests.conftest import (
    base_vcr_config,
    CustomSerializer,
    CustomPersister,
)
from vcr import VCR

def remove_response_headers(response: dict) -> dict:
    response["headers"] = {}
    return response

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["filter_headers"].extend([
        ("x-partner-token", "PLACEHOLDER"),
        ("x-request-id", "PLACEHOLDER"),
    ])
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config

def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
```
