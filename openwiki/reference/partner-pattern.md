---
type: Integration Reference
title: Partner Pattern Reference
description: Quick reference documentation for partner integration package structure, ChatModel implementation requirements, and integration checklist templates for LangChain provider packages.
tags: [partner, integration, reference, chat-models, package-structure, template]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-a12395e96060f062d8ec1ca2
    resource: repo://libs/partners/anthropic/langchain_anthropic/__init__.py
  - id: openwiki-source-7de1ace618efbdfd8bacb5cb
    resource: repo://libs/partners/anthropic/langchain_anthropic/chat_models.py
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-8641a971af4f11b852966d77
    resource: repo://libs/partners/openai/langchain_openai/chat_models/__init__.py
  - id: openwiki-source-74e5bef080f1af7da12371cf
    resource: repo://libs/partners/openai/langchain_openai/data/_profiles.py
  - id: openwiki-source-82d936a0b444a10f97753ac8
    resource: repo://libs/partners/openai/langchain_openai/data/profile_augmentations.toml
  - id: openwiki-source-3953aa29dbaaf738e6efc09d
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py
  - id: openwiki-source-4065d8687d2708ee826c7818
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

# Partner Pattern Reference

This reference provides templates, checklists, and structured guidance for implementing partner integrations in LangChain. For detailed step-by-step instructions and conceptual explanation, see [Adding a New Chat Model Provider](/openwiki/partner-pattern.md).

## Partner Package Structure Template

Standard directory layout for a provider integration:

```
/libs/partners/provider_name/
├── langchain_provider_name/              # Python package name
│   ├── __init__.py                       # Exports public API
│   ├── _version.py                       # Version string
│   ├── chat_models/
│   │   ├── __init__.py                   # Export ChatProviderModel
│   │   └── base.py                       # ChatProviderModel implementation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── _profiles.py                  # Auto-generated model profiles
│   │   └── profile_augmentations.toml    # Provider-specific overrides
│   ├── py.typed                          # PEP 561 type marker
│   └── [embeddings/]                     # Optional: other integrations
├── tests/
│   ├── unit_tests/
│   │   ├── __init__.py
│   │   └── chat_models/
│   │       ├── test_standard.py          # Standard unit tests (required)
│   │       └── test_*.py                 # Provider-specific tests
│   └── integration_tests/
│       ├── __init__.py
│       └── chat_models/
│           ├── test_standard.py          # Standard integration tests (required)
│           └── test_*.py                 # Advanced mode tests (e.g., Responses API)
├── pyproject.toml
├── Makefile
├── README.md
├── LICENSE
└── uv.lock
```

## pyproject.toml Configuration Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langchain-provider-name"
description = "LangChain integrations for Provider Name"
requires-python = ">=3.10.0,<4.0.0"
version = "0.1.0"

dependencies = [
    "langchain-core>=1.7.5,<2.0.0",
    "provider-sdk>=X.Y.Z",           # Pin minimum version
    "certifi>=2024.6.2",
]

[dependency-groups]
test = [
    "pytest>=9.0.3,<10.0.0",
    "pytest-asyncio>=1.3.0,<2.0.0",
    "langchain>=1.0.0,<2.0.0",
    "langchain-tests>=1.1.9,<2.0.0",
]
lint = ["ruff>=0.13.1,<0.17.0"]
typing = ["mypy>=2.1.0,<2.2.0"]

[tool.uv.sources]
langchain-core = { path = "../../core", editable = true }
langchain-tests = { path = "../../standard-tests", editable = true }
langchain = { path = "../../langchain_v1", editable = true }
```

## Module Exports Template (`__init__.py`)

```python
"""LangChain integrations for Provider Name."""

from langchain_provider_name._version import __version__
from langchain_provider_name.chat_models import ChatProviderModel

__all__ = [
    "ChatProviderModel",
    "__version__",
]
```

Reference: [repo://libs/partners/openai/langchain_openai/__init__.py](repo://libs/partners/openai/langchain_openai/__init__.py)

## ChatModel Implementation Checklist

### Required Methods

- [ ] **`_llm_type` property**: Return provider identifier (e.g., `"openai"`)
- [ ] **`_generate(messages, stop, run_manager, **kwargs)`**: Synchronous generation
  - Input: `list[BaseMessage]`
  - Output: `ChatResult` with `ChatGeneration` containing `AIMessage`
- [ ] **`_stream(messages, stop, run_manager, **kwargs)`**: Streaming generation
  - Yields: `ChatGenerationChunk` with `AIMessageChunk`
- [ ] **`_agenerate(...)`**: Async variant of `_generate`
- [ ] **`_astream(...)`**: Async streaming variant

### Required Attributes (from Pydantic model)

- [ ] `model: str` - Model identifier
- [ ] `api_key: str | None = None` - API credential (or load from environment)
- [ ] `temperature: float = 1.0` - Temperature parameter (provider-specific range)
- [ ] `max_tokens: int | None = None` - Max output tokens

### Optional Methods

- [ ] **`get_num_tokens(text)`**: Token counting (override if provider has native tokenizer)
- [ ] **`get_num_tokens_from_messages(messages, tools)`**: Accurate token accounting
- [ ] **`bind_tools(tools, tool_choice)`**: Tool/function calling support
- [ ] **`with_structured_output(schema, ...)`**: Structured output support

### Key Responsibilities

| Responsibility | Implementation |
|---|---|
| **Message conversion** | Implement `_format_messages_for_provider()` to convert LangChain `BaseMessage` → provider API format |
| **Response parsing** | Extract content, tool calls, usage from provider response → `AIMessage` |
| **Error handling** | Map provider exceptions → LangChain unified exception hierarchy |
| **Streaming tokens** | Yield chunks and call `run_manager.on_llm_new_token()` |
| **Async support** | Implement async variants using `async def` and `await` |
| **Model profiles** | Auto-generate in `data/_profiles.py` from provider's model list |

Reference: [repo://libs/partners/openai/langchain_openai/chat_models/base.py](repo://libs/partners/openai/langchain_openai/chat_models/base.py)

## Message Conversion Reference

### Input Direction: LangChain → Provider Format

Map message types:

| LangChain | Provider Role | Notes |
|---|---|---|
| `HumanMessage` | `"user"` | User input |
| `AIMessage` | `"assistant"` | Model output (may include tool_calls) |
| `SystemMessage` | `"system"` | System prompt (some providers treat as user) |
| `ToolMessage` | `"tool"` (or embed in content) | Tool execution result |

Handle content blocks:

```python
# LangChain unified format:
[
    {"type": "text", "text": "..."},
    {"type": "image", "source": {...}},
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
]

# Convert to provider format (e.g., OpenAI):
{
    "type": "text",
    "text": "..."
}
# images, tool_use become separate content blocks or parameters
```

### Output Direction: Provider Response → LangChain

Required extraction:

1. **Content**: Plain text or list of content blocks
2. **Tool calls**: Parse into `list[ToolCall]` with `id`, `name`, `args`
3. **Usage metadata**: Normalize token counts (`input_tokens`, `output_tokens`)
4. **Response metadata**: Provider-specific data (model name, finish reason, latency, etc.)

```python
AIMessage(
    content="...",  # str or list[dict]
    tool_calls=[ToolCall(id="...", name="...", args={...})],
    usage_metadata=UsageMetadata(input_tokens=X, output_tokens=Y),
    response_metadata={"model": "...", "finish_reason": "...", ...},
)
```

## Testing Integration Checklist

### Unit Tests (`test_standard.py`)

```python
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

class TestProviderModel(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @property
    def chat_model_params(self) -> dict:
        return {"model": "test-model", "api_key": "test-key"}
```

- [ ] Inherits from `ChatModelUnitTests`
- [ ] Provides `chat_model_class` property
- [ ] Provides `chat_model_params` fixture
- [ ] Auto-detects `has_tool_calling`, `has_structured_output`

### Integration Tests (`test_standard.py`)

```python
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

class TestProviderModelIntegration(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @pytest.fixture
    def chat_model_params(self) -> dict:
        # Load from environment variables
        return {"model": "prod-model"}
```

- [ ] Inherits from `ChatModelIntegrationTests`
- [ ] Provides `chat_model_class` fixture
- [ ] Provides `chat_model_params` fixture (loads live API credentials from environment)
- [ ] Only runs when `PROVIDER_NAME_API_KEY` is set

### Standard Test Coverage

The standard test suite validates:

- [ ] Invoke and stream (both sync and async)
- [ ] All message types (Human, AI, System, Tool)
- [ ] Content blocks (text, images, tool_use)
- [ ] Tool calling (if `bind_tools()` implemented)
- [ ] Structured output (if `with_structured_output()` implemented)
- [ ] Callbacks and token counting
- [ ] Model profile existence and validity

Reference: [repo://libs/standard-tests/README.md](repo://libs/standard-tests/README.md)

## Model Profiles Checklist

### Auto-Generated Profiles (`_profiles.py`)

- [ ] File generated from `models.dev` provider registry
- [ ] Contains model metadata: name, context window, capabilities
- [ ] Regenerated when provider releases new models

### Profile Augmentations (`profile_augmentations.toml`)

Add provider-specific overrides for capabilities not auto-detected:

```toml
[model_profiles]

[model_profiles."model-id"]
name = "Human-readable model name"
context_window = 8192
max_output_tokens = 4096
supports_vision = true
supports_tool_calling = true
supports_structured_output = true
cost_input_per_1k = 0.003
cost_output_per_1k = 0.006
```

- [ ] Define `name` (human-readable)
- [ ] Set `context_window` (tokens)
- [ ] Set capability flags: `supports_vision`, `supports_tool_calling`, `supports_structured_output`
- [ ] Set pricing (cost per 1K tokens)

## Error Handling Mapping

Map provider exceptions to LangChain unified exceptions:

```python
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelConnectionError,
)

try:
    response = self._client.chat.create(...)
except provider_sdk.AuthenticationError as e:
    raise ModelAuthenticationError(str(e)) from e
except provider_sdk.RateLimitError as e:
    raise ModelRateLimitError(str(e)) from e
except provider_sdk.APIConnectionError as e:
    raise ModelConnectionError(str(e)) from e
except provider_sdk.APITimeoutError as e:
    raise ModelTimeoutError(str(e)) from e
except provider_sdk.APIError as e:
    raise ModelAPIError(str(e)) from e
```

Reference: [repo://libs/partners/openai/langchain_openai/chat_models/base.py#L54-L64](repo://libs/partners/openai/langchain_openai/chat_models/base.py#L54-L64)

## Integration Checklist (Complete)

### Step 1: Package Setup
- [ ] Create `/libs/partners/provider_name/` directory
- [ ] Create Python package under `langchain_provider_name/`
- [ ] Add `__init__.py` with public exports
- [ ] Create `_version.py` with version constant
- [ ] Add `py.typed` marker (PEP 561)
- [ ] Create `pyproject.toml` with dependencies
- [ ] Create `tests/` directory structure

### Step 2: ChatModel Implementation
- [ ] Create `chat_models/base.py` with `ChatProviderModel` class
- [ ] Inherit from `BaseChatModel`
- [ ] Implement `_generate()` and `_stream()` (sync)
- [ ] Implement `_agenerate()` and `_astream()` (async)
- [ ] Set `_llm_type` property
- [ ] Add required attributes: `model`, `api_key`, `temperature`, `max_tokens`
- [ ] Implement message conversion (`_format_messages_for_provider()`)
- [ ] Implement response parsing (provider response → `AIMessage`)
- [ ] Implement error handling (map exceptions)

### Step 3: Tool Calling (if applicable)
- [ ] Implement `bind_tools(tools, tool_choice, **kwargs)`
- [ ] Convert LangChain tools → provider function_calling format
- [ ] Parse tool calls from provider response
- [ ] Create `ToolCall` objects with `id`, `name`, `args`
- [ ] Handle invalid tool calls

### Step 4: Structured Output (if applicable)
- [ ] Implement `with_structured_output(schema, ...)`
- [ ] Convert LangChain schema → provider format (JSON Schema, etc.)
- [ ] Parse structured response back to Python dict/Pydantic model
- [ ] Validate schema compatibility

### Step 5: Model Profiles
- [ ] Auto-generate `data/_profiles.py` from provider model registry
- [ ] Create `data/profile_augmentations.toml`
- [ ] Define all supported models with capabilities
- [ ] Set pricing information
- [ ] Set context windows and token limits

### Step 6: Testing
- [ ] Create `tests/unit_tests/chat_models/test_standard.py`
  - [ ] Inherit from `ChatModelUnitTests`
  - [ ] Provide `chat_model_class` property
  - [ ] Provide `chat_model_params` fixture
- [ ] Create `tests/integration_tests/chat_models/test_standard.py`
  - [ ] Inherit from `ChatModelIntegrationTests`
  - [ ] Load API credentials from environment
- [ ] Run `make test` successfully
- [ ] Run `make lint` (ruff) successfully
- [ ] Run `make type-check` (mypy) successfully

### Step 7: Documentation & Build
- [ ] Create `README.md` with usage examples
- [ ] Create `Makefile` with standard targets (test, lint, type-check)
- [ ] Add `LICENSE` (MIT)
- [ ] Create `uv.lock` by running `uv lock`
- [ ] Test installation: `pip install -e .`

### Step 8: Registration (Optional)
- [ ] Add provider to `init_chat_model` factory registry if needed
- [ ] Update monorepo documentation

## Common Patterns and Gotchas

### API Client Initialization

Store provider client instances lazily or in `__init__`:

```python
class ChatProviderModel(BaseChatModel):
    model: str
    api_key: str | None = None
    
    @property
    def _client(self):
        """Lazily initialize provider client."""
        if not hasattr(self, "_client_cache"):
            self._client_cache = ProviderClient(api_key=self.api_key)
        return self._client_cache
```

### Environment Variable Loading

Use Pydantic's field validation:

```python
from pydantic import Field

class ChatProviderModel(BaseChatModel):
    api_key: str | None = Field(
        default=None,
        alias="provider_api_key"
    )
    
    def __init__(self, **kwargs):
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("PROVIDER_API_KEY")
        super().__init__(**kwargs)
```

### Streaming Token Callbacks

Always notify `run_manager` of new tokens:

```python
def _stream(self, messages, stop, run_manager, **kwargs):
    for event in self._client.stream(...):
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=event.delta))
        if run_manager:
            run_manager.on_llm_new_token(event.delta, chunk=chunk)
        yield chunk
```

### Handling Tool Calls in Streaming

Accumulate tool call deltas before creating complete `ToolCall` objects:

```python
tool_call_buffer = {}
for event in stream:
    if event.choices[0].delta.tool_calls:
        for tc in event.choices[0].delta.tool_calls:
            tool_id = tc.id
            if tool_id not in tool_call_buffer:
                tool_call_buffer[tool_id] = {...}
            # Accumulate delta
            tool_call_buffer[tool_id]["args"] += tc.args_delta
# Convert accumulated to ToolCall objects after stream complete
```

### Context Window Padding

Some providers reserve tokens for response. Add safety margin when checking limits:

```python
if num_tokens > self.model_context_window - 4096:  # Reserve 4K for response
    raise ValueError("Input exceeds context window with safety margin")
```

## References

- **Full Integration Guide**: [Adding a New Chat Model Provider](/openwiki/partner-pattern.md)
- **ChatModel Base Class**: [repo://libs/core/langchain_core/language_models/chat_models.py](repo://libs/core/langchain_core/language_models/chat_models.py)
- **OpenAI Reference Implementation**: [repo://libs/partners/openai/langchain_openai/chat_models/base.py](repo://libs/partners/openai/langchain_openai/chat_models/base.py)
- **Anthropic Reference Implementation**: [repo://libs/partners/anthropic/langchain_anthropic/chat_models.py](repo://libs/partners/anthropic/langchain_anthropic/chat_models.py)
- **Standard Test Suite**: [repo://libs/standard-tests/README.md](repo://libs/standard-tests/README.md)
- **Message Types**: [repo://libs/core/langchain_core/messages/base.py](repo://libs/core/langchain_core/messages/base.py)
