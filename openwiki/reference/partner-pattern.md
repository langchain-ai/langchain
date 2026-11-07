---
type: Reference
title: Partner Pattern Reference
description: Technical reference for LangChain partner package structure, core abstractions, message conversion, tool calling, structured output, streaming, error handling, and model profiles.
tags: [partner-pattern, chat-models, integration, llm, streaming, tool-calling, structured-output]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

# Partner Pattern Reference

This page provides technical reference documentation for the partner pattern—the standardized structure for implementing LLM provider integrations in LangChain. See [the integration guide](/openwiki/partner-pattern.md) for step-by-step instructions.

## Package Structure Overview

Partner packages live under `/libs/partners/provider_name/` and follow a consistent structure:

```
/libs/partners/provider_name/
├── langchain_provider_name/              # Python package (public API)
│   ├── __init__.py                       # Public exports
│   ├── _version.py                       # Version constant
│   ├── chat_models/
│   │   ├── __init__.py
│   │   └── base.py                       # ChatProviderModel class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── _profiles.py                  # Auto-generated model capability profiles
│   │   └── profile_augmentations.toml    # Provider-specific profile overrides
│   ├── py.typed                          # PEP 561 type-checking marker
│   └── (optional) other modules
├── tests/
│   ├── unit_tests/
│   │   └── chat_models/
│   │       ├── test_standard.py          # Inherits ChatModelUnitTests
│   │       └── test_*.py                 # Provider-specific tests
│   └── integration_tests/
│       └── chat_models/
│           ├── test_standard.py          # Inherits ChatModelIntegrationTests
│           └── test_*.py                 # Provider-specific tests
├── pyproject.toml
├── Makefile
├── README.md
└── LICENSE (MIT)
```

## Core Module Exports

The main `langchain_provider_name/__init__.py` must export:

**Required:**
- `ChatProviderModel` (or similar name): The primary chat model class
- `__version__`: Package version string

**Commonly exported:**
- `AzureChatProviderModel` or `ChatProviderModelAzure`: Azure/cloud variants
- Provider-specific embedding classes (e.g., `ProviderEmbeddings`)
- Legacy LLM completion classes (for backward compatibility)
- Utility classes (e.g., `Decisions`, custom tools)
- Exceptions or error-handling utilities

Example from `langchain_openai/__init__.py`:
```python
from langchain_openai._version import __version__
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_openai.llms import AzureOpenAI, OpenAI

__all__ = [
    "ChatOpenAI",
    "AzureChatOpenAI",
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "OpenAI",
    "AzureOpenAI",
    "__version__",
]
```

## BaseChatModel Contract

All provider implementations must inherit from `BaseChatModel` defined in `repo://libs/core/langchain_core/language_models/chat_models.py`. This class defines the contract for invoking and streaming chat models.

### Core Methods and Properties

| Symbol | Type | Required | Purpose |
|--------|------|----------|---------|
| `_generate()` | method | **Required** | Synchronous generation: transform `list[BaseMessage]` → `ChatResult` |
| `_stream()` | method | Optional | Streaming: yield `ChatGenerationChunk` objects containing partial tokens |
| `_agenerate()` | method | Optional | Async variant of `_generate()` |
| `_astream()` | method | Optional | Async variant of `_stream()` |
| `_llm_type` | property | **Required** | Return provider identifier string (e.g., `"openai"`) |
| `_identifying_params` | property | Optional | Return dict of init params for tracing/logging |
| `bind_tools()` | method | Optional | Create wrapper that binds tools to the model |
| `with_structured_output()` | method | Optional | Create wrapper that structures model output |

### Required Initialization Fields

Every chat model class must accept via Pydantic:
- `model` (or `model_name`, `deployment_name`): Model identifier as a string
- `api_key` (optional): Authentication credential (prefer environment variable loading)
- Sampling parameters: `temperature`, `max_tokens`, `top_p`, etc. (provider-specific)

Use Pydantic field validators or the `from_env()` utility to load credentials from environment variables to ensure they follow `PROVIDER_API_KEY` convention.

### Message Generation Contract

The `_generate()` method must:

1. Accept `messages: list[BaseMessage]`, `stop: list[str] | None`, `run_manager: CallbackManagerForLLMRun | None`, `**kwargs`
2. Convert LangChain messages to provider format (see [Message Conversion](#message-conversion))
3. Call the provider API
4. Extract the response text and metadata
5. Wrap in `AIMessage` with `response_metadata` dict
6. Return `ChatResult(generations=[ChatGeneration(message=ai_msg)], llm_output=...)`

Key fields in the returned `ChatResult`:
- `generations`: List of `ChatGeneration` objects, each wrapping an `AIMessage`
- `llm_output`: Dict containing `usage` (token counts), `finish_reason`, or other metadata
- `run_id`: (auto-populated by base class)

Example structure:
```python
def _generate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult:
    # Convert messages
    provider_messages = self._convert_to_provider_format(messages)
    
    # Call API
    response = self.client.chat.completions.create(
        model=self.model,
        messages=provider_messages,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        stop=stop,
        **kwargs,
    )
    
    # Extract content and metadata
    content = response.choices[0].message.content
    message = AIMessage(
        content=content,
        response_metadata={
            "model": response.model,
            "stop_reason": response.choices[0].finish_reason,
        },
    )
    
    # Return ChatResult
    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output={"usage": response.usage.model_dump()},
    )
```

## Message Conversion

LangChain uses a unified message abstraction (`BaseMessage` and subclasses) that must be converted to each provider's native format.

### LangChain Message Types

- `HumanMessage`: User input
- `AIMessage`: Model output (may contain `tool_calls`, `invalid_tool_calls`)
- `SystemMessage`: System instructions
- `ToolMessage`: Response to a tool call (contains `tool_call_id`)
- `FunctionMessage`: Legacy function call response
- `ChatMessage`: Generic message with custom role

### Conversion Pattern

Providers implement a `_convert_messages_to_provider_format()` helper that:

1. Iterates over each message
2. Maps role names: `"human"` → provider's user role, `"ai"` → provider's assistant role, etc.
3. Handles message content blocks (text, images, tool calls, etc.)
4. Returns a list of provider-native message dicts

Example (OpenAI):
```python
def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    message_dict = {
        "content": _format_message_content(message.content)
    }
    if isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) 
                for tc in message.tool_calls
            ]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    # ... etc for other message types
    return message_dict
```

### Response Conversion

After receiving a provider response, extract and convert:

```python
# Parse provider response
response_message = response.choices[0].message

# Convert to LangChain AIMessage
tool_calls = []
if response_message.tool_calls:
    tool_calls = [
        parse_tool_call(tc, return_id=True)  # Converts to ToolCall
        for tc in response_message.tool_calls
    ]

ai_message = AIMessage(
    content=response_message.content,
    tool_calls=tool_calls,
    response_metadata={
        "model": response.model,
        "finish_reason": response_message.finish_reason,
    },
)
```

## Streaming Implementation

The `_stream()` method must yield `ChatGenerationChunk` objects incrementally:

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    # Request streaming response
    response_stream = self.client.chat.completions.create(
        model=self.model,
        messages=self._convert_messages_to_provider_format(messages),
        stream=True,
        **kwargs,
    )
    
    # Yield chunks as they arrive
    for event in response_stream:
        delta = event.choices[0].delta
        if delta.content:
            # Create chunk message
            chunk_message = AIMessageChunk(content=delta.content)
            chunk = ChatGenerationChunk(message=chunk_message)
            
            # Notify run_manager of new token
            if run_manager:
                run_manager.on_llm_new_token(delta.content, chunk=chunk)
            
            yield chunk
```

**Key contract:**
- Yield immediately as tokens arrive (not buffered)
- Use `AIMessageChunk` (not `AIMessage`) in chunks
- Call `run_manager.on_llm_new_token()` for each token if run_manager exists
- Accumulate tokens across chunks to construct the full message

## Tool Calling Integration

Tool calling allows the model to decide when to call tools and with what arguments.

### bind_tools() Pattern

Implement `bind_tools()` to create a model variant that can call tools:

```python
def bind_tools(
    self,
    tools: Sequence[BaseTool | dict] | Callable,
    *,
    tool_choice: str | dict | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
    """Create a model that can call tools.
    
    Args:
        tools: List of BaseTool, tool dicts, or a callable returning either
        tool_choice: Which tool to prefer (optional)
        **kwargs: Additional bind parameters
    
    Returns:
        A Runnable that calls the model with tools enabled
    """
    # 1. Convert tools to provider format
    provider_tools = [convert_to_openai_tool(tool) for tool in tools]
    
    # 2. Create a bound version with tools in model_kwargs
    bound_kwargs = {
        "tools": provider_tools,
        **({"tool_choice": tool_choice} if tool_choice else {}),
    }
    
    # 3. Return self.bind(**bound_kwargs)
    return self.bind(**bound_kwargs)
```

### Tool Call Extraction

When a model response includes tool calls, extract them into `AIMessage.tool_calls`:

```python
tool_calls = []
for raw_tool_call in response.tool_calls:
    tool_call = ToolCall(
        id=raw_tool_call["id"],
        name=raw_tool_call["function"]["name"],
        args=json.loads(raw_tool_call["function"]["arguments"]),
    )
    tool_calls.append(tool_call)

ai_message = AIMessage(
    content=response.message.content,
    tool_calls=tool_calls,
    response_metadata={"tool_calls": response.tool_calls},
)
```

### Tool Call Streaming

Some providers stream tool call chunks (progressive JSON parsing). Handle with `tool_call_chunk()`:

```python
from langchain_core.messages.tool import tool_call_chunk

# For each partial tool call event:
chunk = tool_call_chunk(
    index=tool_call_index,
    id=tool_id,
    name=tool_name_partial,
    args=args_partial_json,
)
ai_chunk = AIMessageChunk(tool_calls=[chunk])
yield ChatGenerationChunk(message=ai_chunk)
```

## Structured Output Support

Structured output allows the model to produce output conforming to a provided schema (JSON Schema, Pydantic, TypedDict, or dict).

### with_structured_output() Pattern

Implement `with_structured_output()` to create a schema-aware model wrapper:

```python
def with_structured_output(
    self,
    schema: type[T] | dict,
    *,
    method: Literal["json_mode", "json_schema", "tool"] = "json_schema",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, T | dict]:
    """Create a model that outputs structured data matching the schema.
    
    Args:
        schema: Pydantic model, TypedDict, dict (JSON Schema), or type
        method: Schema enforcement method (provider-dependent)
        include_raw: If True, return {"parsed": T, "raw": AIMessage}
        **kwargs: Additional parameters
    
    Returns:
        A Runnable that enforces the schema on output
    """
    # 1. Validate and normalize schema
    schema_dict = convert_to_json_schema(schema)
    
    # 2. Create wrapped version with response_format
    def process_output(message: AIMessage) -> T | dict:
        # Parse structured content and validate against schema
        return json.loads(message.content)
    
    # 3. Bind response_format and wrap output parser
    return self.bind(response_format=schema_dict) | RunnableLambda(
        process_output
    )
```

### Method Support by Provider

Providers support different schema enforcement methods:

| Method | Description | Example Provider |
|--------|-------------|------------------|
| `json_mode` | Model outputs valid JSON | OpenAI |
| `json_schema` | Strict JSON Schema validation | OpenAI's Strict mode |
| `tool` | Constrain output to a single tool call | Anthropic |
| `language_model` | Use model to parse output | Fallback |

## Async Support

Implement async variants for non-blocking execution:

```python
async def _agenerate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Async variant of _generate."""
    payload = self._build_payload(messages, stop=stop, **kwargs)
    response = await self._async_client.chat.completions.create(**payload)
    return self._create_chat_result(response)

async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]:
    """Async variant of _stream."""
    payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)
    async for chunk in await self._async_client.chat.completions.create(**payload):
        yield self._process_stream_chunk(chunk, run_manager)
```

**Key patterns:**
- Use `async def` for native async (not executor-based)
- Maintain both sync and async httpx clients if possible
- Preserve run_manager callback semantics in async context

## Error Handling and Mapping

Provider exceptions must be mapped to LangChain's standard exception hierarchy:

```python
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    ContextOverflowError,
)
```

### Mapping Pattern

```python
def _handle_api_error(e: ProviderException) -> None:
    """Map provider exception to LangChain exception."""
    if isinstance(e, ProviderAuthError):
        raise ModelAuthenticationError(str(e)) from e
    elif isinstance(e, ProviderRateLimitError):
        raise ModelRateLimitError(str(e)) from e
    elif isinstance(e, ProviderTimeoutError):
        raise ModelTimeoutError(str(e)) from e
    elif isinstance(e, ProviderNotFoundError):
        raise ModelNotFoundError(str(e)) from e
    elif isinstance(e, ProviderPermissionError):
        raise ModelPermissionDeniedError(str(e)) from e
    elif e.status_code == 400 or e.status_code == 422:
        raise ModelInvalidRequestError(str(e)) from e
    elif hasattr(e, "status_code") and e.status_code >= 500:
        raise ModelAPIError(str(e)) from e
    else:
        raise ModelAPIError(str(e)) from e
```

### Context Overflow Handling

Detect context window exhaustion:

```python
if "context_length_exceeded" in str(e).lower() or (
    hasattr(e, "code") and e.code == "context_length_exceeded"
):
    raise ContextOverflowError(str(e)) from e
```

## Model Profiles

Model profiles publish capability metadata used by LangChain and applications.

### Profile Data Structure

Profiles are defined as dictionaries with capability flags:

```python
{
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "release_date": "2024-04-09",
        "last_updated": "2024-12-17",
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "tool_calling": True,
        "structured_output": True,
        "temperature": True,
    }
}
```

### Profile Location and Generation

- **Source:** Profiles are auto-generated from the [models.dev](https://github.com/sst/models.dev) project
- **Location:** `langchain_provider_name/data/_profiles.py` (generated)
- **Augmentation:** `langchain_provider_name/data/profile_augmentations.toml` (manual overrides)

Example augmentation file (`profile_augmentations.toml`):
```toml
[anthropic."claude-opus-4-7"]
# Anthropic-specific capabilities
mid_conversation_tools = true
pdf_tool_message = true
```

### Profile Resolution

During model initialization, the base class automatically loads profiles:

```python
def _resolve_model_profile(self) -> ModelProfile | None:
    """Return the default model profile, or None if unavailable."""
    return _get_default_model_profile(self.model)

def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Load profile from _profiles.py."""
    default = _MODEL_PROFILES.get(model_name)
    if default:
        return default.copy()
    return {}
```

The profile is accessible at `model.profile` after initialization.

## Standard Tests

Partner packages must include test suites that inherit from `langchain-tests`:

### Unit Tests

```python
# tests/unit_tests/chat_models/test_standard.py
from langchain_tests.unit_tests import ChatModelUnitTests
from langchain_provider_name import ChatProviderModel

class TestProviderStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @property
    def chat_model_params(self) -> dict:
        return {
            "api_key": "test-key",
            "model": "provider-model-name",
        }
    
    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        # (env_vars, expected_from_env, expected_model_params)
        return (
            {"PROVIDER_API_KEY": "test-key"},
            {},
            {"openai_api_key": "test-key"},
        )
```

### Integration Tests

```python
# tests/integration_tests/chat_models/test_standard.py
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_provider_name import ChatProviderModel

class TestProviderStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @property
    def chat_model_params(self) -> dict:
        return {"model": "provider-model-name"}
```

### Configurable Fixtures

The standard test suites offer optional fixtures:

| Fixture | Type | Purpose |
|---------|------|---------|
| `chat_model_class` | property | **Required** – The model class to test |
| `chat_model_params` | property | Constructor kwargs (API key, model name, etc.) |
| `chat_model_has_tool_calling` | property | Whether `bind_tools()` is supported (auto-detected) |
| `chat_model_has_structured_output` | property | Whether `with_structured_output()` is supported (auto-detected) |
| `init_from_env_params` | property | Tuple of (env_vars, from_env_dict, model_kwargs) |

## Client Management

Partner implementations typically maintain both sync and async HTTP clients:

```python
class ChatProviderModel(BaseChatModel):
    client: ProviderClient = Field(default=None, exclude=True)
    async_client: ProviderAsyncClient = Field(default=None, exclude=True)
    
    @model_validator(mode="after")
    def _initialize_clients(self) -> Self:
        """Initialize sync/async clients on first use."""
        if self.client is None:
            self.client = ProviderClient(api_key=self.api_key)
        if self.async_client is None:
            self.async_client = ProviderAsyncClient(api_key=self.api_key)
        return self
    
    def _ensure_sync_client_available(self) -> None:
        """Lazily initialize sync client."""
        if self.client is None:
            self.client = ProviderClient(api_key=self.api_key)
```

## Usage in init_chat_model Factory

Partner models are registered in the `init_chat_model()` factory for unified initialization:

```python
from langchain import init_chat_model

# Fixed model
model = init_chat_model("gpt-4", model_provider="openai", temperature=0.5)

# Configurable model
model = init_chat_model(
    configurable_fields=["model", "temperature"],
    model_provider="openai",
)
result = model.invoke(
    "What is 2+2?",
    config={"configurable": {"model": "gpt-4", "temperature": 0}}
)
```

The factory discovers and instantiates models via provider package entry points and introspection.

## Configuration and Environment Variables

Partners should support environment variable configuration for credentials:

**Standard naming:**
- `{PROVIDER}_API_KEY` – API authentication
- `{PROVIDER}_API_BASE` – Custom API endpoint
- `{PROVIDER}_ORG_ID` – Organization/account identifier
- `{PROVIDER}_PROXY` – HTTP proxy URL

**Implementation pattern:**
```python
from langchain_core.utils.utils import from_env

class ChatProviderModel(BaseChatModel):
    api_key: str | None = Field(
        default_factory=from_env("PROVIDER_API_KEY", default=None)
    )
    api_base: str | None = Field(
        default_factory=from_env("PROVIDER_API_BASE", default=None)
    )
    organization: str | None = Field(
        default_factory=from_env("PROVIDER_ORG_ID", default=None)
    )
```

## Testing Patterns and Fixtures

### Mocking Provider Responses

Use `unittest.mock` to simulate provider responses without API calls:

```python
from unittest.mock import MagicMock, patch

def test_chat_model():
    with patch.object(ChatProviderModel, "client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "provider-model"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        
        mock_client.chat.completions.create.return_value = mock_response
        
        model = ChatProviderModel(api_key="test", model="test-model")
        result = model.invoke("Hi")
        
        assert result.content == "Hello!"
```

### Environment Variable Testing

Use pytest fixtures to manage environment variables:

```python
import pytest
import os

@pytest.fixture
def provider_env_vars(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key-123")
    monkeypatch.setenv("PROVIDER_API_BASE", "https://test.provider.com")
    yield
    # Auto-cleanup by pytest

def test_init_from_env(provider_env_vars):
    model = ChatProviderModel(model="test-model")
    assert model.api_key == "test-key-123"
```

## Identifying Parameters and Tracing

The `_identifying_params` property exposes model configuration for LangSmith and debug logging:

```python
@property
def _identifying_params(self) -> dict[str, Any]:
    """Return identifying parameters for tracing."""
    return {
        "model": self.model,
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "top_p": self.top_p,
    }
```

This dict appears in traces and is used to identify distinct model configurations.

## Gateway Metadata Handling

Some providers integrate with LangSmith's gateway service, which returns metadata in response headers:

```python
from langchain_core.utils._gateway import (
    GATEWAY_METADATA_RESPONSE_KEY,
    _parse_gateway_metadata,
)

def _handle_response(response: Response, generation_info: dict[str, Any]) -> None:
    """Extract and store gateway metadata."""
    headers = getattr(response, "headers", None)
    if headers:
        gateway_metadata = _parse_gateway_metadata(headers)
        if gateway_metadata is not None:
            generation_info[GATEWAY_METADATA_RESPONSE_KEY] = gateway_metadata
```

---

For step-by-step implementation guidance, see [Adding a New Chat Model Provider](/openwiki/partner-pattern.md).
