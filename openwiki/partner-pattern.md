---
type: Integration Pattern
title: Adding a New Chat Model Provider
description: Step-by-step guide to integrate a new LLM provider into LangChain's monorepo, including package structure, ChatModel implementation, streaming, function calling, structured output, and standard tests. Covers message conversion, error handling, model profiles, and optional advanced API modes like Responses API.
tags: [chat-models, provider-integration, llm, function-calling, structured-output, streaming]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-09T08:26:28.144Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-e0b95eafb4bbd52f491c8cee
    resource: repo://libs/model-profiles/README.md
  - id: openwiki-source-7de1ace618efbdfd8bacb5cb
    resource: repo://libs/partners/anthropic/langchain_anthropic/chat_models.py
  - id: openwiki-source-f416dbc063b474398e38ff3c
    resource: repo://libs/partners/anthropic/langchain_anthropic/data/_profiles.py
  - id: openwiki-source-d14c2b8060843a8a89b74733
    resource: repo://libs/partners/anthropic/langchain_anthropic/data/profile_augmentations.toml
  - id: openwiki-source-8641a971af4f11b852966d77
    resource: repo://libs/partners/openai/langchain_openai/chat_models/__init__.py
  - id: openwiki-source-3bc725a9a39d534be6f46d18
    resource: repo://libs/partners/openai/langchain_openai/chat_models/_compat.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-e14c13faa5b908b1c95925d3
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_responses_api.py
  - id: openwiki-source-3953aa29dbaaf738e6efc09d
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py
  - id: openwiki-source-4065d8687d2708ee826c7818
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py
  - id: openwiki-source-025cad4ae99967890152b7e0
    resource: repo://libs/standard-tests/README.md
generated: { by: "openwiki/0.5.0", at: "2026-09-09T08:26:28.144Z" }
---

## Overview

This guide documents the integration pattern for adding a new chat model provider (LLM service) to the LangChain monorepo. A **provider** represents an LLM service (e.g., OpenAI, Anthropic, Mistral) with its own client library, model lineup, and API conventions. Each provider integration lives in its own package under `/libs/partners/` and provides a `ChatModel` subclass bridging LangChain's message abstraction to the provider's API.

The integration process involves:

1. **Package structure**: Creating `/libs/partners/provider_name/` with Python module, tests, and configuration
2. **ChatModel implementation**: Inheriting `BaseChatModel` and implementing generation/streaming methods
3. **Message conversion**: Translating between LangChain's unified message format and provider-specific API schemas
4. **Provider registration**: Adding the provider to the built-in `init_chat_model` factory registry
5. **Model profiles**: Publishing capability data (context window, tool calling, structured output, etc.)
6. **Standard tests**: Inheriting unit and integration test suites to validate the implementation

## 1. Package Structure

Create a new directory under `/libs/partners/` with the provider name in lowercase, using hyphens as needed:

```
/libs/partners/provider_name/
├── langchain_provider_name/              # Python package
│   ├── __init__.py                       # Exports: ChatProviderModel, version
│   ├── _version.py                       # Version constant
│   ├── chat_models/                      # Chat model implementation
│   │   ├── __init__.py
│   │   └── base.py                       # ChatProviderModel class
│   ├── data/                             # Model profiles and augmentations
│   │   ├── __init__.py
│   │   ├── _profiles.py                  # Auto-generated profiles from models.dev
│   │   └── profile_augmentations.toml    # Provider-specific overrides
│   ├── py.typed                          # PEP 561 marker for type checking
│   └── middleware/                       # (Optional) Custom middleware
├── tests/
│   ├── unit_tests/
│   │   ├── __init__.py
│   │   └── chat_models/
│   │       ├── test_standard.py          # Standard unit test suite
│   │       └── test_*.py                 # Provider-specific unit tests
│   └── integration_tests/
│       ├── __init__.py
│       └── chat_models/
│           ├── test_standard.py          # Standard integration test suite
│           └── test_*.py                 # Provider-specific integration tests
├── pyproject.toml                        # Package metadata and dependencies
├── Makefile                              # Common build/test targets
├── README.md                             # User-facing documentation
├── LICENSE                               # MIT license
└── uv.lock                               # Locked dependency versions

```

### Package Metadata (pyproject.toml)

Key configuration for a provider package (reference: `repo://libs/partners/openai/pyproject.toml#L1-L76`):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langchain-provider-name"              # PyPI package name
description = "LangChain integration for Provider Name"
requires-python = ">=3.10.0,<4.0.0"
version = "0.1.0"

dependencies = [
    "langchain-core>=1.6.0,<2.0.0",           # Required: base LangChain
    "provider-client-library>=2.45.0,<4.0.0", # Provider's own SDK (pinned version)
    "certifi>=2024.6.2",                      # SSL certificates
]

[dependency-groups]
test = [
    "pytest>=9.0.3,<10.0.0",
    "pytest-asyncio>=1.3.0,<2.0.0",
    "langchain>=1.0.0,<2.0.0",
    "langchain-tests>=1.1.9,<2.0.0",          # Standard test suite with version constraint
]
lint = ["ruff>=0.13.1,<0.17.0"]
dev = []
test_integration = [
    # Additional integration test dependencies (API clients, test fixtures)
]
typing = [
    "mypy>=2.1.0,<2.2.0",
]

[tool.uv.sources]
langchain-core = { path = "../../core", editable = true }
langchain-tests = { path = "../../standard-tests", editable = true }
langchain = { path = "../../langchain_v1", editable = true }

[tool.uv]
constraint-dependencies = ["urllib3>=2.6.3", "pygments>=2.20.0"]
```

**Key patterns:**
- **Lock provider SDK versions** to prevent breaking API changes (e.g., `openai>=2.45.0,<4.0.0`)
- **Add version constraints to all test dependencies** for reproducibility
- **Use `[tool.uv.sources]`** to reference local LangChain packages in monorepo
- **Separate integration test dependencies** in a dedicated group (only needed in CI)

## 2. ChatModel Implementation

### BaseChatModel and Core Requirements

All provider implementations must inherit from **`BaseChatModel`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L284-L2400`), which defines the contract for invoking and streaming chat models.

**Core responsibilities** (location: `repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1-L150`):

1. **Inherit `BaseChatModel`** with type parameter `[AIMessage]`
2. **Implement `_generate` method** (required sync): Transform messages into `ChatResult` with `ChatGeneration` objects wrapping `AIMessage` output
3. **Implement `_stream` method** (optional for streaming support): Yield `ChatGenerationChunk` objects containing `AIMessageChunk` with incremental tokens
4. **Implement `_agenerate` method** (async variant of `_generate`) or `_astream` method (async variant of `_stream`)
5. **Set `_llm_type` property**: Return the provider identifier string for identification

### Minimal ChatModel Template

```python
"""Provider chat model integration."""

from typing import Any, Iterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

class ChatProviderModel(BaseChatModel):
    """Chat model for Provider Name."""
    
    model: str                           # Model identifier (e.g., "model-123")
    api_key: str | None = None           # Provider API key
    temperature: float = 1.0             # Temperature parameter
    max_tokens: int | None = None        # Max output tokens
    
    @property
    def _llm_type(self) -> str:
        """Return provider identifier."""
        return "provider_name"
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion synchronously.
        
        Args:
            messages: Conversation history and user input
            stop: Optional stop sequences
            run_manager: Callback manager for logging
            **kwargs: Additional provider-specific parameters
        
        Returns:
            ChatResult containing one or more ChatGeneration objects
        """
        # 1. Convert LangChain messages to provider format
        provider_messages = self._convert_messages_to_provider_format(messages)
        
        # 2. Build request payload
        payload = {
            "model": self.model,
            "messages": provider_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **kwargs,
        }
        
        # 3. Call provider API
        response = self._client.chat.completions.create(**payload)
        
        # 4. Extract and convert response to AIMessage
        content = response.choices[0].message.content
        message = AIMessage(
            content=content,
            response_metadata={
                "model": response.model,
                "stop_reason": response.choices[0].finish_reason,
            },
        )
        
        # 5. Return ChatResult with generation info
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"usage": response.usage.model_dump()} if response.usage else None,
        )
    
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion tokens in real time.
        
        This method is called when `stream=True` or streaming callbacks are attached.
        
        Args:
            messages: Conversation history
            stop: Optional stop sequences
            run_manager: Callback manager for per-token callbacks
            **kwargs: Additional parameters
        
        Yields:
            ChatGenerationChunk objects containing AIMessageChunk with partial content
        """
        # 1. Build streaming request
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_provider_format(messages),
            "stream": True,
            "temperature": self.temperature,
            "stop": stop,
            **kwargs,
        }
        
        # 2. Stream from API
        accumulated_content = ""
        for event in self._client.chat.completions.create(**payload):
            # 3. Extract token delta
            delta = event.choices[0].delta
            if delta.content:
                accumulated_content += delta.content
                
                # 4. Yield chunk with incremental token
                chunk_message = AIMessageChunk(content=delta.content)
                chunk = ChatGenerationChunk(message=chunk_message)
                
                # 5. Notify run_manager of new token
                if run_manager:
                    run_manager.on_llm_new_token(delta.content, chunk=chunk)
                
                yield chunk
    
    def _convert_messages_to_provider_format(
        self, messages: list[BaseMessage]
    ) -> list[dict[str, Any]]:
        """Translate LangChain messages to provider API format.
        
        Provider APIs often use a different schema for messages (e.g., different
        role names, content representation). This method maps the unified LangChain
        format to the provider's specific requirements.
        """
        # Implementation: map LangChain message types to provider format
        # Handle HumanMessage, AIMessage, SystemMessage, ToolMessage
        pass
```

### Message Conversion and Content Blocks

LangChain messages have a **unified, provider-agnostic content format** using **content blocks** (reference: `/openwiki/messages.md`). Each provider must translate between this format and its own API schema on both input and output.

**Key message fields:**

- **`content`**: `str | list[dict]` - Either plain text or structured content blocks
- **`tool_calls`**: `list[ToolCall]` - Structured tool invocation requests from the model
- **`usage_metadata`**: Token counts and category breakdowns
- **`response_metadata`**: Provider-specific response data (model name, usage, finish reason, etc.)

**Unified content block types:**
- `{"type": "text", "text": "..."}` - Plain text
- `{"type": "text", "text": "...", "annotations": [...]}` - Text with citations/annotations (Responses API)
- `{"type": "image", "source": {...}}` - Images (multiple source formats: base64, URL, etc.)
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` - Tool calls
- `{"type": "tool_result", ...}` - Tool execution results
- `{"type": "reasoning", ...}` - Reasoning content (advanced models)
- `{"type": "audio", ...}` - Audio input/output (multimodal)

**Conversion responsibilities:**

1. **Input (messages → provider format)**:
   - Map LangChain message types to provider roles (HumanMessage → "user", AIMessage → "assistant", etc.)
   - Convert unified content blocks to provider-specific formats
   - Handle tool calls and structured output schemas
   - Preserve special annotations and extras (e.g., prompt caching, reasoning parameters)

2. **Output (provider response → AIMessage)**:
   - Parse provider-specific response format (JSON, streaming chunks, tool results)
   - Map provider content types to unified blocks
   - Extract and normalize tool calls into `ToolCall` objects
   - Populate `usage_metadata` (input/output token counts)
   - Store provider-specific metadata in `response_metadata`

**Example: Message conversion (Anthropic reference):**

The Anthropic provider converts both input and output (reference: `repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1-L150`):

```python
def _convert_messages_to_provider_format(
    self, messages: list[BaseMessage]
) -> list[dict]:
    """Convert LangChain messages to Anthropic API format."""
    provider_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Convert HumanMessage to Anthropic user role
            provider_messages.append({
                "role": "user",
                "content": self._format_content(msg.content),
            })
        elif isinstance(msg, AIMessage):
            # Convert AIMessage to Anthropic assistant role, including tool calls
            content = self._format_content(msg.content)
            if msg.tool_calls:
                # Append structured tool_use blocks
                content.extend([
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    }
                    for tc in msg.tool_calls
                ])
            provider_messages.append({
                "role": "assistant",
                "content": content,
            })
        elif isinstance(msg, SystemMessage):
            provider_messages.append({
                "role": "user",
                "content": msg.content,
            })
    
    return provider_messages
```

### Converting Provider Response Format to Unified Messages

After receiving a response from the provider API, convert it back to LangChain's unified `AIMessage` format. This is the reverse direction of message conversion.

**Key conversion steps:**

1. **Parse provider response** - Extract choices/outputs from provider-specific format
2. **Create unified content blocks** - Map provider content types to LangChain blocks
3. **Handle tool calls** - Parse provider tool results into `ToolCall` objects
4. **Populate metadata** - Capture usage info and provider-specific response metadata
5. **Create AIMessage** - Construct the output message with all components

**Example: OpenAI response conversion** (reference: `repo://libs/partners/openai/langchain_openai/chat_models/base.py#L217-L282`):

```python
from langchain_core.messages import AIMessage, UsageMetadata
from langchain_core.output_parsers.openai_tools import parse_tool_call, make_invalid_tool_call

def _convert_response_to_message(self, response: dict) -> AIMessage:
    """Convert provider API response to AIMessage."""
    # Parse the response choice (provider-specific format)
    choice = response["choices"][0]
    message_data = choice["message"]
    
    # Extract content, tool calls, and metadata
    content = message_data.get("content", "")
    
    # Convert tool calls from provider format to LangChain ToolCall
    tool_calls = []
    invalid_tool_calls = []
    if "tool_calls" in message_data:
        for raw_tc in message_data["tool_calls"]:
            try:
                # Parse provider tool call into ToolCall
                tool_calls.append(parse_tool_call(raw_tc, return_id=True))
            except Exception as e:
                # Store malformed tool calls for inspection
                invalid_tool_calls.append(
                    make_invalid_tool_call(raw_tc, str(e))
                )
    
    # Extract usage metadata (normalize across providers)
    usage_metadata = {}
    if "usage" in response:
        usage_metadata = {
            "input_tokens": response["usage"].get("prompt_tokens", 0),
            "output_tokens": response["usage"].get("completion_tokens", 0),
        }
    
    # Build response metadata with provider-specific fields
    response_metadata = {
        "model_name": response.get("model"),
        "finish_reason": choice.get("finish_reason"),
    }
    
    # Create unified AIMessage
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        usage_metadata=UsageMetadata(**usage_metadata) if usage_metadata else None,
        response_metadata=response_metadata,
    )
```

**Handling provider-specific response fields:**

Some providers return additional fields not in the standard format. Store these in `response_metadata`:

- OpenAI: `model_name`, `service_tier`, `incomplete_details` (Responses API), `id` (response ID)
- Anthropic: `stop_reason`, `usage`, `content` (structured blocks)
- Custom fields: Any provider-specific data the application may need

## 3. Streaming Architecture

### Stream Implementation Pattern

Streaming returns `AIMessageChunk` objects incrementally as the model generates tokens. The implementation must:

1. **Enable streaming at request time** by setting the streaming flag on the provider API
2. **Iterate over provider events** (e.g., SSE chunks, iterator)
3. **Extract delta/token content** from each event
4. **Create `AIMessageChunk`** with incremental content
5. **Wrap in `ChatGenerationChunk`** for the generation abstraction
6. **Notify run_manager** via `on_llm_new_token` callback for observability

**Key pattern** (Anthropic reference: `repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1862-L1910`):

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    # Enable streaming in API payload
    kwargs["stream"] = True
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    
    # Stream from API
    raw_response = self._client.create(payload)
    for event in raw_response.parse():
        # Convert Anthropic streaming event to AIMessageChunk
        msg_chunk = self._make_message_chunk_from_anthropic_event(event)
        
        if msg_chunk is not None:
            chunk = ChatGenerationChunk(message=msg_chunk)
            
            # Notify callbacks
            if run_manager and isinstance(msg_chunk.content, str):
                run_manager.on_llm_new_token(msg_chunk.content, chunk=chunk)
            
            yield chunk
```

### Async Streaming

Implement `_astream` as the async variant of `_stream`, using `async for` instead of `for`:

```python
async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]:
    kwargs["stream"] = True
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    
    raw_response = await self._acreate(payload)
    async for event in await _aparse(raw_response):
        msg_chunk = self._make_message_chunk_from_anthropic_event(event)
        if msg_chunk is not None:
            chunk = ChatGenerationChunk(message=msg_chunk)
            if run_manager and isinstance(msg_chunk.content, str):
                await run_manager.on_llm_new_token(msg_chunk.content, chunk=chunk)
            yield chunk
```

## 4. Provider-Specific Features

### Function Calling / Tool Use

Implement `bind_tools()` (inherited from `BaseChatModel`) to support tool calling. Tools are converted to the provider's schema (OpenAI, Anthropic, JSON Schema, etc.) before sending to the API. The model response includes tool calls, which are extracted and populated in `AIMessage.tool_calls`.

**Implementation approach:**

1. **Accept `BaseTool` objects, Pydantic models, or dicts** via `bind_tools()`
2. **Convert to provider schema** using utility functions:
   - `convert_to_openai_tool()` - For OpenAI-compatible APIs
   - `convert_to_json_schema()` - For JSON Schema format
   - Provider-specific converters for custom formats
3. **Include tools in API request** as part of the payload
4. **Parse tool calls** from the response into `ToolCall` objects
5. **Handle invalid/malformed tool calls** by storing them in `invalid_tool_calls`

### Structured Output

Implement `with_structured_output()` to enforce the model to return responses matching a Pydantic model or JSON schema. This typically maps to the provider's structured output or JSON mode feature.

**Pattern:**

1. Accept a Pydantic model or JSON schema
2. Convert to provider's structured output format
3. Include in API request
4. Parse response and validate against schema
5. Return parsed model instance or dict

### Vision / Multimodal Input

Support image, video, and audio inputs via content blocks:

- **Images**: `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}`
- **Video/Audio**: Similar structure with appropriate media types

Translate these to provider-specific formats (e.g., OpenAI's `image_url`, Anthropic's `source` block).

## 5. Provider Registration in init_chat_model

Add your provider to the **built-in registry** to enable automatic factory instantiation (`repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L97`):

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    # ... existing providers ...
    
    "provider_name": (
        "langchain_provider_name",           # Module path
        "ChatProviderModel",                 # Class name
        _call,                               # Instantiation function (_call is standard)
    ),
    
    # Special case: custom instantiation function (e.g., IBM Watson)
    # "ibm": ("langchain_ibm", "ChatWatsonx", lambda cls, model, **kwargs: cls(model_id=model, **kwargs)),
}
```

**After registration, users can instantiate your model:**

```python
from langchain.chat_models import init_chat_model

# With explicit provider prefix
model = init_chat_model("provider_name:model-id", temperature=0.5)

# With inferred provider (if model name starts with unique prefix)
model = init_chat_model("unique-prefix-model-id")
```

**Provider inference heuristics** are defined in `_attempt_infer_model_provider()`:

| Model Prefix | Inferred Provider |
|---|---|
| `gpt-`, `o1`, `o3` | `openai` |
| `claude` | `anthropic` |
| `mistral`, `mixtral` | `mistralai` |

Add your provider's prefixes to the inference function to enable bare model name registration.

## 6. Model Profiles

**Model profiles** expose capability data (context window, supported modalities, tool calling, etc.) via `model.profile` property. Users and integrations query this to determine model capabilities.

### Profile Structure and Data Source

Profiles are dictionaries stored in `data/_profiles.py` and generated from the open-source [models.dev](https://github.com/sst/models.dev) project via the `langchain-model-profiles` CLI tool.

**Sample profile** (reference: `repo://libs/partners/anthropic/langchain_anthropic/data/_profiles.py#L18-L52`):

```python
_PROFILES: dict[str, dict[str, Any]] = {
    "claude-opus-4-7": {
        "name": "Claude Opus 4.7",
        "release_date": "2025-09-01",
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "pdf_inputs": True,
        "tool_calling": True,
        "structured_output": True,
        "tool_call_streaming": True,
        "reasoning_output": True,
    },
    # ... more models ...
}
```

### Updating Profiles

Use the `langchain-model-profiles` CLI tool to refresh profiles from models.dev:

```bash
uv add langchain-model-profiles  # Install once globally or in dev dependencies

# Refresh profiles for your provider
langchain-model-profiles refresh \
    --provider provider_name \
    --data-dir ./langchain_provider_name/data
```

This downloads the latest model data, merges provider-specific augmentations from `profile_augmentations.toml`, and generates `_profiles.py`.

### Provider Augmentations

Create `data/profile_augmentations.toml` for LangChain-specific capability overrides (reference: `repo://libs/partners/anthropic/langchain_anthropic/data/profile_augmentations.toml`):

```toml
provider = "provider_name"

[overrides]
# Global overrides for all models
tool_call_streaming = true

[overrides."specific-model-id"]
# Model-specific overrides
structured_output = true
reasoning_effort_levels = ["low", "medium", "high"]
reasoning_effort_default = "high"
```

## 7. Standard Tests

LangChain provides a standard test suite for chat models via the `langchain-tests` package. Providers must implement unit and integration tests by inheriting the base test classes.

### Unit Tests

Create `tests/unit_tests/chat_models/test_standard.py` (reference: `/libs/standard-tests/README.md`):

```python
"""Standard LangChain interface tests for ChatProviderModel."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelStandard(ChatModelUnitTests):
    """Standard unit tests for ChatProviderModel."""
    
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatProviderModel
    
    @pytest.fixture
    def chat_model_params(self) -> dict:
        """Parameters to instantiate the chat model.
        
        Must include all required constructor arguments (e.g., api_key if it's required).
        """
        return {
            "model": "model-123",
            "api_key": "test-key",  # Use environment variable in real tests
        }
```

**Configurable test fixtures** (from `langchain-tests` README):

- `chat_model_class` (required): The `BaseChatModel` subclass to test
- `chat_model_params`: Kwargs for instantiation (defaults to empty dict)
- `chat_model_has_tool_calling`: Whether the model supports `bind_tools()` (auto-detected)
- `chat_model_has_structured_output`: Whether the model supports `with_structured_output()` (auto-detected)

### Integration Tests

Create `tests/integration_tests/chat_models/test_standard.py`:

```python
"""Standard integration tests for ChatProviderModel."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatProviderModel."""
    
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatProviderModel
    
    @pytest.fixture
    def chat_model_params(self) -> dict:
        """Live API credentials (loaded from environment)."""
        return {
            "model": "model-123",
            # API key loaded from PROVIDER_NAME_API_KEY environment variable
        }
```

### Test Coverage

The standard test suite validates:

- **Invoke/stream methods**: Both sync and async
- **Message handling**: All message types and content blocks
- **Tool calling**: If `bind_tools()` is implemented
- **Structured output**: If `with_structured_output()` is implemented
- **Callbacks**: Token counting, error handling
- **Model profile**: Presence and validity

### Advanced API Testing (Optional)

If your provider supports advanced response modes (e.g., OpenAI's Responses API), create additional test classes for those modes:

**Example: Responses API unit tests** (reference: `repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py`):

```python
"""Standard tests with Responses API enabled."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelResponsesAPI(ChatModelUnitTests):
    """Standard tests using the advanced Responses API mode."""
    
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @property
    def chat_model_params(self) -> dict:
        """Enable advanced API mode and any special configuration."""
        return {
            "model": "model-123",
            "use_responses_api": True,  # or provider-specific equivalent
            "stream_usage": True,  # Common for advanced modes
        }
```

Integration tests for advanced modes should verify:
- Response format compatibility (content blocks, tool outputs, etc.)
- Streaming with structured output preserved
- Tool execution in the advanced format
- Incomplete/truncated responses (if applicable)

## 8. Advanced API Modes and Response Formats

### Responses API (Provider-Specific Advanced Feature)

Some providers (e.g., OpenAI) offer advanced response APIs alongside standard chat completions. These APIs typically support enhanced features like streaming tool execution, reasoning outputs, and structured response formats.

**Example: OpenAI Responses API** (reference: `/openwiki/openai-provider.md`, `repo://libs/partners/openai/tests/integration_tests/chat_models/test_responses_api.py`):

The `ChatOpenAI` provider supports an optional `use_responses_api` parameter to switch between the Chat Completions API and the Responses API:

```python
from langchain_openai import ChatOpenAI

# Standard Chat Completions API (default)
model = ChatOpenAI(model="gpt-4o-mini")

# Advanced Responses API (for streaming, reasoning models, etc.)
model = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True,
    reasoning={"effort": "medium"},  # Reasoning support
    include=["web_search_call.results"],  # Include tool outputs
)

# Invoke the model
response = model.invoke("What is the capital of France?")
# Returns AIMessage with structured content including reasoning, tool calls
```

**Key differences in Responses API:**

1. **Response format**: Returns structured content blocks (text, reasoning, tool calls, web_search_calls, etc.) rather than plain text
2. **Message content**: `response.content` is a list of blocks with `type`, `id`, and metadata
3. **Streaming behavior**: Streaming yields partial blocks; final aggregation preserves structure
4. **Tool execution**: Tool calls and results are handled as content blocks
5. **Output format versions**: Use `output_version="responses/v1"` to control parsing

**Message conversion for Responses API:**

Provider implementations must convert Responses API output (which includes tool execution results, reasoning, and other structured items) back to unified `AIMessage` format. The OpenAI provider uses `_convert_from_v1_to_responses()` (reference: `repo://libs/partners/openai/langchain_openai/chat_models/_compat.py#L420-L514`) to translate content blocks:

```python
# Responses API returns:
{
    "type": "text",
    "text": "Paris is the capital of France",
    "annotations": [{"type": "web_search", "title": "..."}],
    "id": "msg_123"
}

# Converted to unified AIMessage content block:
{
    "type": "text",
    "text": "Paris is the capital of France",
    "annotations": [{"type": "web_search", "title": "..."}]
}
```

**Recommendation**: Responses API support is optional and provider-specific. Implement it only if your provider's SDK supports it. For reference implementations, see OpenAI's `ChatOpenAI.use_responses_api` and corresponding test fixtures in `test_responses_standard.py`.

## 8. Error Handling

Map provider-specific exceptions to LangChain's unified exception hierarchy (reference: `repo://libs/core/langchain_core/exceptions.py`):

| Provider Exception | LangChain Exception |
|---|---|
| `ProviderAPIError` | `ModelAPIError` |
| `ProviderAuthenticationError` | `ModelAuthenticationError` |
| `ProviderRateLimitError` | `ModelRateLimitError` |
| `ProviderTimeoutError` | `ModelTimeoutError` |
| `ProviderConnectionError` | `ModelConnectionError` |

**Implementation pattern:**

```python
def _generate(self, messages, **kwargs):
    try:
        response = self._client.chat.create(...)
    except provider_sdk.AuthenticationError as e:
        raise ModelAuthenticationError(str(e)) from e
    except provider_sdk.RateLimitError as e:
        raise ModelRateLimitError(str(e)) from e
    except provider_sdk.APIError as e:
        raise ModelAPIError(str(e)) from e
    # ... rest of generation logic
```

## 9. Example: OpenAI Provider Reference

The OpenAI provider (`repo://libs/partners/openai/langchain_openai/chat_models/base.py`) is a comprehensive reference implementation demonstrating:

- **Message conversion**: Support for images, function calling, reasoning content
- **Streaming**: Proper delta extraction and token counting
- **Tool calling**: Convert to OpenAI format, parse structured responses
- **Structured output**: JSON Schema validation and parsing
- **Error mapping**: Detailed provider-specific error handling
- **Async support**: Full async/await implementation for all methods

## 10. Maintenance and Updates

### Dependency Updates

Keep the provider SDK locked in `pyproject.toml` to prevent breaking changes. Review provider release notes regularly for new models and API changes.

### Model Profile Updates

Run the CLI tool periodically to fetch new models from models.dev:

```bash
langchain-model-profiles refresh --provider provider_name --data-dir ./langchain_provider_name/data
```

### Testing

Run standard tests before releasing updates:

```bash
# Unit tests (no API credentials required)
pytest tests/unit_tests/

# Integration tests (requires provider API credentials)
pytest tests/integration_tests/
```

## Checklist for Adding a New Provider

- [ ] Create package structure in `/libs/partners/provider_name/`
- [ ] Implement `ChatProviderModel` inheriting `BaseChatModel`
- [ ] Implement `_generate` method for synchronous generation
- [ ] Implement `_stream` method for token streaming
- [ ] Implement `_agenerate` or `_astream` for async support
- [ ] Convert messages from LangChain format to provider API schema
- [ ] Parse and convert provider responses to `AIMessage`/`AIMessageChunk`
- [ ] Implement `bind_tools()` for function calling (if supported)
- [ ] Implement `with_structured_output()` for structured output (if supported)
- [ ] Map provider exceptions to LangChain exception hierarchy
- [ ] Fetch and store model profiles via `langchain-model-profiles` CLI
- [ ] Add provider to `_BUILTIN_PROVIDERS` registry in `init_chat_model`
- [ ] Create unit test suite inheriting `ChatModelUnitTests`
- [ ] Create integration test suite inheriting `ChatModelIntegrationTests`
- [ ] Document public API in docstrings and README
- [ ] Add provider to model name inference heuristics (if applicable)
- [ ] Update integrations documentation and changelog

## Related Documentation

- [Chat Models Interface](/openwiki/chat-models.md) - Core chat model protocol
- [Message Types](/openwiki/messages.md) - Message abstraction and content blocks
- [Model Initialization](/openwiki/model-initialization.md) - `init_chat_model` factory details
- [OpenAI Provider](/openwiki/openai-provider.md) - Reference implementation
- [LangChain Integrations Documentation](https://docs.langchain.com/oss/python/integrations/providers/overview) - User-facing guide
