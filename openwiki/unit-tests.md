---
type: "Testing & QA"
title: "Unit Testing: Strategies and Patterns"
description: "How to write unit tests for langchain-core and langchain components using pytest, fixtures, mocking, and standard test classes from langchain-tests."
tags: [unit-tests, pytest, testing, fixtures, mocking, chat-models, tools, embeddings, type-checking, mypy]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-043c2520f819900dc753650e
    resource: repo://libs/core/tests/unit_tests/callbacks/test_async_callback_manager.py
  - id: openwiki-source-727aef6a92fb635fdbb41cd6
    resource: repo://libs/core/tests/unit_tests/conftest.py
  - id: openwiki-source-5e13d2c899eb5925ef28fddf
    resource: repo://libs/core/tests/unit_tests/fake/callbacks.py
  - id: openwiki-source-e8916b46b41eee662deabd17
    resource: repo://libs/core/tests/unit_tests/fake/test_fake_chat_model.py
  - id: openwiki-source-f0e376fe9b6befdcc2505465
    resource: repo://libs/core/tests/unit_tests/pydantic_utils.py
  - id: openwiki-source-344bd4b667096c3c45c8fa82
    resource: repo://libs/core/tests/unit_tests/runnables/conftest.py
  - id: openwiki-source-4717abc86db20c5c76bbf23a
    resource: repo://libs/core/tests/unit_tests/runnables/test_runnable.py
  - id: openwiki-source-5839db669f618a6d604790ca
    resource: repo://libs/core/tests/unit_tests/stubs.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
  - id: openwiki-source-3eb9100e02f9d70098d1b30d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py
  - id: openwiki-source-54e69c0cb7aa4a73b87cf97d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/embeddings.py
  - id: openwiki-source-a6b31954b6df57580d0f3ed0
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

Unit testing in LangChain covers components in isolation without network calls or external API dependencies. Tests live in `tests/unit_tests/` directories and are run via `make test` or `uv run --group test pytest` with strict socket restrictions and parallelization.

This page covers the test infrastructure, standard test classes for chat models and tools, common patterns (fixtures, parametrization, mocking, callbacks, snapshot testing), and type checking with mypy.

## Test Structure and Organization

### Directory Layout

Every LangChain package organizes tests consistently:

```
libs/core/
├── tests/
│   ├── unit_tests/       # No network calls; run via make test
│   ├── integration_tests/ # Live API calls; require credentials and API keys
│   └── benchmarks/       # Performance measurement tests
├── Makefile              # Task automation
└── pyproject.toml        # Dependencies
```

Unit tests mirror the source code structure: a module at `langchain_core/runnables/base.py` has tests in `tests/unit_tests/runnables/test_runnable.py`.

### Running Unit Tests

All unit tests are run in parallel with socket restrictions to prevent accidental network access:

```bash
# Run all unit tests in a package
make test

# Run a specific test file or directory
make test TEST_FILE=tests/unit_tests/runnables/test_runnable.py

# Run using uv directly
uv run --group test pytest tests/unit_tests/

# Watch mode: auto-rerun on code changes
make test_watch

# Extended tests (marked with @pytest.mark.requires)
make extended_tests
```

The Makefile test target sets `--disable-socket --allow-unix-socket` and uses `pytest-xdist` (`-n auto`) for parallel execution. Environment variables for LangSmith tracing (`LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`, etc.) are explicitly unset to keep tests independent.

## Standard Test Classes

The `langchain-tests` package (in `/libs/standard-tests/`) provides reusable base test classes for integrations. These enforce consistent testing across chat models, embeddings, and tools.

### ChatModelUnitTests

**Location**: `langchain_tests.unit_tests.ChatModelUnitTests`

For any chat model, create a test class that inherits from `ChatModelUnitTests` and implements two required properties:

```python
# tests/unit_tests/test_standard.py
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from my_package.chat_models import MyChatModel


class TestMyChatModelUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return MyChatModel

    @property
    def chat_model_params(self) -> dict:
        return {"model": "my-model-001", "temperature": 0}
```

**What It Tests**:
- **Initialization**: Model instantiation with standard parameters
- **Sync/async invoke**: Single message handling in sync and async contexts
- **Streaming**: Chunked streaming responses and chunk accumulation
- **Tool binding**: `bind_tools()` interface (if supported)
- **Structured output**: `with_structured_output()` for schema enforcement (if supported)
- **Serialization**: Dumping and loading the model via LangChain's serialization API
- **Message types**: Single and multi-message conversations, system prompts, tool messages
- **Tool calling**: Correct tool call invocation and result handling (if supported)

**Configurable Features** (override as properties):

- `has_tool_calling` (bool): Whether the model's `bind_tools` method is overridden; auto-detected but can be set explicitly
- `has_tool_choice` (bool): Whether `bind_tools` accepts a `tool_choice` parameter for forcing tool calls
- `has_structured_output` (bool): Whether `with_structured_output()` or `bind_tools()` is implemented
- `structured_output_kwargs` (dict): Additional kwargs for `with_structured_output()` (e.g., `{"method": "json_schema"}`)
- `supports_json_mode` (bool): Whether the model supports `method='json_mode'` in structured output
- `supports_image_inputs` (bool): Whether the model accepts image content blocks
- `supports_image_urls` (bool): Whether the model accepts image URLs in content
- `supports_pdf_inputs` (bool): Whether the model accepts PDF file content
- `supports_audio_inputs` (bool): Whether the model accepts audio content
- `returns_usage_metadata` (bool): Whether `invoke()` and `stream()` return usage token counts (default: True)
- `supports_model_override` (bool): Whether the model accepts a `model` parameter in `invoke()` to override at runtime (default: True)
- `model_override_value` (str): Alternative model name for testing dynamic model selection (required if `supports_model_override=True`)

Example with feature flags:

```python
class TestOpenAIChatModel(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4"}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def structured_output_kwargs(self) -> dict:
        return {"method": "json_schema"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def model_override_value(self) -> str:
        return "gpt-4-turbo"
```

### EmbeddingsUnitTests

**Location**: `langchain_tests.unit_tests.EmbeddingsUnitTests`

Test embeddings models similarly:

```python
from typing import Type

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests

from my_package.embeddings import MyEmbeddings


class TestMyEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return MyEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "embedding-v1"}
```

**What It Tests**:
- Model initialization
- Embedding a single text string
- Embedding a batch of text strings
- Initialization from environment variables (if `init_from_env_params` is set)

**Configurable**:
- `init_from_env_params` (tuple): Return `(env_vars, init_args, expected_attrs)` to test env-based initialization

### ToolsUnitTests

**Location**: `langchain_tests.unit_tests.ToolsUnitTests`

Test custom tools:

```python
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from my_package.tools import MyTool


class TestMyToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool] | BaseTool:
        return MyTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "example query"}
```

**What It Tests**:
- Tool initialization
- Tool invocation with example parameters
- Tool schema generation (JSON schema)
- Initialization from environment variables

## Shared Fixtures and Configuration

### conftest.py Patterns

The root `conftest.py` in `tests/unit_tests/` provides shared fixtures and pytest hooks.

**From `/libs/core/tests/unit_tests/conftest.py`**:

```python
@pytest.fixture(autouse=True)
def blockbuster() -> Iterator[BlockBuster]:
    """Blockbuster fixture prevents blocking I/O in async code."""
    with blockbuster_ctx("langchain_core") as bb:
        # Allow blocking in specific functions (e.g., internal API checks)
        bb.functions["os.stat"].can_block_in(
            "langchain_core/_api/internal.py", "is_caller_internal"
        )
        yield bb
```

**Custom Markers**:

```python
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--only-extended",
        action="store_true",
        help="Only run extended tests marked with @pytest.mark.requires",
    )
    parser.addoption(
        "--only-core",
        action="store_true",
        help="Only run core tests (skip extended tests)",
    )

def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """Automatically skip tests marked with @pytest.mark.requires if dependencies are missing."""
    for item in items:
        requires_marker = item.get_closest_marker("requires")
        if requires_marker:
            for pkg in requires_marker.args:
                if util.find_spec(pkg) is None:
                    item.add_marker(pytest.mark.skip(reason=f"Requires pkg: {pkg}"))
```

**Fixture for Deterministic UUIDs**:

```python
@pytest.fixture
def deterministic_uuids(mocker):
    """Replace random UUIDs with deterministic values for snapshot testing."""
    side_effect = (UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000))
    return mocker.patch("uuid.uuid4", side_effect=side_effect)
```

Use the `deterministic_uuids` fixture in tests where UUIDs must be stable across runs:

```python
def test_runnable_with_trace(deterministic_uuids):
    # UUIDs will be predictable now
    ...
```

### Marker Patterns

```python
# Skip test if dependency is missing
@pytest.mark.requires("anthropic")
def test_anthropic_tool_calling():
    from anthropic import Anthropic
    ...

# Extended tests (run with make extended_tests or --only-extended)
@pytest.mark.requires("openai")
def test_openai_structured_output():
    ...

# Parametrized tests
@pytest.mark.parametrize("model_name,expected_tokens", [
    ("small", 100),
    ("large", 1000),
])
def test_model_sizes(model_name, expected_tokens):
    ...

# Skip on Python version
@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires 3.11+")
def test_new_feature():
    ...

# Expected failure
@pytest.mark.xfail(reason="Feature not yet implemented")
def test_future_feature():
    ...
```

## Fake Implementations for Testing

LangChain provides fake/mock chat models and other components to avoid API calls in unit tests.

### FakeChatModel Classes

Located in `langchain_core.language_models`:

```python
from langchain_core.language_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
```

**FakeListChatModel**: Cycles through a fixed list of string responses.

```python
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(responses=["Hello", "Hi", "Hey"])
response = model.invoke("How are you?")
# Returns AIMessage(content="Hello")

response = model.invoke("What's up?")
# Returns AIMessage(content="Hi")
```

**GenericFakeChatModel**: Cycles through AIMessage objects; useful for testing streaming.

```python
from itertools import cycle
from langchain_core.messages import AIMessage
from langchain_core.language_models import GenericFakeChatModel

messages = cycle([AIMessage(content="response1"), AIMessage(content="response2")])
model = GenericFakeChatModel(messages=messages)

# Test streaming
chunks = list(model.stream("query"))
# Chunks are character-level splits of "response1"
```

**ParrotFakeChatModel**: Echoes the input message back.

```python
from langchain_core.language_models import ParrotFakeChatModel

model = ParrotFakeChatModel()
response = model.invoke("Hello!")
# Returns AIMessage(content="Hello!")
```

**FakeListLLM and FakeStreamingListLLM**: Older LLM interface (text-in, text-out).

```python
from langchain_core.language_models import FakeListLLM

llm = FakeListLLM(responses=["Response 1", "Response 2"])
output = llm.invoke("Query")
```

### FakeEmbeddings

```python
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(model="fake-model", size=1536)

# Embed a single string
vector = embeddings.embed_query("hello")
# Returns a list of 1536 float values (deterministic based on input hash)

# Embed a batch
vectors = embeddings.embed_documents(["hello", "world"])
# Returns list of vectors, one per input
```

### FakeCallbackHandler

Located in `tests.unit_tests.fake.callbacks`, a test callback handler that counts events:

```python
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

handler = FakeCallbackHandler()

# Track various events
assert handler.llm_starts == 0
assert handler.chain_starts == 0

# After invoke on a chain with LLM calls:
model.invoke("query", callbacks=[handler])

assert handler.llm_starts == 1
assert handler.llm_ends == 1
assert handler.starts == 1  # Total starts

# Fine-grained counters
assert handler.llm_streams == 0  # for streaming models
assert handler.tool_starts == 0
assert handler.tool_ends == 0
assert handler.chain_starts == 1
assert handler.chain_ends == 1
```

## Common Testing Patterns

### Fixture Usage

Define reusable components as pytest fixtures:

```python
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

@pytest.fixture
def system_prompt():
    return SystemMessage(content="You are a helpful assistant.")

@pytest.fixture
def chat_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{user_input}"),
    ])

def test_with_prompt(chat_prompt):
    # Use the fixture
    assert chat_prompt is not None
```

Fixtures in `conftest.py` are automatically discovered and available to all tests in that directory and subdirectories.

### Parametrization

Test multiple input/output combinations:

```python
import pytest

@pytest.mark.parametrize("input_text,expected_length", [
    ("hello", 5),
    ("world", 5),
    ("testing", 7),
    ("", 0),
])
def test_text_length(input_text, expected_length):
    assert len(input_text) == expected_length
```

Parametrize with fixtures:

```python
@pytest.fixture(params=["gpt-3.5-turbo", "gpt-4"])
def model_name(request):
    return request.param

def test_model_response(model_name):
    # Test runs twice, once for each model
    model = ChatOpenAI(model=model_name)
    response = model.invoke("Hello")
    assert response.content is not None
```

### Mocking with pytest-mock

The `pytest-mock` library provides a `mocker` fixture:

```python
from unittest import mock

def test_with_mock(mocker):
    # Mock a function
    mock_api_call = mocker.patch("my_module.api_call")
    mock_api_call.return_value = {"status": "success"}

    # Call code that uses api_call
    result = my_function()
    
    assert mock_api_call.called
    assert result == {"status": "success"}

    # Check call arguments
    mock_api_call.assert_called_with("expected_arg")
```

Mock environment variables:

```python
from unittest import mock
import os

def test_env_initialization(mocker):
    mocker.patch.dict(os.environ, {"API_KEY": "test-key"})
    
    # Code that reads API_KEY will get "test-key"
    model = MyModel()  # reads from os.environ
    assert model.api_key == "test-key"
```

### Callback Testing

Test that callbacks are invoked with correct data:

```python
from langchain_core.callbacks.manager import CallbackManager
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

def test_callbacks_on_chain():
    handler = FakeCallbackHandler()
    
    # Create a chain
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    model = FakeChatModel(responses=["Hello Alice"])
    chain = prompt | model
    
    # Invoke with callbacks
    result = chain.invoke(
        {"name": "Alice"},
        config={"callbacks": [handler]}
    )
    
    # Verify callbacks were fired
    assert handler.starts == 2  # prompt + model
    assert handler.ends == 2
    assert handler.chain_starts == 0  # Only LLM runs were tracked
    assert handler.llm_starts == 1
    assert handler.llm_ends == 1
```

### Async Testing

Mark async tests with `async def` and `pytest` handles them:

```python
import pytest

@pytest.mark.asyncio
async def test_async_invoke():
    model = ChatOpenAI()
    result = await model.ainvoke("Hello")
    assert result.content is not None

@pytest.mark.asyncio
async def test_async_streaming():
    model = ChatOpenAI()
    chunks = []
    async for chunk in model.astream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 0
```

### Snapshot Testing with Syrupy

Snapshot tests capture output and compare against baseline snapshots. Useful for complex structures, traces, and serialized objects.

```python
from syrupy.assertion import SnapshotAssertion

def test_runnable_serialization(snapshot: SnapshotAssertion):
    prompt = ChatPromptTemplate.from_template("Say {msg}")
    model = ChatOpenAI(model="gpt-4")
    chain = prompt | model
    
    # Dump to serializable form
    dumped = dumpd(chain)
    
    # Compare against snapshot
    assert dumped == snapshot
```

Snapshots are stored in `__snapshots__/` directories. Update them with:

```bash
make test_watch  # Auto-updates snapshots
# or
pytest --snapshot-update
```

### Helper Stubs for Message Tests

When testing messages with generated IDs, use helper functions from `tests.unit_tests.stubs` to match any ID:

```python
from tests.unit_tests.stubs import (
    _any_id_ai_message,
    _any_id_ai_message_chunk,
    _any_id_human_message,
    AnyStr,
)

def test_message_response():
    model = GenericFakeChatModel(messages=cycle([AIMessage(content="hello")]))
    response = model.invoke("hi")
    
    # Matches any ID
    assert response == _any_id_ai_message(content="hello")

def test_message_streaming():
    model = GenericFakeChatModel(messages=cycle([AIMessage(content="hello")]))
    chunks = list(model.stream("hi"))
    
    assert chunks[0] == _any_id_ai_message_chunk(content="h")
    assert chunks[1] == _any_id_ai_message_chunk(content="ello", chunk_position="last")
```

The `AnyStr` class matches any string when used as a value:

```python
message.id = AnyStr()  # Now message.id == any_other_id is True
```

## Type Checking with mypy

Type checking is part of the standard lint workflow:

```bash
# Full type checking
make type

# Or directly with mypy
mypy libs/core/langchain_core/

# Type check specific file
mypy libs/core/langchain_core/runnables/base.py
```

The Makefile runs `mypy` as part of `make lint`, which also runs ruff and format checks:

```bash
make lint  # runs: ruff check, ruff format --diff, mypy
```

### Type Checking Patterns

Use type hints throughout:

```python
from typing import Any, Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

def create_chain(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    temperature: float = 0.7,
) -> str:
    """Create and invoke a chain.
    
    Args:
        model: The language model to use.
        messages: Input messages.
        temperature: Sampling temperature.
        
    Returns:
        The model's response as a string.
    """
    response = model.invoke(messages, {"temperature": temperature})
    return response.content
```

Handle complex types with `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_tests.unit_tests import ChatModelUnitTests
```

Suppress type errors where necessary with comments (sparingly):

```python
# mypy cannot infer this type from the lambda
my_dict: dict[str, Any] = {}  # type: ignore[assignment]

# Intentional override
result = chain.invoke(message)  # type: ignore[return-value]
```

## Test Coverage and Reporting

Generate coverage reports:

```bash
make coverage

# Reports generated:
# - coverage.xml (for CI)
# - term-missing (terminal output with uncovered lines)
```

## Key Test Infrastructure Files

- **conftest.py** (`libs/core/tests/unit_tests/conftest.py`): Shared fixtures, markers, blockbuster configuration
- **stubs.py** (`libs/core/tests/unit_tests/stubs.py`): Helper functions for message testing with wildcard IDs
- **pydantic_utils.py** (`libs/core/tests/unit_tests/pydantic_utils.py`): Schema normalization for cross-version Pydantic compatibility
- **fake/callbacks.py** (`libs/core/tests/unit_tests/fake/callbacks.py`): FakeCallbackHandler for tracking events
- **fake/test_fake_chat_model.py**: Examples of testing fake models

## Best Practices

1. **Isolate tests**: Each test should be independent and not rely on other tests' state.

2. **Use fixtures**: Factor out setup code into fixtures for reuse and clarity.

3. **Mock external dependencies**: Mock API calls, file I/O, and network operations.

4. **Test behavior, not implementation**: Test what the component does, not how it does it.

5. **Parametrize to reduce duplication**: Use `@pytest.mark.parametrize` for multiple input cases.

6. **Snapshot test complex structures**: Use Syrupy for traces, serialized objects, and large outputs.

7. **Document test intent**: Use clear test names and docstrings.

8. **Run tests before committing**: Use pre-commit hooks or `make test` locally.

9. **Type-check as you go**: Run `make lint` or `make type` during development.

10. **Use markers for categorization**: Mark tests with `@pytest.mark.requires` or custom markers for selective execution.
