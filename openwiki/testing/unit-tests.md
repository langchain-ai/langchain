---
type: "Reference"
title: "Unit Testing: Test Organization & Patterns"
description: "Guide to writing unit tests for LangChain: test structure, fixtures, mocking models/tools, test patterns, edge cases, and CI integration."
tags: [unit-testing, pytest, fixtures, async-testing, mocking, fake-models, tools, callbacks, runnables, snapshot-testing, edge-cases, ci-integration]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-06T08:17:28.425Z
sources:
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-6f37033dbe10f8628b3355af
    resource: repo://libs/core/tests/benchmarks/test_async_callbacks.py
  - id: openwiki-source-043c2520f819900dc753650e
    resource: repo://libs/core/tests/unit_tests/callbacks/test_async_callback_manager.py
  - id: openwiki-source-727aef6a92fb635fdbb41cd6
    resource: repo://libs/core/tests/unit_tests/conftest.py
  - id: openwiki-source-e8916b46b41eee662deabd17
    resource: repo://libs/core/tests/unit_tests/fake/test_fake_chat_model.py
  - id: openwiki-source-f2640c09053c8d4424d50999
    resource: repo://libs/core/tests/unit_tests/messages/test_ai.py
  - id: openwiki-source-344bd4b667096c3c45c8fa82
    resource: repo://libs/core/tests/unit_tests/runnables/conftest.py
  - id: openwiki-source-ab713857815256e872473e57
    resource: repo://libs/core/tests/unit_tests/runnables/test_config.py
  - id: openwiki-source-4717abc86db20c5c76bbf23a
    resource: repo://libs/core/tests/unit_tests/runnables/test_runnable.py
  - id: openwiki-source-f1101c726c80bfb588acd467
    resource: repo://libs/core/tests/unit_tests/test_ssrf_protection.py
  - id: openwiki-source-a690cf632a02205e3f555be8
    resource: repo://libs/core/tests/unit_tests/test_tools.py
generated: { by: "openwiki/0.5.0", at: "2026-11-06T08:17:28.425Z" }
---

## Overview

LangChain's unit test suite validates core components in isolation using **pytest** with specialized plugins for network isolation, async support, mocking, and snapshot testing. Tests enforce determinism, reproducibility, and rapid feedback through fixtures, markers, and careful configuration. The test suite is organized hierarchically with module-specific conftest files and shared test utilities.

## Test Organization and Structure

### Directory Layout

Tests in `libs/core/tests/` follow this mirror structure:

- **`unit_tests/`** – Core component tests organized by module mirroring source structure
  - `runnables/` – Runnable chain and executor tests
  - `callbacks/` – Callback handler and manager tests
  - `language_models/` – Chat model and LLM tests
  - `messages/` – Message type and content block tests
  - `prompts/` – Prompt template tests
  - `output_parsers/` – Parser and schema tests
  - `tools/` → Test files in `test_tools.py` alongside main test files
- **`integration_tests/`** – Tests requiring external services (marked with `@pytest.mark.requires`)
- **`benchmarks/`** – Performance tests using pytest-benchmark and pytest-codspeed

This mirror structure ensures test discovery and organization aligns with source organization. Each major module (e.g., `runnables/`, `callbacks/`) defines its own `conftest.py` for module-specific fixtures and configuration.

### conftest.py Hierarchy

Pytest discovers and applies `conftest.py` files hierarchically from root to test file:

1. **Root conftest** (`tests/unit_tests/conftest.py`) 
   - Global fixtures: `blockbuster` (async blocking detection), `deterministic_uuids`
   - Custom markers and CLI options: `--only-extended`, `--only-core`, `requires` marker
   - Central point for project-wide test behavior

2. **Module conftest** (e.g., `tests/unit_tests/runnables/conftest.py`)
   - Module-specific fixtures and setup
   - Environment isolation (e.g., disabling LangSmith tracing for runnable tests)

#### Global Test Configuration

The root `conftest.py` provides:

- **`blockbuster` fixture** – Autouse fixture that detects async blocking violations by monitoring `os.stat`, `os.path.abspath`, and `io.TextIOWrapper.read` calls. Allows blocking only in specific safe callsites like langsmith client configuration and freezegun internal APIs.

- **`deterministic_uuids` fixture** – Replaces `uuid.uuid4()` with a generator yielding sequential UUIDs (`UUID('00000000-0000-4000-8000-{i:012}')`) for reproducible snapshot tests and deterministic trace comparison.

- **`requires` marker** – Conditional test skipping for optional dependencies:
  ```python
  @pytest.mark.requires("langchain-openai", "anthropic")
  def test_feature():
      # Test is skipped if packages not installed (unless --only-extended)
      pass
  ```

- **Custom CLI options**:
  - `--only-core` – Skip all `@pytest.mark.requires` tests
  - `--only-extended` – Run only `@pytest.mark.requires` tests; fail if dependencies missing

#### Module-Level Configuration

Runnables conftest isolates tests from developer LangSmith environment:

```python
@pytest.fixture(scope="session", autouse=True)
def _disable_local_langsmith_tracing():
    """Keep runnable unit tests independent of developer LangSmith env vars."""
    # Disables LANGSMITH_TRACING, clears API key, project, and cache
```

This prevents trace leakage when developers have LangSmith configured locally.

## Test Directory Structure and Naming

### Mirror Structure Pattern

Test files mirror the source code structure for clarity and organization:

```
libs/core/
├── langchain_core/
│   ├── runnables/
│   │   ├── base.py
│   │   ├── config.py
│   │   └── ...
│   ├── language_models/
│   │   ├── chat_model.py
│   │   ├── fake_chat_models.py
│   │   └── ...
│   └── ...
└── tests/
    └── unit_tests/
        ├── runnables/
        │   ├── conftest.py
        │   ├── test_base.py
        │   ├── test_config.py
        │   └── ...
        ├── language_models/
        │   ├── test_chat_model.py
        │   ├── test_fake_chat_models.py
        │   └── ...
        ├── conftest.py
        └── ...
```

### Test File Naming Conventions

- **`test_*.py`** – Test modules containing test functions and classes
- **`conftest.py`** – Pytest configuration and fixture definitions at any directory level
- **`__snapshots__/`** – Directory containing syrupy snapshot files adjacent to test modules
- **`stubs.py`** or `fixtures.py`** – Shared test utilities and helper functions

### Conftest Hierarchy

Pytest discovers and applies `conftest.py` files in order:

1. **Root conftest** (`tests/unit_tests/conftest.py`) – Global fixtures and markers
2. **Module conftest** (e.g., `tests/unit_tests/runnables/conftest.py`) – Module-specific configuration
3. **Test file imports** – Local fixtures and setup

Fixtures at higher levels are available to all tests in subdirectories, but lower-level fixtures override higher-level ones with the same name.

## Pytest Configuration

### Core Settings

Pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--snapshot-warn-unused --strict-markers --strict-config --durations=5"
markers = [
    "requires: mark tests as requiring a specific library",
    "compile: mark placeholder test used to compile integration tests without running them",
]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

Key settings:
- **`asyncio_mode = "auto"`** – Automatically marks async test functions and fixtures as asyncio tests
- **`asyncio_default_fixture_loop_scope = "function"`** – Fresh event loop per test function prevents state leakage
- **`--snapshot-warn-unused`** – Warns about unused snapshot files during test runs
- **`--strict-markers`** – Rejects undefined markers to catch typos

### Execution with Network Isolation

Standard test command enforces network isolation:

```bash
pytest -n auto --benchmark-disable --disable-socket --allow-unix-socket
```

- **`-n auto`** – Parallel test execution via pytest-xdist, distributing tests across CPU cores
- **`--disable-socket`** – Prevents all network calls (pytest-socket); catches accidental external dependencies
- **`--allow-unix-socket`** – Permits Unix sockets for inter-process communication
- **`--benchmark-disable`** – Skips performance benchmarks in standard runs

## Testing Frameworks and Dependencies

### Core Testing Packages

| Package | Purpose |
|---------|---------|
| `pytest>=9.0.3` | Test runner and framework |
| `pytest-asyncio>=1.3.0` | Async/await test support with `asyncio_mode = "auto"` |
| `pytest-mock>=3.10.0` | `MockerFixture` for mocking and spying on methods |
| `pytest-socket>=0.7.0` | Network call blocking and validation |
| `pytest-xdist>=3.6.1` | Parallel test execution via `-n auto` |
| `syrupy>=5.0.0` | Snapshot testing for complex outputs |
| `freezegun>=1.2.2` | Time mocking for deterministic tests |
| `responses>=0.25.0` | HTTP request mocking for service tests |
| `blockbuster>=1.5.18` | Async blocking detection in sync contexts |
| `pytest-benchmark` | Performance micro-benchmarks per test |
| `pytest-codspeed` | Continuous benchmarking integration |

## Async Testing Patterns

### Basic Async Test

Pytest-asyncio automatically runs async test functions when `asyncio_mode = "auto"`:

```python
async def test_async_invoke() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    result = await prompt.ainvoke({"name": "World"})
    assert isinstance(result, ChatPromptValue)
```

### Async Fixtures

Fixtures can be async with function-level loop scope:

```python
@pytest.fixture
async def async_handler() -> AsyncCallbackHandler:
    handler = MyAsyncCallbackHandler()
    await handler.setup()
    yield handler
    await handler.teardown()

async def test_with_async_fixture(async_handler):
    await async_handler.on_chain_start(...)
```

### Context Variable Propagation in Async Tests

Tests verify context propagation for inline handlers:

```python
async def test_context_propagation() -> None:
    some_var: contextvars.ContextVar[str] = contextvars.ContextVar("some_var")
    
    class CustomHandler(AsyncCallbackHandler):
        async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            some_var.set("on_llm_start")
    
    manager = AsyncCallbackManager(handlers=[CustomHandler(run_inline=True)])
    some_var.set("unset")
    await manager.on_llm_start({}, ["prompt"])
    assert some_var.get() == "on_llm_start"  # Inline handlers update parent context
```

## Fake Models and Testing Without API Calls

### FakeChatModel Classes

LangChain provides multiple fake chat models to test chains without API calls:

#### FakeListChatModel

Returns responses from a predefined list in order:

```python
def test_with_fake_chat_model() -> None:
    chat = FakeListChatModel(responses=["Hello", "World"])
    result = chat.invoke([HumanMessage(content="Hi")])
    assert result.content == "Hello"
    
    result2 = chat.invoke([HumanMessage(content="Bye")])
    assert result2.content == "World"
```

#### FakeListLLM

Returns string responses from a predefined list:

```python
def test_with_fake_llm() -> None:
    llm = FakeListLLM(responses=["Hello", "World"])
    result = llm.invoke("prompt")
    assert result == "Hello"
```

#### GenericFakeChatModel

Accepts any iterable (including `itertools.cycle`) for infinite repeating responses:

```python
def test_with_generic_fake_chat_model() -> None:
    from itertools import cycle
    infinite_responses = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_responses)
    
    # Invoke multiple times; cycles through responses
    assert model.invoke("input1").content == "hello"
    assert model.invoke("input2").content == "goodbye"
    assert model.invoke("input3").content == "hello"
```

#### FakeMessagesListChatModel

Returns message objects with additional kwargs and streaming:

```python
async def test_streaming_with_fake_model() -> None:
    messages = [
        AIMessage(content="hello goodbye")
    ]
    model = GenericFakeChatModel(messages=cycle(messages))
    
    chunks = [chunk async for chunk in model.astream("input")]
    # Chunks are split by word: "hello", " ", "goodbye"
    assert len(chunks) == 3
```

#### FakeStreamingListLLM

Streams LLM responses incrementally:

```python
def test_streaming_llm() -> None:
    llm = FakeStreamingListLLM(responses=["Hello World"])
    chunks = list(llm.stream("prompt"))
    # Chunks are split by token/space: "Hello", " ", "World"
    assert len(chunks) >= 2
```

## Mocking Patterns

### Using pytest-mock

The `MockerFixture` provides convenient mock creation:

```python
def test_with_mock(mocker: MockerFixture) -> None:
    mock_func = mocker.Mock(side_effect=ValueError("error"))
    
    with pytest.raises(ValueError):
        mock_func()
    
    assert mock_func.call_count == 1
    mock_func.assert_called_once()
```

### Spying on Methods

Use `mocker.spy()` to track calls to real methods without replacing implementation:

```python
def test_with_spy(mocker: MockerFixture) -> None:
    fake = FakeRunnable()
    spy = mocker.spy(fake.__class__, "invoke")
    
    result = fake.invoke("input", {"metadata": {"a": "b"}})
    
    spy.assert_called_once_with(
        "input", {"tags": [], "metadata": {"a": "b"}}
    )
    mocker.stop(spy)
```

### Callback Mocking

Mock callback handlers to verify runnable execution paths:

```python
def test_callback_invocation(mocker: MockerFixture) -> None:
    mock_start = mocker.Mock()
    mock_end = mocker.Mock()
    
    class TestCallback(BaseCallbackHandler):
        def on_chain_start(self, *args, **kwargs):
            mock_start(*args, **kwargs)
        def on_chain_end(self, *args, **kwargs):
            mock_end(*args, **kwargs)
    
    chain = prompt | chat
    chain.invoke(
        {"input": "test"},
        config={"callbacks": [TestCallback()]}
    )
    
    assert mock_start.called
    assert mock_end.called
```

### HTTP Request Mocking

The `responses` library mocks HTTP calls for services that make network requests:

```python
import responses

@responses.activate
def test_http_endpoint():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"status": "ok"},
        status=200
    )
    
    result = fetch_data()
    assert result["status"] == "ok"
```

## Network Isolation

### pytest-socket Enforcement

Tests run with `--disable-socket` to prevent unintended network calls. This catches tests that accidentally depend on external services:

```bash
pytest tests/unit_tests/ --disable-socket --allow-unix-socket
```

### Allowing Specific Network Operations

Unix sockets are allowed for local inter-process communication but TCP/UDP sockets are blocked by default.

### SSRF Protection Testing

Tests validate SSRF (Server-Side Request Forgery) protections with mocked socket operations:

```python
from langchain_core._security import SSRFPolicy, validate_resolved_ip, SSRFBlockedError

def test_private_ip_blocked() -> None:
    policy = SSRFPolicy()
    with pytest.raises(SSRFBlockedError, match="private IP"):
        validate_resolved_ip("192.168.1.1", policy)

def test_private_ip_allowed_with_flag() -> None:
    policy = SSRFPolicy(block_private_ips=False)
    validate_resolved_ip("192.168.1.1", policy)  # Does not raise
```

## Snapshot Testing with Syrupy

Snapshot testing captures complex output (JSON schemas, serialized structures, graphs) and verifies changes explicitly.

### Basic Snapshot Test

```python
from syrupy.assertion import SnapshotAssertion

def test_runnable_schema(snapshot: SnapshotAssertion) -> None:
    fake = FakeRunnable()  # str -> int
    
    schema = fake.get_input_jsonschema()
    assert schema == snapshot
```

On first run, syrupy stores the snapshot in `__snapshots__/`. On subsequent runs, it compares against stored snapshots. Update snapshots with:

```bash
pytest --snapshot-update
```

### Snapshot Updates and Review

Snapshots enable tests to detect unintended changes while allowing intentional updates:

- Snapshots are stored in `__snapshots__/` directories adjacent to test files
- `--snapshot-warn-unused` flags snapshots no longer referenced by tests
- Use `--snapshot-update` to commit intentional schema or output changes
- Always review snapshot diffs to catch unintended regressions

### Example: Runnable Graph Snapshots

Tests for Runnable graph visualization use snapshots to verify Mermaid diagram output:

```python
def test_graph_mermaid(snapshot: SnapshotAssertion) -> None:
    runnable = prompt | chat | parser
    graph_str = runnable.get_graph().draw_mermaid()
    assert graph_str == snapshot
```

## Test Determinism and Reproducibility

### Deterministic UUIDs

Tests use the `deterministic_uuids` fixture to replace random UUIDs:

```python
def test_with_deterministic_ids(deterministic_uuids):
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    assert id1 == UUID("00000000-0000-4000-8000-000000000000")
    assert id2 == UUID("00000000-0000-4000-8000-000000000001")
```

This ensures trace IDs, run IDs, and message IDs are predictable for snapshot tests.

### Time Freezing

`freezegun` freezes time at specific moments for deterministic timestamps:

```python
from freezegun import freeze_time

@freeze_time("2023-01-01 12:00:00")
def test_timestamp_in_trace() -> None:
    tracer = MyTracer()
    # All timestamps in recorded traces will use frozen time
```

### Fake Tracer for Testing

Tests define a `FakeTracer` that captures execution traces and replaces run IDs with deterministic UUIDs:

```python
class FakeTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []
        self.uuids_map: dict[UUID, UUID] = {}
        self.uuids_generator = (
            UUID(f"00000000-0000-4000-8000-{i:012}", version=4) 
            for i in range(10000)
        )
    
    def _persist_run(self, run: Run) -> None:
        """Persist a run with deterministic UUIDs."""
        self.runs.append(self._copy_run(run))
```

## Message and Tool Testing Patterns

### Message Type Testing

Tests verify message construction, serialization, and content blocks:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call

def test_message_with_tool_calls() -> None:
    msg = AIMessage(
        content=[{"text": "I'll use a tool", "type": "text"}],
        tool_calls=[tool_call(name="search", args={"query": "LLM"}, id="call_1")]
    )
    assert msg.tool_calls[0].name == "search"
    assert msg.tool_calls[0].args == {"query": "LLM"}
```

### Tool Creation and Testing

Tests verify tool schema, execution, and async variants:

```python
from langchain_core.tools import tool, BaseTool, StructuredTool

@tool
def search_api(query: str) -> str:
    """Search the API for the query."""
    return f"Results for {query}"

def test_tool_invocation() -> None:
    result = search_api.invoke({"query": "LLM"})
    assert "LLM" in result

async def test_tool_async() -> None:
    if hasattr(search_api, "_arun"):
        result = await search_api.ainvoke({"query": "LLM"})
        assert "LLM" in result
```

Tool schema testing with structured arguments:

```python
from pydantic import BaseModel

class SearchSchema(BaseModel):
    query: str
    limit: int = 10

search_tool = StructuredTool.from_function(
    func=search_api,
    name="search",
    args_schema=SearchSchema
)

def test_tool_schema() -> None:
    schema = search_tool.tool_call_schema
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]
```

## Runnable Testing Patterns

### Testing Invoke

Test synchronous invocation with various input types:

```python
def test_prompt_invoke() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    result = prompt.invoke({"name": "World"})
    assert isinstance(result, ChatPromptValue)
    assert "World" in str(result)
```

### Testing Stream

Test streaming output in chunks:

```python
def test_prompt_stream() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    chunks = list(prompt.stream({"name": "World"}))
    assert len(chunks) > 0
    assert isinstance(chunks[0], ChatPromptValue)
```

### Testing Batch

Test batch processing of multiple inputs:

```python
def test_prompt_batch() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    results = prompt.batch([{"name": "Alice"}, {"name": "Bob"}])
    assert len(results) == 2
    assert "Alice" in str(results[0])
    assert "Bob" in str(results[1])
```

### Testing Async Variants

Test async invoke, stream, and batch:

```python
async def test_prompt_ainvoke() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    result = await prompt.ainvoke({"name": "World"})
    assert isinstance(result, ChatPromptValue)

async def test_prompt_astream() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    chunks = [part async for part in prompt.astream({"name": "World"})]
    assert len(chunks) > 0

async def test_prompt_abatch() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    results = await prompt.abatch([{"name": "Alice"}, {"name": "Bob"}])
    assert len(results) == 2
```

### Configuration and Metadata Testing

Tests verify config merging, metadata propagation, and recursion limits:

```python
def test_config_with_metadata(mocker: MockerFixture) -> None:
    spy = mocker.spy(fake.__class__, "invoke")
    
    fake.invoke("input", {"metadata": {"a": "b"}})
    
    spy.assert_called_once_with(
        "input", {"tags": [], "metadata": {"a": "b"}}
    )
```

### Middleware Testing Patterns

Middleware in LangChain is implemented by wrapping runnables in chains. Test middleware behavior by composing runnables and verifying state propagation:

```python
def test_middleware_state_propagation() -> None:
    """Test that wrapping runnables propagates config through the chain."""
    
    class StateTrackingRunnable(Runnable):
        def invoke(self, input, config=None):
            # Verify config reaches wrapped runnable
            assert config is not None
            assert config.get("metadata", {}).get("request_id") is not None
            return input
    
    # Wrap middleware around core runnable
    wrapped = RunnablePassthrough() | StateTrackingRunnable()
    
    result = wrapped.invoke(
        "test",
        config={"metadata": {"request_id": "123"}}
    )
    assert result == "test"

def test_middleware_chain_with_callbacks() -> None:
    """Test middleware-like callback behavior through chain composition."""
    
    call_log = []
    
    class LoggingMiddleware(RunnableLambda):
        def __init__(self, runnable):
            self.runnable = runnable
            super().__init__(lambda x: runnable.invoke(x))
        
        def invoke(self, input, config=None):
            call_log.append(("before", input))
            result = self.runnable.invoke(input, config)
            call_log.append(("after", result))
            return result
    
    llm = FakeListChatModel(responses=["Hello"])
    middleware = LoggingMiddleware(llm)
    
    middleware.invoke([HumanMessage(content="Hi")])
    
    assert call_log[0] == ("before", [HumanMessage(content="Hi")])
    assert call_log[1][0] == "after"
```

## Edge Cases and Error Conditions

Testing edge cases and error conditions ensures components fail safely and provide meaningful feedback:

### Empty Input Handling

Test how components handle empty or minimal inputs:

```python
def test_prompt_template_empty_input() -> None:
    """Test handling of empty input dict."""
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    
    # Should raise ValueError when required variable is missing
    with pytest.raises(ValueError, match="Missing required"):
        prompt.invoke({})

def test_runnable_batch_empty_list() -> None:
    """Test batch processing with empty input list."""
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    results = prompt.batch([])
    assert results == []
```

### Malformed Input Validation

Test schema validation and type checking:

```python
def test_tool_invalid_argument_types() -> None:
    """Test tool execution with wrong argument types."""
    @tool
    def add(a: int, b: int) -> int:
        return a + b
    
    # Should fail validation
    with pytest.raises(ValidationError):
        add.invoke({"a": "not_a_number", "b": 5})

def test_message_invalid_content_block() -> None:
    """Test message construction with invalid content blocks."""
    with pytest.raises(ValidationError):
        AIMessage(
            content=[{"type": "invalid_type", "data": "content"}]
        )
```

### Error Propagation

Test that errors are propagated correctly through chains:

```python
def test_chain_error_propagation() -> None:
    """Test that errors in later stages propagate through earlier stages."""
    
    def failing_function(input_dict):
        raise ValueError("Intentional failure")
    
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    failing_runnable = RunnableLambda(failing_function)
    chain = prompt | failing_runnable
    
    with pytest.raises(ValueError, match="Intentional failure"):
        chain.invoke({"name": "World"})
```

### Timeout and Cancellation

Test handling of long-running operations:

```python
async def test_runnable_cancellation() -> None:
    """Test that tasks can be cancelled safely."""
    import asyncio
    
    async def slow_runnable(input):
        await asyncio.sleep(10)
        return "done"
    
    runnable = RunnableLambda(slow_runnable)
    task = asyncio.create_task(runnable.ainvoke("test"))
    
    # Cancel the task
    task.cancel()
    
    with pytest.raises(asyncio.CancelledError):
        await task
```

## Test Markers and Filtering

### `requires` Marker

Tests requiring optional packages use the `requires` marker:

```python
@pytest.mark.requires("langchain-openai")
def test_with_openai_integration():
    from langchain_openai import ChatOpenAI
    # ...
```

The conftest checks if packages are installed and skips tests if missing:

```python
if pkg not in required_pkgs_info:
    installed = util.find_spec(pkg) is not None
    required_pkgs_info[pkg] = installed

if not required_pkgs_info[pkg]:
    item.add_marker(pytest.mark.skip(reason=f"Requires pkg: `{pkg}`"))
```

### Filtering Test Execution

```bash
# Run only core tests (skip all @pytest.mark.requires tests)
pytest --only-core

# Run only extended tests (fail if required packages missing)
pytest --only-extended

# Run single test file
pytest tests/unit_tests/runnables/test_runnable.py

# Run tests matching pattern
pytest tests/unit_tests/ -k "test_invoke"
```

## Performance Testing

### Benchmark Tests

pytest-benchmark measures execution time for performance regressions:

```python
def test_sync_callback_performance(benchmark: BenchmarkFixture) -> None:
    model = GenericFakeChatModel(messages=infinite_cycle)
    
    @benchmark
    def run_with_callbacks() -> None:
        for _ in range(5):
            for _ in model.stream("meow", {"callbacks": [MyHandler()]}):
                pass
```

Benchmarks run separately and are disabled in standard test runs:

```bash
pytest --benchmark-disable  # Standard tests (benchmarks disabled)
pytest tests/benchmarks --codspeed  # Performance tests
```

### Continuous Benchmarking

`pytest-codspeed` integrates with CodSpeed for continuous performance tracking and regression detection across commits.

## Test Execution and CI Integration

### Local Test Execution

#### Standard Test Run

```bash
cd libs/core
make test
```

This command (via `Makefile`):
- Unsets LangSmith environment variables (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, etc.) to prevent accidental tracing
- Runs pytest with parallel execution (`-n auto`) across all CPU cores
- Disables socket access (`--disable-socket`) to prevent accidental network calls
- Allows Unix sockets (`--allow-unix-socket`) for inter-process communication
- Disables benchmarks (`--benchmark-disable`) for faster feedback

#### Watch Mode

```bash
make test_watch
```

Watches test files for changes and reruns affected tests with:
- Snapshot auto-update enabled (`--snapshot-update`)
- Verbose output for detailed feedback
- Unix socket support for local testing

#### Extended Tests

```bash
make extended_tests
```

Runs tests marked with `@pytest.mark.requires` to verify optional dependencies:
- Fails if required packages are not installed
- Allows testing against all available integrations
- Useful for testing in CI or before releases

### GitHub Actions CI Integration

Tests run in `.github/workflows/_test.yml` with the following stages:

1. **Environment Setup**
   - Checkout code
   - Install Python version (e.g., 3.10, 3.11, 3.12)
   - Cache dependencies with UV

2. **Current Dependency Testing**
   - Install test dependencies (`uv sync --group test`)
   - Run full test suite with `make test`
   - Report test failures and coverage

3. **Minimum Dependency Testing**
   - Calculate minimum supported versions using `.github/scripts/get_min_versions.py`
   - Reinstall dependencies at minimum versions
   - Rerun full test suite to verify backward compatibility

4. **Validation**
   - Verify working directory is clean (no uncommitted changes from test artifacts)
   - Check that test execution didn't modify source files

#### CI Test Artifacts and Reporting

- Test output is captured with `-q` flag for quiet mode in CI
- Test durations are reported via `--durations=5` to identify slow tests
- Snapshot changes are detected and reported as test failures
- Test runs timeout after 20 minutes to prevent hanging jobs

### Pre-commit Hook Integration

Tests are integrated into pre-commit hooks (`.pre-commit-config.yaml`):

```yaml
- repo: local
  hooks:
    - id: core
      name: format and lint core
      language: system
      entry: make -C libs/core format lint
      files: ^libs/core/
      pass_filenames: false
```

When running `git commit`:
1. Pre-commit hooks automatically run `make format lint` for changed files
2. This includes running code formatters (ruff) and linters
3. Tests are NOT run on commit (only unit test changes trigger test runs in CI)
4. Use `git commit --no-verify` to skip hooks if necessary

## Standard Tests and Cross-Provider Conformance

LangChain uses **standard-tests** (in `libs/standard-tests/`) for cross-provider conformance testing. Partner integrations inherit test suites to verify consistency across providers:

```python
# In langchain-openai or other partner packages
from langchain_tests.standard_tests import ChatModelTests

class TestChatOpenAI(ChatModelTests):
    """Verify OpenAI's ChatOpenAI meets standard chat model interface."""
    
    @property
    def chat_model_class(self):
        return ChatOpenAI
    
    @property
    def chat_model_params(self):
        return {"model": "gpt-4"}
```

Standard tests verify:
- Input/output types match documented interfaces
- Async variants work correctly
- Streaming behaves consistently
- Tool calling follows conventions
- Configuration propagates properly

## Best Practices

1. **Use Fake Models** – Replace real LLM calls with `FakeListChatModel`, `GenericFakeChatModel` to avoid API costs and network dependency
2. **Enable Network Isolation** – Always run with `--disable-socket` to catch accidental network calls
3. **Snapshot Critical Outputs** – Use syrupy for JSON schemas, serialized structures, and graph visualizations
4. **Deterministic Tests** – Use `deterministic_uuids`, `freezegun`, and fake models to ensure reproducible snapshots
5. **Test Async Thoroughly** – Test both sync and async variants of public methods (invoke/ainvoke, stream/astream, batch/abatch)
6. **Mock External Dependencies** – Use `mocker.Mock()` and `mocker.spy()` to isolate units under test
7. **Organize by Module** – Place tests in `tests/unit_tests/` mirroring source structure, with module-specific conftest files
8. **Document Markers** – Define custom pytest markers in conftest and document their meaning
9. **Parallelize Safely** – Use pytest-xdist (`-n auto`) for speed, ensuring tests don't share mutable state or global config
10. **Review Snapshot Changes** – Always review snapshot diffs when updating to catch unintended regressions

## Related Pages

- [Integration Testing](/openwiki/testing/integration-tests.md) – Testing with external services, VCR cassettes, and real APIs
- [Development Commands](/openwiki/operations/dev-commands.md) – uv, make, and pre-commit setup
