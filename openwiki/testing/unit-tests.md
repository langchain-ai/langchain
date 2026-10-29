---
type: "Reference"
title: "Run only core tests (skip all @pytest.mark.requires tests)"
openwiki_generated: true
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
  - id: openwiki-source-344bd4b667096c3c45c8fa82
    resource: repo://libs/core/tests/unit_tests/runnables/conftest.py
  - id: openwiki-source-4717abc86db20c5c76bbf23a
    resource: repo://libs/core/tests/unit_tests/runnables/test_runnable.py
  - id: openwiki-source-f1101c726c80bfb588acd467
    resource: repo://libs/core/tests/unit_tests/test_ssrf_protection.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
---


## Overview

LangChain's unit test suite validates core components in isolation using **pytest** with specialized plugins for network isolation, async support, mocking, and snapshot testing. Tests enforce determinism, reproducibility, and rapid feedback through fixtures, markers, and careful configuration.

## Test Organization and Structure

### Directory Layout

Tests in `libs/core/tests/` follow this hierarchy:

- **`unit_tests/`** – Core component tests organized by module (`runnables/`, `callbacks/`, `language_models/`, etc.)
- **`integration_tests/`** – Tests requiring external services (marked with `@pytest.mark.requires`)
- **`benchmarks/`** – Performance tests using pytest-benchmark and pytest-codspeed

Each major module (e.g., `runnables/`, `callbacks/`) has its own `conftest.py` to define module-specific fixtures and configuration.

### conftest.py Hierarchy

Pytest discovers and applies `conftest.py` files from parent directories up through the test file itself:

1. **Root conftest** (`tests/unit_tests/conftest.py`) – Global fixtures, markers, CLI options
2. **Module conftest** (e.g., `tests/unit_tests/runnables/conftest.py`) – Shared fixtures for a test module

#### Global Test Configuration

The main `conftest.py` provides:

- **`blockbuster` fixture** – Async blocking detection preventing accidental I/O in async contexts
- **`deterministic_uuids` fixture** – Replaces `uuid.uuid4()` with deterministic values for reproducible snapshots
- **Custom markers and CLI options** – `--only-extended`, `--only-core` to filter test execution
- **`requires` marker** – Conditional test skipping when optional dependencies are missing

Example:
```python
@pytest.mark.requires("optional_package")
def test_feature():
    # Test is skipped if optional_package is not installed
    pass
```

#### Module-Level Configuration

Runnables module (`conftest.py`) isolates tests from developer LangSmith environment variables to prevent tracing leakage:

```python
@pytest.fixture(scope="session", autouse=True)
def _disable_local_langsmith_tracing():
    """Keep runnable unit tests independent of developer LangSmith env vars."""
    # Disables LANGSMITH_TRACING, clears API key, project, etc.
```

## Pytest Configuration

### Core Settings

pytest configuration in `pyproject.toml`:

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

- **`asyncio_mode = "auto"`** – Automatically marks async test functions and fixtures as asyncio tests
- **`asyncio_default_fixture_loop_scope = "function"`** – Fresh event loop per test function
- **`--snapshot-warn-unused`** – Warns about unused snapshot files during test runs
- **`--strict-markers`** – Rejects undefined markers to catch typos

### Execution with Network Isolation

The standard test command enforces network isolation:

```bash
pytest -n auto --benchmark-disable --disable-socket --allow-unix-socket
```

- **`--disable-socket`** – Prevents all network calls (from pytest-socket)
- **`--allow-unix-socket`** – Permits Unix sockets for inter-process communication
- **`-n auto`** – Parallel test execution (pytest-xdist)

## Testing Frameworks and Dependencies

### Core Testing Packages

| Package | Purpose |
|---------|---------|
| `pytest>=9.0.3` | Test runner and framework |
| `pytest-asyncio>=1.3.0` | Async/await test support |
| `pytest-mock>=3.10.0` | `MockerFixture` for mocking |
| `pytest-socket>=0.7.0` | Network call blocking |
| `pytest-xdist>=3.6.1` | Parallel test execution |
| `syrupy>=5.0.0` | Snapshot testing |
| `freezegun>=1.2.2` | Time mocking for determinism |
| `responses>=0.25.0` | HTTP request mocking |
| `blockbuster>=1.5.18` | Async blocking detection |
| `pytest-benchmark` | Performance micro-benchmarks |
| `pytest-codspeed` | Continuous benchmarking |

## Async Testing Patterns

### Basic Async Test

Pytest-asyncio automatically runs async test functions:

```python
async def test_async_invoke() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    result = await prompt.ainvoke({"name": "World"})
    assert result == ChatPromptValue(...)
```

### Async Fixtures

Fixtures can be async and are scoped per function by default:

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

### Context Variables in Async Tests

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
    assert some_var.get() == "on_llm_start"
```

## Mocking Patterns

### Using pytest-mock

The `MockerFixture` provides convenient mock creation and assertion:

```python
def test_with_mock(mocker: MockerFixture) -> None:
    mock_func = mocker.Mock(side_effect=ValueError("error"))
    
    with pytest.raises(ValueError):
        mock_func()
    
    # Verify call arguments
    assert mock_func.call_count == 1
    mock_func.assert_called_once()
```

### Spying on Methods

Use `mocker.spy()` to track calls to real methods:

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

### Mocking LLM Responses

LangChain provides fake language models for testing without API calls:

```python
def test_with_fake_llm() -> None:
    llm = FakeListLLM(responses=["Hello", "World"])
    result = llm.invoke("prompt")
    assert result == "Hello"
    
    result2 = llm.invoke("another prompt")
    assert result2 == "World"
```

Fake chat models return messages:

```python
def test_with_fake_chat_model() -> None:
    chat = FakeListChatModel(responses=["I'm a chatbot"])
    result = chat.invoke([HumanMessage(content="Hi")])
    assert result.content == "I'm a chatbot"
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
    
    # Your code that makes HTTP calls
    result = fetch_data()
    assert result["status"] == "ok"
```

## Network Isolation

### pytest-socket Enforcement

Tests run with `--disable-socket` to prevent unintended network calls. This catches tests that accidentally depend on external services.

### Allowing Specific Network Operations

**Unix sockets** are allowed for local inter-process communication:

```bash
pytest --disable-socket --allow-unix-socket
```

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

On first run, syrupy stores the snapshot. On subsequent runs, it compares against the stored snapshot. Update snapshots with:

```bash
pytest --snapshot-update
```

### Snapshot Updates and Review

Snapshots enable tests to detect unintended changes while allowing intentional updates:

- Snapshots are stored in `__snapshots__/` directories adjacent to test files
- `--snapshot-warn-unused` flags snapshots no longer referenced by tests
- Use `--snapshot-update` to commit intentional schema or output changes

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

Tests use a fixture to replace random UUIDs:

```python
@pytest.fixture
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) 
        for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)
```

This ensures trace IDs, run IDs, and message IDs are predictable in snapshot tests.

### Time Freezing

`freezegun` freezes time for deterministic timestamps:

```python
from freezegun import freeze_time

@freeze_time("2023-01-01 12:00:00")
def test_timestamp_in_trace() -> None:
    tracer = MyTracer()
    # Timestamps in recorded traces will use frozen time
```

### Fake Tracer for Testing

Tests define a `FakeTracer` that captures and normalizes execution traces:

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

## Common Fixtures

### UUID Determinism

```python
def test_with_deterministic_ids(deterministic_uuids):
    # uuid.uuid4() returns sequential UUIDs for snapshots
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    assert id1 == UUID("00000000-0000-4000-8000-000000000000")
```

### Config Handling

Tests verify config merging and preservation:

```python
def test_ensure_config() -> None:
    run_id = str(uuid.uuid4())
    arg = {
        "callbacks": [StdOutCallbackHandler()],
        "metadata": {"foo": "bar"},
        "tags": ["tag1"],
    }
    
    config = ensure_config(cast(RunnableConfig, arg))
    
    # Verify original arg unchanged
    assert arg["callbacks"] is not config["callbacks"]
    # Verify merged config
    assert config["metadata"] == {"foo": "bar"}
```

### Async Callback Manager Testing

```python
async def test_callback_manager():
    manager = AsyncCallbackManager(handlers=[CustomHandler()])
    await manager.on_llm_start({}, ["prompt1", "prompt2"])
    # Verify handler was invoked with multiple prompts
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
pytest --benchmark-disable  # Standard tests
pytest tests/benchmarks --codspeed  # Performance tests
```

### Continuous Benchmarking

`pytest-codspeed` integrates with CodSpeed for continuous performance tracking:

```bash
# Run benchmarks for performance analysis
pytest tests/benchmarks --codspeed
```

## Test Execution and CI Integration

### Standard Test Run

```bash
cd libs/core
make test
```

This command:
- Unsets LangSmith environment variables
- Runs pytest with parallel execution (`-n auto`)
- Disables socket access
- Disables benchmarks

### Watch Mode

```bash
make test_watch
```

Watches test files and reruns affected tests with snapshot auto-update.

### Extended Tests

```bash
make extended_tests
```

Runs tests with `@pytest.mark.requires` only if dependencies are installed.

## Test Patterns in LangChain

### Runnables Testing

Runnable tests verify invoke, stream, and async variants with various input types:

```python
def test_prompt_invoke() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    result = prompt.invoke({"name": "World"})
    assert isinstance(result, ChatPromptValue)

def test_prompt_batch() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    results = prompt.batch([{"name": "Alice"}, {"name": "Bob"}])
    assert len(results) == 2

async def test_prompt_astream() -> None:
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    chunks = [part async for part in prompt.astream({"name": "World"})]
    assert len(chunks) > 0
```

### Callback Handler Testing

Tests verify callbacks are invoked with correct arguments:

```python
def test_callback_invocation(mocker: MockerFixture) -> None:
    mock_start = mocker.Mock()
    mock_end = mocker.Mock()
    
    chain = prompt | chat
    
    # Run with callback listener
    chain.invoke(
        {"input": "test"},
        config={"callbacks": [MyCallback(mock_start, mock_end)]}
    )
    
    # Verify callbacks were called
    assert mock_start.called
    assert mock_end.called
```

### Configuration Testing

Tests verify config merging, metadata propagation, and recursion limits:

```python
def test_config_with_metadata(mocker: MockerFixture) -> None:
    spy = mocker.spy(fake.__class__, "invoke")
    
    fake.invoke("input", {"metadata": {"a": "b"}})
    
    spy.assert_called_once_with(
        "input", {"tags": [], "metadata": {"a": "b"}}
    )
```

## Best Practices

1. **Use Fake Models** – Replace real LLM calls with `FakeListLLM`, `FakeListChatModel` to avoid API costs and network dependency
2. **Enable Network Isolation** – Always run with `--disable-socket` to catch accidental network calls
3. **Snapshot Critical Outputs** – Use syrupy for JSON schemas, serialized structures, and graph visualizations
4. **Deterministic Tests** – Use `deterministic_uuids`, `freezegun`, and fake models to ensure reproducible snapshots
5. **Test Async Thoroughly** – Test both sync and async variants of public methods (invoke/ainvoke, stream/astream)
6. **Mock External Dependencies** – Use `mocker.Mock()` and `mocker.spy()` to isolate units under test
7. **Organize by Module** – Place tests in `tests/unit_tests/` mirroring source structure, with module-specific conftest files
8. **Document Markers** – Define custom pytest markers in conftest and document their meaning
9. **Parallelize Safely** – Use pytest-xdist (`-n auto`) for speed, ensuring tests don't share mutable state
10. **Review Snapshot Changes** – Always review snapshot diffs when updating to catch unintended regressions

## Related Pages

- [Integration Testing](/openwiki/testing/integration-tests.md) – Testing with external services and real APIs
- [CI Workflows](/openwiki/operations/ci-workflows.md) – GitHub Actions test execution and validation
- [Development Commands](/openwiki/operations/dev-commands.md) – Running tests and linting locally
