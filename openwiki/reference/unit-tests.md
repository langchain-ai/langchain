---
type: Reference
title: Unit Tests Reference
description: Reference documentation for test fixtures, mocking utilities, assertion helpers, and testing infrastructure used in LangChain unit tests.
tags: [testing, fixtures, pytest, unit-tests, mocking]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-727aef6a92fb635fdbb41cd6
    resource: repo://libs/core/tests/unit_tests/conftest.py
  - id: openwiki-source-5e13d2c899eb5925ef28fddf
    resource: repo://libs/core/tests/unit_tests/fake/callbacks.py
  - id: openwiki-source-f0e376fe9b6befdcc2505465
    resource: repo://libs/core/tests/unit_tests/pydantic_utils.py
  - id: openwiki-source-344bd4b667096c3c45c8fa82
    resource: repo://libs/core/tests/unit_tests/runnables/conftest.py
  - id: openwiki-source-5839db669f618a6d604790ca
    resource: repo://libs/core/tests/unit_tests/stubs.py
  - id: openwiki-source-e8b4a2351d42b9ac7cbb0196
    resource: repo://libs/langchain_v1/tests/unit_tests/conftest.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
  - id: openwiki-source-db02c1dda8563ab005cd9d62
    resource: repo://libs/standard-tests/langchain_tests/conftest.py
  - id: openwiki-source-3eb9100e02f9d70098d1b30d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py
  - id: openwiki-source-a6b31954b6df57580d0f3ed0
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

LangChain's unit test infrastructure provides a comprehensive suite of fixtures, utilities, and standard test classes to support consistent testing across the codebase. This reference documents the key testing infrastructure components, fixtures, and patterns available to test authors.

## Test Configuration and Hooks

### Core Conftest Configuration

The conftest files at `repo://libs/core/tests/unit_tests/conftest.py` and `repo://libs/langchain_v1/tests/unit_tests/conftest.py` establish global test configuration and behavior.

**blockbuster fixture** – Automatically injected by autouse fixture into all tests. The `blockbuster` fixture uses the BlockBuster library to prevent blocking I/O operations during async tests and validate socket restrictions. It configures allowed blocking contexts for specific functions in internal modules:
- `os.stat` and `os.path.abspath` allowed in `langchain_core/_api/internal.py` and `langchain_core/runnables/base.py`
- `os.stat` and `io.TextIOWrapper.read` allowed in langsmith client code
- All blocking operations allowed in freezegun during time mocking

Configuration is established via `blockbuster_ctx()` which yields a BlockBuster instance that enforces these constraints throughout the test.

**Custom command-line options** – pytest extends argument parsing to support:
- `--only-extended`: Run only tests marked with `@pytest.mark.requires()`. Fails if required packages are not installed.
- `--only-core`: Run only core tests, skipping all extended tests that require optional dependencies.

These options are mutually exclusive and cause an error if both are specified.

**requires marker processing** – The `pytest_collection_modifyitems` hook handles the `@pytest.mark.requires()` marker for conditional test skipping based on installed packages. When a test is marked with `@pytest.mark.requires("package1", "package2")`:
- The hook checks if each package is installed using `importlib.util.find_spec()`
- Results are cached to avoid repeated lookups
- Under `--only-extended`, missing packages cause test failure
- Under `--only-core` or default mode, tests are skipped if any required package is missing
- Under default mode with all packages present, tests run normally

## Fixtures

### Deterministic UUID Generation

**deterministic_uuids fixture** (`repo://libs/core/tests/unit_tests/conftest.py#L115-L120`) – Provides a mocker-based fixture that generates sequential UUIDs for test reproducibility. When invoked, it returns a mocker that patches `uuid.uuid4` to yield UUIDs in sequence: `00000000-0000-4000-8000-000000000000`, `00000000-0000-4000-8000-000000000001`, etc., up to 10,000 unique values. This enables deterministic test snapshots and comparisons.

### LangSmith Tracing Isolation

**_disable_local_langsmith_tracing fixture** (`repo://libs/core/tests/unit_tests/runnables/conftest.py#L10-L37`) – A session-scoped autouse fixture that isolates runnable unit tests from developer LangSmith environment variables. It captures the current state of environment variables (LANGCHAIN_TESTS_USER_AGENT, LANGCHAIN_TRACING_V2, LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING, and others), disables tracing by setting LANGSMITH_TRACING to "false", removes other LangSmith env vars, and restores the original state after tests complete. It also clears any cached environment variable lookups via `get_env_var.cache_clear()` if available.

### Model and Tool Fixtures

**model fixture** (`repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py#L68-L78`) – Provides a configured instance of a chat model for tests. The fixture receives initialization parameters from `standard_chat_model_params` (temperature, max_tokens, timeout, stop, max_retries) merged with model-specific `chat_model_params` and any pytest parametrization overrides from the test request.

**my_adder_tool fixture** (`repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py#L80-L92`) – A simple tool fixture decorated with `@tool` that adds two integers. Used for testing tool calling and invocation patterns.

**tool fixture** (`repo://libs/standard-tests/langchain_tests/unit_tests/tools.py#L44-L55`) – Provides an instance of a tool for testing. If `tool_constructor` is already a BaseTool instance, returns it directly (requiring empty `tool_constructor_params`). If `tool_constructor` is a class, instantiates it with `tool_constructor_params`.

### VCR Configuration (langchain_v1)

**vcr_config fixture** (`repo://libs/langchain_v1/tests/unit_tests/conftest.py#L41-L51`) – Session-scoped fixture that extends the default VCR configuration from `langchain_tests.conftest.base_vcr_config()`. It configures:
- JSON body matching instead of raw body matching for requests
- Header filtering to redact sensitive headers (openai-organization, user-agent, x-openai-client-user-agent)
- YAML gzip serialization and deserialization for cassettes
- Request and response processors to redact headers and URIs

**pytest_recording_configure hook** (`repo://libs/langchain_v1/tests/unit_tests/conftest.py#L71-L74`) – Registers custom persister (`CustomPersister`), serializer (`CustomSerializer`), and matcher (`json_body_matcher`) with VCR for gzip-compressed YAML cassette handling and JSON-aware request matching.

## Utility Classes and Helpers

### Fake Callback Handler

**FakeCallbackHandler** (`repo://libs/core/tests/unit_tests/fake/callbacks.py#L119-L180`) – A concrete implementation of `BaseCallbackHandler` and `BaseFakeCallbackHandlerMixin` for testing. It tracks callback invocations by maintaining counters for:
- Overall lifecycle: starts, ends, errors
- LLM callbacks: llm_starts, llm_ends, llm_streams
- Chain callbacks: chain_starts, chain_ends
- Tool callbacks: tool_starts, tool_ends
- Agent callbacks: agent_actions, agent_ends
- Retriever callbacks: retriever_starts, retriever_ends, retriever_errors
- Retry callbacks: retries

It supports ignoring specific callback categories (llm, chain, agent, retriever) via boolean flags and supports a `fake_id` for test identification.

**BaseFakeCallbackHandlerMixin** (`repo://libs/core/tests/unit_tests/fake/callbacks.py#L48-L116`) – Provides common callback mixin methods that increment counters. Each lifecycle event method updates both a specific counter (e.g., `llm_starts`) and a general counter (e.g., `starts`).

### Pydantic Utilities

**AnyStr** (`repo://libs/core/tests/unit_tests/stubs.py#L7-L13`) – A string subclass that compares equal to any other string. Used to match id fields in generated messages and documents where the actual ID value is non-deterministic but the type and presence matter.

**Message creators with flexible IDs** (`repo://libs/core/tests/unit_tests/pydantic_utils.py#L24-L49`) – Helper functions to create messages with id fields set to `AnyStr()`:
- `_any_id_document(**kwargs)` – Creates a Document with flexible ID
- `_any_id_ai_message(**kwargs)` – Creates an AIMessage with flexible ID
- `_any_id_ai_message_chunk(**kwargs)` – Creates an AIMessageChunk with flexible ID
- `_any_id_human_message(**kwargs)` – Creates a HumanMessage with flexible ID

These functions work around a Pydantic issue where custom `__eq__` methods on string subclasses are ignored if set before model creation.

**Pydantic schema utilities** (`repo://libs/core/tests/unit_tests/pydantic_utils.py#L18-L100`) – Helper functions for normalizing Pydantic schemas across v1 and v2:
- `replace_all_of_with_ref(schema)` – Converts `allOf` with single `$ref` to direct `$ref` for schema compatibility
- `remove_all_none_default(schema)` – Removes incorrect None defaults that Pydantic v2 generates for NotRequired fields
- `_remove_enum(obj)` – Removes enum and type information for enum fields
- `_schema(obj)` – Generates normalized JSON schema from BaseModel subclasses, handling both v1 and v2

**skip_if_no_pydantic_v1** (`repo://libs/core/tests/unit_tests/pydantic_utils.py#L11-L14`) – Pytest marker that skips tests on Python 3.14+ where pydantic.v1 compatibility shim is not available.

## Standard Test Classes

### ChatModelTests

**ChatModelTests** (`repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py#L43-L178`) – Base class for testing chat model implementations. Requires subclasses to:
- Define `chat_model_class` property returning the model class to test
- Optionally override `chat_model_params` with model-specific initialization parameters
- Optionally override `standard_chat_model_params` for different standard parameters

Provides fixtures:
- `model` – Instance of chat_model_class with merged parameters
- `my_adder_tool` – Basic tool for testing tool calling

Offers properties to detect model capabilities:
- `has_tool_calling` – Whether model supports tool calling (checks for non-default `bind_tools` override)
- `has_tool_choice` – Whether model supports tool_choice parameter
- `has_structured_output` – Whether model supports structured output or tool calling

### ToolsTests and ToolsUnitTests

**ToolsTests** (`repo://libs/standard-tests/langchain_tests/unit_tests/tools.py#L17-L56`) – Base class for tool testing. Requires:
- `tool_constructor` property returning a tool class or instance
- Optionally `tool_constructor_params` for class constructor arguments
- Optionally `tool_invoke_params_example` for example invocation arguments (not a ToolCall dict)

Provides the `tool` fixture that instantiates the tool if needed.

**ToolsUnitTests** (`repo://libs/standard-tests/langchain_tests/unit_tests/tools.py#L58-L180`) – Extends ToolsTests with standard unit test methods:
- `test_init` – Verifies tool instantiation with provided parameters
- `test_init_from_env` – Tests initialization from environment variables (override `init_from_env_params` to provide env vars, init kwargs, and expected attributes)
- `test_has_name` – Verifies tool has a name attribute
- `test_has_input_schema` – Verifies tool has input schema via `get_input_schema()`
- `test_input_schema_matches_invoke_params` – Validates example invoke params match declared schema

## VCR Cassette Handling

### CustomSerializer and CustomPersister

**CustomSerializer** (`repo://libs/standard-tests/langchain_tests/conftest.py#L47-L82`) – Custom VCR serializer that:
- Serializes cassettes to YAML then compresses with gzip
- Converts `io.BytesIO` objects to raw bytes (recursively through nested structures)
- Deserializes gzip-compressed YAML back to cassette dicts
- Reconstructs `Request` objects from cassette data

Addresses the fact that the default VCR YAML serializer is unsafe for untrusted input.

**CustomPersister** (`repo://libs/standard-tests/langchain_tests/conftest.py#L85-L110`) – Custom VCR persister that works with CustomSerializer to load and save cassettes from/to the filesystem as gzip-compressed YAML.

### Request and Response Processing

**remove_request_headers** (`repo://libs/langchain_v1/tests/unit_tests/conftest.py#L26-L31`) – Callback that redacts all request headers by replacing values with "**REDACTED**" and replaces the URI with "**REDACTED**". Used as `before_record_request` in VCR configuration.

**remove_response_headers** (`repo://libs/langchain_v1/tests/unit_tests/conftest.py#L34-L38`) – Callback that redacts all response headers similarly. Used as `before_record_response` in VCR configuration.

**_json_body_matcher** (`repo://libs/langchain_v1/tests/unit_tests/conftest.py#L54-L68`) – Custom VCR matcher that compares request bodies as parsed JSON objects, ignoring key order. Falls back to byte comparison for non-JSON bodies.

## Testing Patterns

### Conditional Test Skipping

Tests requiring optional dependencies use the `@pytest.mark.requires()` decorator:

```python
@pytest.mark.requires("langchain_openai", "langchain_anthropic")
def test_something(): ...
```

This allows tests to be skipped when dependencies are unavailable, or to fail during extended test runs that enforce all dependencies.

### Deterministic Snapshots

The `deterministic_uuids` fixture enables reliable snapshot testing when UUIDs are generated during test execution:

```python
def test_snapshot(deterministic_uuids):
    result = some_function_that_generates_uuids()
    assert result == expected_snapshot  # UUIDs are sequential and predictable
```

### Standard Test Inheritance

Custom test classes inherit from the standard test base classes and implement required properties. The standard test base automatically validates that no standard tests are overridden without `@pytest.mark.xfail(reason="...")`:

```python
class TestCustomTool(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return MyCustomTool
    
    @property
    def tool_constructor_params(self):
        return {"api_key": "test-key"}
```

### Async Testing

For async tests, standard async callback handlers and async fixtures are available through langchain_core. The blockbuster fixture validates that async event loops are not blocked by I/O operations.

## Key Invariants and Constraints

- The blockbuster fixture prevents unintended blocking I/O in async contexts and fails tests that violate this invariant
- Fake callback handlers maintain strict counter consistency: `starts` equals the sum of all category-specific starts
- The `requires` marker enforces optional dependency availability, preventing silent test skips in extended runs
- Standard test classes enforce that subclasses do not override inherited tests without explicit xfail markers with reasons
- VCR cassettes are compressed to reduce repository size while remaining inspectable via custom serialization
- Pydantic v1 and v2 schema normalization utilities ensure consistent schema representation across Python versions

## Extension Points

- Subclass `BaseStandardTests` to create custom standard test classes
- Override `chat_model_params` or `tool_constructor_params` to customize initialization for specific implementations
- Implement `init_from_env_params` to test environment variable configuration patterns
- Create custom callback handlers by subclassing `BaseCallbackHandler` with `BaseFakeCallbackHandlerMixin`
- Use `pytest.mark.requires()` to conditionally require optional dependencies for specific tests
