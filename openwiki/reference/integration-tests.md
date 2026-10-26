---
type: Integration Testing Framework
title: Integration Tests
description: LangChain's integration testing framework for partner libraries and chat models, including VCR cassette recording, pytest markers for test classification, and scheduled CI/CD execution against live APIs.
tags: [integration-tests, testing, vcr-cassettes, pytest, ci-cd, chat-models]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-26T08:33:32.568Z
sources:
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-711811fb8df6b32329091d23
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
generated: { by: "openwiki/0.5.0", at: "2026-10-26T08:33:32.568Z" }
---

## Overview

LangChain integration tests exercise partner library implementations and chat models against real APIs in both cassette-backed (recorded) and live modes. The testing infrastructure supports recording HTTP interactions via VCR cassettes, classifying tests by execution context using pytest markers, and orchestrating multi-library testing across various provider implementations through GitHub Actions workflows.

## Test Classification System

Integration tests are classified into categories using pytest markers, enabling selective execution in different contexts:

### VCR-Backed Tests

Tests marked with `@pytest.mark.vcr` use recorded HTTP interactions (VCR cassettes) rather than live API calls. These tests:

- Replay previously recorded HTTP request/response pairs from YAML cassettes stored in `tests/cassettes/`
- Execute in playback-only mode (`--record-mode=none`) during pull request checks via `make test_vcr`
- Support cassette re-recording when test inputs change, with configurable recording behavior
- Require explicit filtering of sensitive headers (authorization, API keys) before cassette storage
- Are performed by the `pytest-recording` plugin using custom VCR configuration from `langchain_tests.conftest`

### Scheduled Live Tests

Tests marked with `@pytest.mark.scheduled` execute against live provider APIs:

- Run on a daily schedule (1 PM UTC / 9 AM EDT / 6 AM PDT) automatically via GitHub Actions
- Can be manually triggered for specific libraries or all libraries via workflow dispatch
- Require valid API credentials injected as GitHub secrets at test time
- Are part of the partner library test matrix alongside unit tests
- Provide continuous validation that implementations remain compatible with upstream provider APIs

### Unmarked Offline Tests

Tests without `@pytest.mark.vcr` or `@pytest.mark.scheduled`:

- Execute as offline unit-like tests without requiring API access
- Run in standard unit test suites with socket networking disabled (`--disable-socket`)
- Test core functionality, error handling, and local behavior independent of API availability
- Form the baseline test suite executable in any environment without secrets or credentials

## VCR Cassette System

The integration testing framework uses VCR (Video Cassette Recorder) to record and replay HTTP interactions:

### Cassette Storage and Format

- Cassettes are stored in `tests/cassettes/` directories within each partner library
- Files are compressed YAML (gzip) format with `.yaml` extension
- Custom `CustomSerializer` and `CustomPersister` classes in `langchain_tests.conftest` handle serialization, ensuring cassettes are human-inspectable after decompression
- Cassette naming corresponds to test module and function paths for easy tracking

### Recording and Playback Configuration

- Default record mode is `'once'` — only records if cassette doesn't exist, otherwise replays
- Playback-only mode (`record_mode='none'`) is used in CI to detect stale cassettes without re-recording
- Matching strategy uses `method`, `uri`, and request `body` to locate matching cassettes
- `allow_playback_repeats` enables multiple replays of the same cassette within a single test session
- Compressed responses are automatically decompressed during playback

### Sensitive Data Filtering

Base filter headers are applied to all cassettes:

- `authorization` → `PLACEHOLDER`
- `x-api-key` → `PLACEHOLDER`
- `api-key` → `PLACEHOLDER`

Additional cassette-specific filtering is configured on a per-partner-library basis. Implementations can override or extend `base_vcr_config()` to add custom filter rules for provider-specific authentication schemes.

## Standard Test Suite Inheritance

Partner libraries inherit integration test implementations from the `langchain-tests` package:

### ChatModelIntegrationTests

The `ChatModelIntegrationTests` base class provides 70+ standard tests for chat model implementations:

- Requires subclasses to implement `chat_model_class` and `chat_model_params` properties
- Tests are selectively enabled based on feature capability properties (e.g., `has_tool_calling`, `supports_image_inputs`)
- VCR-backed tests are opt-in via the `enable_vcr_tests` property (defaults to `False` but enabled in OpenAI implementation)
- Streaming, structured output, tool calling, multi-modality, and advanced features are covered

Example test class from OpenAI:

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
    def enable_vcr_tests(self) -> bool:
        return True
```

## CI/CD Workflow Architecture

### Matrix Computation

The GitHub Actions workflow `/.github/workflows/integration_tests.yml` uses dynamic matrix generation to support:

- **Manual targeting** via dropdown selection of specific partner libraries (openai, anthropic, groq, etc.)
- **Scheduled execution** (daily at 1 PM UTC) against a default set of libraries
- **Exclusion filtering** for the "all" run (e.g., `exclude=openai,anthropic`)
- **Python version selection** with optional overrides (defaults to 3.10 and 3.14; some packages require ≥3.11)

The `compute-matrix` job:

1. Resolves working directories from dropdown choice, override string, or DEFAULT_LIBS
2. Maps short package names to full paths (`openai` → `libs/partners/openai`)
3. Generates a test matrix with all combinations of Python versions and working directories
4. Serializes execution per (working-directory, python-version) to prevent credential contention

### Integration Test Execution

The `integration-tests` job:

1. Checks out the main langchain monorepo and external dependencies (langchain-google, langchain-aws)
2. Reorganizes external libraries into the monorepo structure
3. Runs `uv sync --group test --group test_integration` to install dependencies
4. Injects ALL secrets into the environment (intentionally, for simplicity across all matrix legs)
5. Invokes `make integration_tests` from each target library
6. Verifies the working directory remains clean (no uncommitted changes after tests)

### Dependency Resolution

- Partner packages use `[tool.uv.sources]` in pyproject.toml to resolve local langchain-core and langchain editable installs
- External packages (Google, AWS) live in separate repositories and receive local overlays after `uv sync` to test against the current branch's langchain code
- Python 3.11+ requirement for certain packages (e.g., deepagents in `test-dependents`) is enforced via computed matrix variants

### Tracing and Observability

Test execution is routed to LangSmith for tracing and observability:

- `LANGSMITH_TRACING: true` enables all traces to LangSmith
- `LANGSMITH_PROJECT` defaults to `'scheduled-testing-py'` (can be overridden via workflow variables)
- Tags automatically include GitHub context: `github-actions`, working directory, Python version, and commit SHA
- Metadata (run ID, attempt, URL) is injected for failure investigation and run linkage

## Makefile Integration

Each partner library Makefile exposes:

```makefile
integration_tests:
    uv run --group test --group test_integration pytest ...
    # Full integration test suite against current and other SDK versions
    # Runs both live and cassette-backed tests

test_vcr:
    uv run --group test pytest --record-mode=none -m vcr ...
    # Cassette-only playback tests (no API keys needed)

test:
    uv run --group test pytest --disable-socket ...
    # Unit tests with networking disabled
```

The workflow calls `make integration_tests` to run the full suite in the scheduled/manual contexts where API credentials are available.

## Configuration and Local Testing

### Environment Variables

Local integration test execution requires:

- **API Keys**: Partner-specific keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) in shell environment or `.env` file
- **LangSmith** (optional): LANGSMITH_API_KEY for tracing; tests run without tracing if unset
- **Feature Flags**: Feature-specific keys (e.g., ANTHROPIC_FILES_API_IMAGE_ID for file uploads)

### Running Tests Locally

```bash
# Run full integration tests with all cassette and live tests
cd libs/partners/openai
make integration_tests

# Run only VCR-backed tests (no API keys required)
make test_vcr

# Run specific test with live API
pytest tests/integration_tests/chat_models/test_base.py::test_chat_openai -m scheduled
```

### Cassette Recording

To record new or updated cassettes:

```bash
# Record mode allows writing new cassettes
pytest tests/integration_tests/ --record-mode=new_episodes
```

Recorded cassettes are committed to the repository; team members later use playback-only (`--record-mode=none`) to avoid redundant API calls.

## Best Practices

- **Isolation**: Each test records its cassette independently; use unique test names or configure custom cassette names to prevent collisions
- **Determinism**: Cassette-backed tests must produce identical results across replay; avoid non-deterministic data (timestamps, random values) in assertions
- **Header Filtering**: Configure provider-specific header filters in test conftest files before cassettes are committed
- **Credential Rotation**: If credentials are rotated in a cassette, re-record or update the PLACEHOLDER values manually
- **Debugging**: Decompress cassettes locally (`gzip -d cassette.yaml.gz`) to inspect recorded requests/responses when debugging test failures
- **Scheduling**: Scheduled live tests should verify critical paths and backward compatibility; reserve VCR for tests with stable, deterministic outputs

## Troubleshooting

**Stale Cassettes**: If cassettes are stale (test inputs changed), the `test_vcr` target will fail in CI. Re-record using the full `integration_tests` target and commit the updated cassettes.

**APIConnectionError During Playback**: Ambient environment variables (e.g., `OPENAI_BASE_URL` pointing at a gateway) can break VCR matching. The OpenAI conftest strips these during VCR tests via `_clear_openai_base_url_env` fixture.

**Missing Credentials in Scheduled Tests**: GitHub repository secrets are injected at test time. Verify that all required API keys are configured in the repository's GitHub Actions secrets.

**Credential Scope**: All secrets are available to all matrix jobs (intentional). Only run scheduled tests from the main repository to prevent credential exposure to untrusted code.
