# Global development guidelines for the Orcest Core monorepo

This document provides context to understand the Orcest Core Python project and assist with development.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages managed by `uv`.

```txt
orcest.ai/
├── libs/
│   ├── core/               # `langchain-core` v1.2.13 — primitives and base abstractions
│   ├── langchain/          # `langchain` v1.0.1 (legacy, no new features)
│   ├── langchain_v1/       # `langchain` v1.2.10 (actively maintained main package)
│   ├── text-splitters/     # `langchain-text-splitters` v1.1.0 — document chunking
│   ├── standard-tests/     # `langchain-tests` v1.1.5 — shared test suite for integrations
│   ├── model-profiles/     # `langchain-model-profiles` v0.0.5 — model config profiles (CLI)
│   ├── partners/           # Third-party integrations (15 packages)
│   │   ├── anthropic/      # langchain-anthropic v1.3.3 (Claude)
│   │   ├── openai/         # langchain-openai v1.1.9
│   │   ├── ollama/         # langchain-ollama v1.0.1
│   │   ├── groq/           # langchain-groq v1.1.2
│   │   ├── chroma/         # Chroma vector store
│   │   ├── deepseek/       # Deepseek LLM
│   │   ├── exa/            # Exa search
│   │   ├── fireworks/      # Fireworks AI
│   │   ├── huggingface/    # HuggingFace models
│   │   ├── mistralai/      # Mistral AI
│   │   ├── nomic/          # Nomic embeddings
│   │   ├── openrouter/     # OpenRouter API
│   │   ├── perplexity/     # Perplexity AI
│   │   ├── qdrant/         # Qdrant vector store
│   │   └── xai/            # X AI
│   └── Makefile            # Workspace-level targets (lock, check-lock)
├── app/                    # Application code (main.py)
├── .github/                # CI/CD workflows and templates
├── .vscode/                # VSCode IDE settings and recommended extensions
├── .mcp.json               # MCP server config (docs-langchain)
├── .pre-commit-config.yaml # Pre-commit hooks
├── .editorconfig           # Editor formatting rules
└── README.md
```

**Architectural layers:**

- **Core layer** (`langchain-core`): Base abstractions, interfaces, and protocols. Users should not need to know about this layer directly.
- **Implementation layer** (`langchain`): Concrete implementations and high-level public utilities. `libs/langchain` is legacy (no new features); `libs/langchain_v1` is actively maintained.
- **Integration layer** (`partners/`): Third-party service integrations. This monorepo is not exhaustive; some integrations are in separate repos (e.g., `langchain-ai/langchain-google`, `langchain-ai/langchain-aws`). These repos are typically cloned at the same level as this monorepo, so navigate to `../langchain-google/` from here if needed.
- **Testing layer** (`standard-tests/`): Standardized integration tests for partner packages.

### Python version

All packages require Python `>=3.10.0, <4.0.0`. The CI matrix tests against 3.11 by default, with support for 3.10 through 3.14.

### Build system

All packages use `hatchling` as the build backend.

## Development tools & commands

### Toolchain

| Tool     | Purpose                      | Version constraint      |
|----------|------------------------------|-------------------------|
| `uv`     | Package management & locking | latest                  |
| `ruff`   | Linting & formatting         | `>=0.15.0, <0.16.0` (core/langchain) |
| `mypy`   | Static type checking         | `>=1.19.1, <1.20.0` (core/langchain) |
| `pytest`  | Test framework               | `>=8.0.0` (core) / `>=7.3.0` (partners) |
| `make`   | Task runner                  | —                       |

Note: Partner packages may pin slightly different versions of ruff/mypy. Always check the specific package's `pyproject.toml`.

### Package setup

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`. Local development uses editable installs via `[tool.uv.sources]`.

```bash
# Navigate to the specific package directory first, e.g.:
cd libs/core

# Install all dependency groups
uv sync --all-groups

# Or install a specific group only
uv sync --group test
```

### Common Makefile targets

Every package has a `Makefile` with these standard targets:

| Target               | Description                                    |
|----------------------|------------------------------------------------|
| `make format`        | Run ruff formatter and auto-fix lints          |
| `make lint`          | Run ruff check, ruff format --diff, and mypy   |
| `make lint_package`  | Lint only the package source (not tests)       |
| `make lint_tests`    | Lint only the test code                        |
| `make test`          | Run unit tests with pytest (no network)        |
| `make test_watch`    | Run unit tests in watch mode                   |
| `make integration_tests` | Run integration tests (network allowed)    |
| `make extended_tests`| Run extended test suites                       |
| `make type`          | Run mypy type checking only                    |
| `make help`          | Show available targets                         |

**Core-specific targets:** `check_imports`, `check_version`, `benchmark`

**Workspace-level targets** (from `libs/Makefile`):

```bash
# Regenerate lockfiles for all core packages
make -C libs lock

# Verify all lockfiles are up-to-date
make -C libs check-lock
```

### Running tests

```bash
# Run all unit tests for a package (no network)
make test

# Run a specific test file
uv run --group test pytest tests/unit_tests/test_specific.py

# Run with parallelism (default in core)
uv run --group test pytest -n auto --disable-socket --allow-unix-socket tests/unit_tests/
```

Unit tests use `pytest-socket` to block network calls by default. The following environment variables are explicitly unset during test runs to prevent external service dependencies:
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING`
- `LANGCHAIN_PROJECT`

### Linting and formatting

```bash
# Format code (auto-fix)
make format

# Lint code (check only — runs ruff check, ruff format --diff, mypy)
make lint

# Type checking only
make type
# or
uv run --group lint mypy .
```

### Dependency groups

Standard groups defined in each package's `pyproject.toml`:

| Group              | Purpose                                              |
|--------------------|------------------------------------------------------|
| `test`             | Unit testing (pytest, pytest-asyncio, pytest-xdist, pytest-socket, syrupy, freezegun, blockbuster) |
| `test_integration` | Integration testing dependencies                     |
| `lint`             | Linting tools (ruff)                                 |
| `typing`           | Type checking (mypy, type stubs)                     |
| `dev`              | Development tools (jupyter, setuptools)              |

### Key config files per package

- `pyproject.toml` — Package metadata, dependencies, tool configuration (ruff, mypy, pytest)
- `uv.lock` — Locked dependencies for reproducible builds
- `Makefile` — Development task runner

## CI/CD workflows

The CI system (`.github/workflows/`) only runs against packages with changed files.

| Workflow                   | Trigger              | Purpose                                         |
|----------------------------|----------------------|-------------------------------------------------|
| `check_diffs.yml` (CI)    | push/PR/merge_group  | Primary CI: detects changes, runs lint/test/pydantic/benchmarks |
| `pr_lint.yml`              | PR open/edit/sync    | Validates PR title follows Conventional Commits  |
| `integration_tests.yml`   | Daily schedule (1PM UTC) | Full integration test suite                  |
| `_lint.yml`                | Called by CI          | Shared linting workflow                          |
| `_test.yml`                | Called by CI          | Shared unit test workflow (current + min deps)   |
| `_test_pydantic.yml`      | Called by CI          | Pydantic compatibility testing                   |
| `_compile_integration_test.yml` | Called by CI   | Integration test compilation checks              |
| `_release.yml`             | Manual/release       | Package release and PyPI publishing              |

CI can be skipped by adding the `ci-ignore` label. CodSpeed benchmarks can be skipped with the `codspeed-ignore` label.

## Commit standards

PR titles and commit messages MUST follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/). All titles must be **lowercase** (except proper nouns). A **scope is always required**.

### Format

```
<type>(<scope>): <description>
```

### Allowed types

`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`, `release`

### Allowed scopes

`core`, `langchain`, `langchain-classic`, `model-profiles`, `standard-tests`, `text-splitters`, `docs`, `anthropic`, `chroma`, `deepseek`, `exa`, `fireworks`, `groq`, `huggingface`, `mistralai`, `nomic`, `ollama`, `openai`, `openrouter`, `perplexity`, `qdrant`, `xai`, `infra`, `deps`

### Examples

```txt
feat(langchain): add new chat completion feature
fix(core): resolve type hinting issue in vector store
chore(anthropic): update infrastructure dependencies
release(core): 1.2.13
feat(core,langchain): add multi-tenant support
feat!: drop Python 3.9 support
```

Note: `feat(langchain)` includes a scope even though it is the main package and name of the repo. Multiple scopes are separated by commas.

## Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.
You should warn the developer for any function signature changes, regardless of whether they look breaking or not.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the `known_users` set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense
- Relative imports are banned — use absolute imports only (`ban-relative-imports = "all"`)
- Google-style pydocstyle convention is enforced

### Ruff configuration

All packages select `ALL` rules with a curated ignore list. Key shared settings:

- Docstring code formatting is enabled
- Google pydocstyle convention with `ignore-var-parameters = true`
- Relative imports are banned
- `flake8-builtins` allows `id`, `input`, `type` as variable names (core)
- Test files ignore: `D1` (missing docstrings), `S101` (assertions), `SLF001` (private access), `PLR2004` (magic values)

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.
- `asyncio_mode = "auto"` is set across all packages — async tests run automatically.

**Key pytest plugins used:** `pytest-asyncio`, `pytest-xdist` (parallel), `pytest-socket` (network blocking), `pytest-mock`, `syrupy` (snapshot testing), `freezegun` (time mocking), `blockbuster`, `pytest-cov`

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)
- [ ] Does the test suite fail if your new logic is broken?

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")
- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.

## Pre-commit hooks

The `.pre-commit-config.yaml` enforces:

1. **General checks:** no direct commits to `master`, YAML/TOML validation, trailing whitespace removal, final newline
2. **Text normalization:** fix curly quotes to straight quotes, fix non-standard spaces
3. **Per-package hooks:** runs `make format lint` scoped to each package's directory (core, langchain, standard-tests, text-splitters, and 11 partner packages)
4. **Version consistency:** checks `core` and `langchain_v1` version fields match between `pyproject.toml` and source files

## Editor configuration

`.editorconfig` enforces:
- UTF-8 charset, LF line endings, final newline
- Python: 4-space indent, 88 char line length
- JSON: 2-space indent
- YAML: 2-space indent
- Makefile: tab indent

VSCode settings (`.vscode/`):
- Ruff as default Python formatter with format-on-save
- Pylance for analysis (basic type checking mode)
- Default interpreter: `./.venv/bin/python`
- Recommended extensions: Python, Ruff, mypy, Jupyter, GitLens, GitHub Actions, and more

## Additional resources

- **Documentation:** https://docs.langchain.com/oss/python/langchain/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview)
- **MCP Server:** docs-langchain at `https://docs.langchain.com/mcp` (configured in `.mcp.json`)
