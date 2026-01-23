# Global development guidelines for the LangChain monorepo

This document provides context to understand the LangChain Python project and assist with development.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages that use `uv`.

```txt
langchain/
├── libs/
│   ├── core/             # `langchain-core` primitives and base abstractions
│   ├── langchain/        # `langchain-classic` (legacy, no new features)
│   ├── langchain_v1/     # Actively maintained `langchain` package
│   ├── partners/         # Third-party integrations
│   │   ├── openai/       # OpenAI models and embeddings
│   │   ├── anthropic/    # Anthropic (Claude) integration
│   │   ├── ollama/       # Local model support
│   │   └── ... (other integrations maintained by the LangChain team)
│   ├── text-splitters/   # Document chunking utilities
│   ├── standard-tests/   # Shared test suite for integrations
│   ├── model-profiles/   # Model configuration profiles
├── .github/              # CI/CD workflows and templates
├── .vscode/              # VSCode IDE standard settings and recommended extensions
└── README.md             # Information about LangChain
```

- **Core layer** (`langchain-core`): Base abstractions, interfaces, and protocols. Users should not need to know about this layer directly.
- **Implementation layer** (`langchain`): Concrete implementations and high-level public utilities
- **Integration layer** (`partners/`): Third-party service integrations. Note that this monorepo is not exhaustive of all LangChain integrations; some are maintained in separate repos, such as `langchain-ai/langchain-google` and `langchain-ai/langchain-aws`. Usually these repos are cloned at the same level as this monorepo, so if needed, you can refer to their code directly by navigating to `../langchain-google/` from this monorepo.
- **Testing layer** (`standard-tests/`): Standardized integration tests for partner integrations

### Development tools & commands**

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `make` – Task runner for common development commands. Feel free to look at the `Makefile` for available commands and usage patterns.
- `ruff` – Fast Python linter and formatter
- `mypy` – Static type checking
- `pytest` – Testing framework

This monorepo uses `uv` for dependency management. Local development uses editable installs: `[tool.uv.sources]`

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

#### Key config files

- pyproject.toml: Main workspace configuration with dependency groups
- uv.lock: Locked dependencies for reproducible builds
- Makefile: Development tasks

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Refer to .github/workflows/pr_lint for allowed types and scopes. Note that all commit/PR titles should be in lowercase with the exception of proper nouns/named entities. All PR titles should include a scope with no exceptions. For example:

```txt
feat(langchain): add new chat completion feature
fix(core): resolve type hinting issue in vector store
chore(anthropic): update infrastructure dependencies
```

Note how `feat(langchain)` includes a scope even though it is the main package and name of the repo.

#### Pull request guidelines

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
        List of users that are not in the known_users set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.

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

## Additional resources

- **Documentation:** https://docs.langchain.com/oss/python/langchain/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [`.github/CONTRIBUTING.md`](https://docs.langchain.com/oss/python/contributing/overview)
