# langchain-typesafe

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-typesafe?label=%20)](https://pypi.org/project/langchain-typesafe/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-typesafe)](https://opensource.org/licenses/MIT)

Three LangChain agent middleware that make decisions with [TypeSafe](https://docs.typesafe.ai/) classifiers.

```bash
uv add langchain-typesafe
export TYPESAFE_API_KEY=...
```

> [!WARNING]
> All three middleware are experimental. APIs may change without notice.

Requests go through the official [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python). This package adds no wrapper around it and re-exports none of its types — configure a middleware by passing a `typesafe_sdk` client, and catch its exception types directly.

| Middleware | Decision | On classification failure |
|---|---|---|
| `ModelRouterMiddleware` | Which model handles the run | Falls back to `default_route` |
| `SkillsMiddleware` | Which skills apply to the request | Adds no skills |
| `AutoModeMiddleware` | Whether a tool call is too risky to run | Propagates; the tool is **not** executed |

Runnable examples for each live in [`examples/`](examples/); `make examples` runs all three against the live API.

## Model routing

```python
from langchain.agents import create_agent
from langchain_typesafe import ModelChoice, ModelRouterMiddleware

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(model=fast_model, criteria="Trivial lookups and one-line edits."),
        "powerful": ModelChoice(model=powerful_model, criteria="Multi-step reasoning."),
    },
    instructions="Choose the least costly model that can do the task well.",
    default_route="powerful",
)
agent = create_agent(fast_model, middleware=[router])
```

The latest human message is classified once before the run, and the selected model is used for every model call in it.

## Skills

```python
from pathlib import Path
from langchain_typesafe import SkillsMiddleware

library = Path("skills")
middleware = SkillsMiddleware(
    skills=[library / "code-review" / "SKILL.md"],
    skills_root=library,
)
```

Each skill is scored independently, so any number can apply. Only the selected skills' instructions are prepended to the model request, so the roster never enters the prompt. Skills load from Agent Skills-compatible `SKILL.md` files, `Skill` objects, or `SKILL.md` strings.

> [!IMPORTANT]
> Skill content becomes a system message and can direct agent behavior, so skill sources are trusted configuration — never load a `SKILL.md` a user can write. `skills_root` confines path sources to one directory, enforced after symlink resolution.

## Tool-risk gating

```python
from langchain_typesafe import AutoModeMiddleware

agent = create_agent(
    model,
    tools=[read_file, delete_file],
    middleware=[AutoModeMiddleware(tools=["delete_file"], risk_threshold=0.2)],
)
```

Only the listed tools are classified. The classifier sees user messages, the tool-call ID and name, redacted arguments, and the tool description — tool output and assistant messages are excluded so content the agent fetched cannot authorize its own execution.

> [!IMPORTANT]
> Tool arguments are sent to TypeSafe. Redaction covers credential-like argument *keys* (`api_key`, `token`, `secret`, and similar), so a secret inside an unrelated free-form value is not caught. This middleware blocks; it does not ask a human — pair it with a human-in-the-loop middleware if approval is what you want.

## Configuration

Middleware create a default SDK client, which resolves `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, and `TYPESAFE_DEFAULT_MODEL` from the environment. Pass `client` and `async_client` to control the model, timeout, retry policy, base URL, or transport:

```python
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient

SkillsMiddleware(
    skills=[...],
    client=TypeSafeClient(model="jev-1.13", retry=RetryPolicy(max_retries=5)),
    async_client=AsyncTypeSafeClient(model="jev-1.13"),
)
```

Classification failures are logged with the error type, HTTP status, and TypeSafe request ID, and deliberately not with the provider's message — the SDK embeds a truncated response body in its error strings, and a validation error can echo back part of the classified state. Note that the SDK's own `typesafe_sdk` logger redacts secret headers but not request and response bodies.

## Documentation

See the [TypeSafe documentation](https://docs.typesafe.ai/) for question and model semantics, and [reference.langchain.com](https://reference.langchain.com/python/integrations/langchain_typesafe/) for the API reference.

## Contributing

For contribution instructions, see the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
