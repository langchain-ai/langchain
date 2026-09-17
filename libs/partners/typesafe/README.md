# langchain-typesafe

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-typesafe?label=%20)](https://pypi.org/project/langchain-typesafe/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-typesafe)](https://opensource.org/licenses/MIT)

LangChain agent middleware that make decisions with [TypeSafe](https://docs.typesafe.ai/) classifiers.

## Installation

```bash
uv add langchain-typesafe
```

Set the `TYPESAFE_API_KEY` environment variable before running an agent. `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL` are also honored, and explicit constructor arguments take precedence over both.

> [!WARNING]
> Every middleware in this package is experimental. APIs may change without notice.

## What this package is

TypeSafe ships a first-party Python SDK. This package does not wrap it, re-export its types, or add a `Runnable` around it — questions, answers, retries, and error semantics are the SDK's. What it adds is the part the SDK cannot know about: where in an agent's loop a classification belongs, and what to do with the verdict.

Import question types, `RetryPolicy`, and the exception types from `typesafe_sdk` directly. Every error raised here subclasses both the `typesafe_sdk` exception and the matching `langchain_core.exceptions.ModelError`, so either import catches it.

## Model routing

`ModelRouterMiddleware` classifies the latest human message once before a run and uses the selected model for every model call in that run. Classification failures and unrecognized routes fall back to `default_route`, so routing never stops an agent from running.

```python
from langchain.agents import create_agent
from langchain_typesafe import ModelChoice, ModelRouterMiddleware

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(model=fast_model, criteria="Simple, well-scoped tasks."),
        "powerful": ModelChoice(
            model=powerful_model,
            criteria="Complex tasks requiring deeper reasoning.",
        ),
    },
    instructions="Choose the least costly model suited to the task.",
    default_route="powerful",
)
agent = create_agent(fast_model, middleware=[router])
```

## Skills

`SkillsMiddleware` scores each skill independently against the latest request and prepends only the selected skills' instructions to the model request. The full roster never enters the prompt, so adding skills does not grow the context of every call.

```python
from pathlib import Path

from langchain.agents import create_agent
from langchain_typesafe import SkillsMiddleware

library = Path("skills")
agent = create_agent(
    model,
    middleware=[
        SkillsMiddleware(
            skills=[library / "code-review" / "SKILL.md"],
            skills_root=library,
        )
    ],
)
```

Skills are read from Agent Skills-compatible `SKILL.md` files, or passed as `Skill` objects or `SKILL.md` strings.

> [!IMPORTANT]
> Skill content is injected as a system message and can direct agent behavior, so skill sources are trusted configuration. Never load a `SKILL.md` that a user can write. Passing `skills_root` confines path sources to one directory, so a symlink or `..` component cannot pull in a file from elsewhere.

## Tool-risk gating

`AutoModeMiddleware` asks TypeSafe for the probability that a tool call is risky or insufficiently authorized, and blocks calls at or above the threshold by returning an error `ToolMessage` instead of running the tool. Only the tools you list are classified.

```python
from langchain.agents import create_agent
from langchain_typesafe import AutoModeMiddleware

agent = create_agent(
    model,
    tools=[read_file, delete_file],
    middleware=[AutoModeMiddleware(tools=["delete_file"], risk_threshold=0.2)],
)
```

The classifier sees only user messages, the tool-call ID and name, redacted arguments, and the tool description. Tool output and assistant messages are excluded so content the agent fetched cannot authorize its own execution. Classification failures propagate and the tool is not executed, which makes this fail-closed — unlike routing and skills, which fail open.

> [!IMPORTANT]
> Tool-call arguments are sent to TypeSafe. Redaction covers credential-like argument *keys* (`api_key`, `token`, `secret`, and similar), so a secret carried inside an unrelated free-form value is not caught. Do not enroll tools whose arguments embed credentials in prose.
>
> This middleware blocks risky calls; it does not ask a human. Pair it with a human-in-the-loop middleware when approval is what you want.

## Failure behavior

| Middleware | On classification failure |
|---|---|
| `ModelRouterMiddleware` | Falls back to `default_route`, logs a warning |
| `SkillsMiddleware` | Adds no skills, logs a warning |
| `AutoModeMiddleware` | Propagates the error; the tool is not executed |

Failures are logged with the error type, HTTP status, and TypeSafe request ID, and deliberately not with the provider's message — the SDK embeds a truncated response body in its error strings, and a validation error can echo back part of the classified state, which routinely contains user content.

Note that the TypeSafe SDK logs to its own `typesafe_sdk` logger, and per its documentation redacts secret headers but not request and response bodies. Configure that logger accordingly when classifying sensitive content.

## Documentation

See the [TypeSafe documentation](https://docs.typesafe.ai/) for question and model semantics. LangChain API reference documentation is available at [reference.langchain.com](https://reference.langchain.com/python/integrations/langchain_typesafe/).

## Contributing

For contribution instructions, see the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
