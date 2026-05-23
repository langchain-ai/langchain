"""Pytest configuration for langchain real-model middleware evals.

These tests run actual model invocations against `create_agent` middleware to
verify behavioral properties (e.g., the final substantive answer lands in the
loop-terminating message). They are intentionally separate from the unit-test
suite because they require API keys and incur cost.

Usage:
    cd libs/langchain_v1
    uv run --group test pytest tests/evals -v --model claude-sonnet-4-6

Markers:
    @pytest.mark.eval_category("name") — group tests for filtering / reporting
    @pytest.mark.eval_tier("baseline" | "hillclimb") — gating level
    @pytest.mark.langsmith — opt-in LangSmith experiment integration

Shape mirrors `libs/evals` in `langchain-ai/deepagents` so anyone moving
between the two repos sees the same API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register eval-specific CLI options."""
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help=(
            "Model identifier to run evals against (required). "
            "Examples: --model claude-sonnet-4-6, --model openai:gpt-5, "
            "--model claude-haiku-4-5"
        ),
    )
    parser.addoption(
        "--openai-reasoning-effort",
        action="store",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Forwarded as `reasoning_effort=` to ChatOpenAI when the selected "
            "model is OpenAI. Ignored for other providers. Use `minimal` to "
            "skip the reasoning preamble for faster/cheaper runs."
        ),
    )
    parser.addoption(
        "--eval-tier",
        action="append",
        default=[],
        help=(
            "Run only evals tagged with this `eval_tier(...)` value. Repeatable. "
            "Example: --eval-tier baseline --eval-tier hillclimb. "
            'Standard `-m` filtering does not work because `eval_tier("name")` '
            "is not a pytest marker boolean expression."
        ),
    )
    parser.addoption(
        "--eval-category",
        action="append",
        default=[],
        help=(
            "Run only evals tagged with this `eval_category(...)` value. "
            "Repeatable. Example: --eval-category middleware/todo"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by eval tests."""
    config.addinivalue_line(
        "markers",
        "eval_category(name): tag an eval with a category for filtering/reporting",
    )
    config.addinivalue_line(
        "markers",
        (
            "eval_tier(name): tag an eval as 'baseline' (regression gate) "
            "or 'hillclimb' (progress tracking)"
        ),
    )
    config.addinivalue_line(
        "markers",
        "langsmith: opt into LangSmith experiment integration for this eval",
    )


def _filter_by_marker_arg(
    items: list[pytest.Item], *, include: list[str], marker_name: str
) -> None:
    """Keep only items whose `marker_name(...)` first arg is in ``include``.

    An empty include list means "include everything". When a test is decorated
    with `@pytest.mark.<marker_name>("value")`, this helper keeps it iff
    ``value`` is in ``include``. Used to make `--eval-tier baseline` and
    `--eval-category middleware/todo` actually filter the test set, since
    pytest's `-m` flag does not understand `<marker>("arg")` syntax.
    """
    if not include:
        return
    kept: list[pytest.Item] = []
    for item in items:
        marker = item.get_closest_marker(marker_name)
        if marker and marker.args and marker.args[0] in include:
            kept.append(item)
    items[:] = kept


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail fast on missing --model, then filter by --eval-tier / --eval-category."""
    if not items:
        return
    if not config.getoption("--model"):
        pytest.exit(
            "langchain evals require an explicit --model argument. "
            "Example: pytest tests/evals --model claude-sonnet-4-6",
            returncode=2,
        )

    _filter_by_marker_arg(
        items,
        include=list(config.getoption("--eval-tier") or []),
        marker_name="eval_tier",
    )
    _filter_by_marker_arg(
        items,
        include=list(config.getoption("--eval-category") or []),
        marker_name="eval_category",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize tests that take a ``model_name`` fixture with the CLI value."""
    if "model_name" in metafunc.fixturenames:
        metafunc.parametrize("model_name", [metafunc.config.getoption("--model")])


@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str:
    """Selected model id (parameterized from ``--model``)."""
    return request.config.getoption("--model")


@pytest.fixture
def model(model_name: str, request: pytest.FixtureRequest) -> BaseChatModel:
    """Construct a `BaseChatModel` for the selected model id.

    Uses `langchain.chat_models.init_chat_model` so any provider-prefixed id
    (e.g., `openai:gpt-5`, `anthropic:claude-sonnet-4-6`) or bare Anthropic id
    (`claude-sonnet-4-6`) is handled uniformly. Per-provider API keys must be
    set in the environment.
    """
    kwargs: dict[str, Any] = {}
    effort = request.config.getoption("--openai-reasoning-effort")
    if effort and (model_name.startswith(("openai:", "gpt-")) or "openai" in model_name.lower()):
        kwargs["reasoning_effort"] = effort

    # Anthropic short ids work with init_chat_model when model_provider is set.
    if model_name.startswith("claude-"):
        return init_chat_model(model=model_name, model_provider="anthropic", **kwargs)
    return init_chat_model(model=model_name, **kwargs)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Emit a one-line summary of the eval environment at session start."""
    if not session.config.getoption("--model"):
        return
    model_arg = session.config.getoption("--model")
    tracing = os.environ.get("LANGSMITH_TRACING") or "(not set)"
    print(  # noqa: T201 - eval banner is intended user-facing output
        f"\nlangchain evals: model={model_arg} LANGSMITH_TRACING={tracing}",
        flush=True,
    )
