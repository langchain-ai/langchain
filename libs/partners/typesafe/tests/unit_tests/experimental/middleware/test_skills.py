"""Tests for `SkillsMiddleware`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from langchain_typesafe import NoulAnswer, TypeSafeClassifier
from langchain_typesafe.experimental.middleware import Skill, SkillsMiddleware
from langchain_typesafe.experimental.middleware import __all__ as middleware_all
from langchain_typesafe.types import ClassificationResponse

SKILL_MARKDOWN = """---
name: code-review
description: Review code for correctness and maintainability.
license: MIT
compatibility: Requires git.
metadata:
  author: langchain
allowed-tools: Read Bash(git:*)
---
# Code review

Inspect the diff before commenting.
"""

OTHER_SKILL_MARKDOWN = """---
name: docs-writer
description: Write clear technical documentation.
---
# Documentation

Prefer runnable examples.
"""


def _response(scores: dict[str, float]) -> ClassificationResponse:
    return ClassificationResponse(
        model="jev-latest",
        answers={
            f"skill::{name}": NoulAnswer(type="noul", noul=score)
            for name, score in scores.items()
        },
    )


def _middleware(
    selected: tuple[str, ...] = ("code-review",),
    *,
    scores: dict[str, float] | None = None,
    relevance_threshold: float = 0.3,
) -> tuple[SkillsMiddleware, MagicMock, MagicMock]:
    response = _response(
        scores
        or {
            name: 0.9 if name in selected else 0.1
            for name in ("code-review", "docs-writer")
        }
    )
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.return_value = response
    classifier.ainvoke = AsyncMock(return_value=response)
    with patch(
        "langchain_typesafe.experimental.middleware.skills.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware = SkillsMiddleware(
            skills=[SKILL_MARKDOWN, OTHER_SKILL_MARKDOWN],
            relevance_threshold=relevance_threshold,
        )
    return middleware, classifier, classifier_class


def _request(state: dict[str, Any]) -> ModelRequest[Any]:
    return ModelRequest(
        model=cast("BaseChatModel", MagicMock()),
        messages=state.get("messages", []),
        state=cast("Any", state),
    )


def test_skill_accepts_agent_skills_fields() -> None:
    """Represent every `SKILL.md` frontmatter field in the public data type."""
    skill = Skill.from_markdown(SKILL_MARKDOWN)

    assert skill.name == "code-review"
    assert skill.description == "Review code for correctness and maintainability."
    assert skill.license == "MIT"
    assert skill.compatibility == "Requires git."
    assert skill.metadata == {"author": "langchain"}
    assert skill.allowed_tools == "Read Bash(git:*)"
    assert skill.content.startswith("# Code review")


def test_skill_sources_accept_objects_paths_and_markdown(tmp_path: Path) -> None:
    """Normalize all three supported skill source forms."""
    skill_dir = tmp_path / "docs-writer"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(OTHER_SKILL_MARKDOWN, encoding="utf-8")
    direct = Skill(
        name="code-review",
        description="Review code.",
        content="Review the entire diff.",
    )

    with patch("langchain_typesafe.experimental.middleware.skills.TypeSafeClassifier"):
        middleware = SkillsMiddleware(skills=[direct, skill_path])

    assert middleware.skills == {
        "code-review": direct,
        "docs-writer": Skill.from_markdown(OTHER_SKILL_MARKDOWN),
    }


@pytest.mark.parametrize(
    ("markdown", "match"),
    [
        ("# Missing frontmatter", "must begin"),
        ("---\nname: Bad_Name\ndescription: invalid\n---\n", "Skill names"),
        ("---\nname: valid\n---\n", "description"),
        ("---\n- not\n- a mapping\n---\n", "YAML mapping"),
    ],
)
def test_invalid_skill_markdown_is_rejected(markdown: str, match: str) -> None:
    """Reject malformed documents and invalid Agent Skills metadata."""
    with pytest.raises((TypeError, ValueError, ValidationError), match=match):
        Skill.from_markdown(markdown)


def test_path_name_must_match_parent_directory(tmp_path: Path) -> None:
    """Enforce the directory-name rule when a path provides that context."""
    skill_dir = tmp_path / "wrong-name"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(SKILL_MARKDOWN, encoding="utf-8")

    with pytest.raises(ValueError, match="must match its parent directory"):
        SkillsMiddleware(skills=[skill_path])


def test_empty_and_duplicate_rosters_are_rejected() -> None:
    """Require a non-empty roster with unique names and a valid gate."""
    with pytest.raises(ValueError, match="At least one skill"):
        SkillsMiddleware(skills=[])
    with pytest.raises(ValueError, match="duplicates"):
        SkillsMiddleware(skills=[SKILL_MARKDOWN, SKILL_MARKDOWN])
    with pytest.raises(ValueError, match="relevance_threshold"):
        SkillsMiddleware(skills=[SKILL_MARKDOWN], relevance_threshold=1.1)


def test_middleware_constructs_classifier_from_skills() -> None:
    """Construct one independent TypeSafe relevance question per skill."""
    middleware, classifier, classifier_class = _middleware()

    classifier_class.assert_called_once()
    questions = classifier_class.call_args.kwargs["questions"]
    assert set(questions) == {"skill::code-review", "skill::docs-writer"}
    assert "code-review" in questions["skill::code-review"].instructions
    assert (
        "Review code for correctness and maintainability."
        in questions["skill::code-review"].instructions
    )
    assert "Inspect the diff" not in repr(questions)
    assert middleware.classifier is classifier


def test_sync_selection_uses_latest_human_message_and_injects_skill() -> None:
    """Classify once and prepend only the selected skill to request messages."""
    middleware, classifier, _ = _middleware()
    old_message = HumanMessage("Earlier request")
    latest_message = HumanMessage("Review this pull request")
    state: dict[str, Any] = {
        "messages": [old_message, AIMessage("Ready"), latest_message]
    }

    state.update(middleware.before_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    assert state["selected_skills"] == ["code-review"]
    classifier.invoke.assert_called_once()
    assert classifier.invoke.call_args.args[0] is latest_message
    assert classifier.invoke.call_args.kwargs["config"]["metadata"] == {
        "lc_source": "typesafe_skills"
    }
    assert isinstance(seen[0].messages[0], SystemMessage)
    assert seen[0].messages[0].text == (
        "<skill name='code-review'>\n"
        "# Code review\n\nInspect the diff before commenting.\n\n</skill>"
    )
    assert seen[0].messages[1:] == request.messages
    assert request.messages[0] is old_message


def test_multiple_relevant_skills_are_injected_in_roster_order() -> None:
    """Inject every independently selected skill rather than forcing one winner."""
    middleware, _, _ = _middleware(("code-review", "docs-writer"))
    state: dict[str, Any] = {
        "messages": [HumanMessage("Review this code and document the API")]
    }
    state.update(middleware.before_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    assert state["selected_skills"] == ["code-review", "docs-writer"]
    assert [message.text.splitlines()[0] for message in seen[0].messages[:2]] == [
        "<skill name='code-review'>",
        "<skill name='docs-writer'>",
    ]
    assert seen[0].messages[2:] == request.messages


@pytest.mark.asyncio
async def test_async_selection_and_injection() -> None:
    """Use asynchronous classification and request handling."""
    middleware, classifier, _ = _middleware(("docs-writer",))
    state: dict[str, Any] = {"messages": [HumanMessage("Write the API guide")]}

    state.update(await middleware.abefore_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    async def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return MagicMock()

    await middleware.awrap_model_call(request, handler)

    assert state["selected_skills"] == ["docs-writer"]
    classifier.ainvoke.assert_awaited_once()
    assert seen[0].messages[0].text.startswith("<skill name='docs-writer'>")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_classifier_failure_does_not_select_skill(*, asynchronous: bool) -> None:
    """Fail open without adding skill instructions when classification fails."""
    middleware, classifier, _ = _middleware()
    classifier.invoke.side_effect = RuntimeError("unavailable")
    classifier.ainvoke.side_effect = RuntimeError("unavailable")
    state = {"messages": [HumanMessage("Review this change")]}

    if asynchronous:
        update = await middleware.abefore_agent(cast("Any", state), MagicMock())
    else:
        update = middleware.before_agent(cast("Any", state), MagicMock())

    assert update == {"selected_skills": []}


def test_no_relevant_skills_does_not_modify_messages() -> None:
    """Leave the request untouched when every relevance score is low."""
    middleware, _, _ = _middleware(())
    state: dict[str, Any] = {"messages": [HumanMessage("What time is it?")]}
    state.update(middleware.before_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    assert state["selected_skills"] == []
    assert seen[0] is request


def test_relevance_threshold_is_applied_per_skill() -> None:
    """Select each skill independently at the inclusive threshold boundary."""
    middleware, _, _ = _middleware(scores={"code-review": 0.3, "docs-writer": 0.29})
    state = {"messages": [HumanMessage("Review and document this change")]}

    update = middleware.before_agent(cast("Any", state), MagicMock())

    assert update == {"selected_skills": ["code-review"]}


def test_missing_human_message_skips_classification() -> None:
    """Avoid a classifier call when there is no human request."""
    middleware, classifier, _ = _middleware()

    update = middleware.before_agent(
        cast("Any", {"messages": [AIMessage("No request yet")]}),
        MagicMock(),
    )

    assert update == {"selected_skills": []}
    classifier.invoke.assert_not_called()


def test_create_agent_keeps_selection_private() -> None:
    """Compose the middleware without exposing its internal selection state."""
    middleware, classifier, _ = _middleware()
    model = GenericFakeChatModel(messages=iter([AIMessage("reviewed")]))
    agent = create_agent(model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Review this change")]})

    assert result["messages"][-1].text == "reviewed"
    assert "selected_skills" not in result
    classifier.invoke.assert_called_once()


def test_experimental_public_interface() -> None:
    """Expose skills from the experimental middleware namespace."""
    assert middleware_all == [
        "Skill",
        "SkillSource",
        "SkillsMiddleware",
        "TsChoiceToolSelectorMiddleware",
        "TsToolSelectorMiddleware",
    ]
