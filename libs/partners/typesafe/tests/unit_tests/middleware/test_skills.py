"""Unit tests for `SkillsMiddleware`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_typesafe import Skill, SkillsMiddleware
from tests.unit_tests.conftest import (
    RUNTIME,
    RecordingTransport,
    answers_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit_tests.conftest import Handler

REVIEW_MD = """---
name: code-review
description: Review a diff for correctness and style problems.
---

Read the diff, then report findings most severe first.
"""

RELEASE_MD = """---
name: release-notes
description: Draft release notes from merged pull requests.
---

Summarize user-visible changes.
"""


def _skill_answers(**scores: float) -> dict[str, Any]:
    return answers_response(
        {
            f"skill::{name}": {"type": "noul", "noul": score}
            for name, score in scores.items()
        }
    )


def _middleware(
    clients: Callable[[Handler], dict[str, Any]],
    transport: RecordingTransport,
    **kwargs: Any,
) -> SkillsMiddleware:
    kwargs.setdefault("skills", [REVIEW_MD, RELEASE_MD])
    return SkillsMiddleware(**kwargs, **clients(transport))


def test_skill_parses_frontmatter_and_body() -> None:
    """A `SKILL.md` document yields its metadata and Markdown body."""
    skill = Skill.from_markdown(REVIEW_MD)

    assert skill.name == "code-review"
    assert skill.description.startswith("Review a diff")
    assert "most severe first" in skill.content


@pytest.mark.parametrize(
    ("markdown", "match"),
    [
        ("no frontmatter", "must begin with YAML frontmatter"),
        ("---\nname: x\n", "missing its closing"),
        ("---\nname: [unclosed\n---\n", "invalid YAML frontmatter"),
    ],
)
def test_malformed_skill_documents_are_rejected(markdown: str, match: str) -> None:
    """Malformed documents fail with an explanation."""
    with pytest.raises(ValueError, match=match):
        Skill.from_markdown(markdown)


def test_frontmatter_must_be_a_mapping() -> None:
    """Frontmatter that is not a mapping is rejected."""
    with pytest.raises(TypeError, match="must be a YAML mapping"):
        Skill.from_markdown("---\n- a\n- b\n---\n")


@pytest.mark.parametrize(
    "name", ["Code-Review", "-leading", "trailing-", "under_score"]
)
def test_invalid_skill_names_are_rejected(name: str) -> None:
    """Skill names follow the Agent Skills format."""
    with pytest.raises(ValueError, match="lowercase letters"):
        Skill.from_markdown(f"---\nname: {name}\ndescription: d\n---\nbody\n")


def test_empty_roster_is_rejected() -> None:
    """At least one skill is required."""
    with pytest.raises(ValueError, match="At least one skill"):
        SkillsMiddleware(skills=[])


def test_duplicate_skill_names_are_rejected() -> None:
    """Two skills cannot share a name, since names key the questions."""
    with pytest.raises(ValueError, match="duplicates"):
        SkillsMiddleware(skills=[REVIEW_MD, REVIEW_MD])


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_out_of_range_threshold_is_rejected(threshold: float) -> None:
    """The relevance threshold is a probability."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        SkillsMiddleware(skills=[REVIEW_MD], relevance_threshold=threshold)


def test_each_skill_becomes_its_own_question(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Skills are scored independently so any number can be selected."""
    transport = RecordingTransport(_skill_answers(**{"code-review": 0.9}))
    middleware = _middleware(clients, transport)

    middleware.before_agent({"messages": [HumanMessage("Review my diff.")]}, RUNTIME)

    assert set(transport.questions) == {"skill::code-review", "skill::release-notes"}
    assert all(q["type"] == "noul" for q in transport.questions.values())


def test_skills_above_the_threshold_are_selected(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Relevance at or above the threshold selects a skill."""
    answers = _skill_answers(**{"code-review": 0.9, "release-notes": 0.05})
    middleware = _middleware(clients, RecordingTransport(answers))

    result = middleware.before_agent(
        {"messages": [HumanMessage("Review my diff.")]},
        RUNTIME,
    )

    assert result == {"selected_skills": ["code-review"]}


def test_multiple_skills_can_be_selected(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Independent scoring means several skills can apply at once."""
    answers = _skill_answers(**{"code-review": 0.9, "release-notes": 0.7})
    middleware = _middleware(clients, RecordingTransport(answers))

    result = middleware.before_agent({"messages": [HumanMessage("Ship it.")]}, RUNTIME)

    assert result == {"selected_skills": ["code-review", "release-notes"]}


def test_threshold_boundary_is_inclusive(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """A score exactly at the threshold selects the skill."""
    answers = _skill_answers(**{"code-review": 0.3, "release-notes": 0.29})
    middleware = _middleware(clients, RecordingTransport(answers))

    result = middleware.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME)

    assert result == {"selected_skills": ["code-review"]}


def test_no_human_message_selects_nothing(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Without a human turn there is nothing to match skills against."""
    transport = RecordingTransport(_skill_answers(**{"code-review": 0.9}))
    middleware = _middleware(clients, transport)

    result = middleware.before_agent({"messages": [AIMessage("Hello.")]}, RUNTIME)

    assert result == {"selected_skills": []}
    assert transport.requests == []


def test_classification_failure_selects_nothing(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Skills fail open so a TypeSafe outage cannot stop the agent."""
    middleware = _middleware(clients, RecordingTransport(429))

    result = middleware.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME)

    assert result == {"selected_skills": []}


def test_failure_logs_no_provider_message(
    clients: Callable[[Handler], dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure logs carry error metadata, not the provider's response body."""
    middleware = _middleware(clients, RecordingTransport(429))

    with caplog.at_level("WARNING"):
        middleware.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME)

    assert "TypeSafeRateLimitError" in caplog.text
    assert "no skills were added" in caplog.text


async def test_async_selection(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """The async hook behaves like the sync one."""
    answers = _skill_answers(**{"code-review": 0.9, "release-notes": 0.0})
    middleware = _middleware(clients, RecordingTransport(answers))

    result = await middleware.abefore_agent(
        {"messages": [HumanMessage("Review my diff.")]},
        RUNTIME,
    )

    assert result == {"selected_skills": ["code-review"]}


async def test_async_failure_selects_nothing(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """The async hook also fails open."""
    middleware = _middleware(clients, RecordingTransport(429))

    result = await middleware.abefore_agent(
        {"messages": [HumanMessage("Hi.")]},
        RUNTIME,
    )

    assert result == {"selected_skills": []}


class _Request:
    """Minimal `ModelRequest` stand-in recording overridden messages."""

    def __init__(self, state: dict[str, Any], messages: list[Any]) -> None:
        self.state = state
        self.messages = messages

    def override(self, **kwargs: Any) -> _Request:
        self.messages = kwargs["messages"]
        return self


def test_selected_skill_content_is_prepended(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Only the selected skill's instructions enter the model request."""
    transport = RecordingTransport(_skill_answers(**{"code-review": 0.9}))
    middleware = _middleware(clients, transport)
    original = HumanMessage("Review my diff.")

    request = middleware._add_selected_skills(
        _Request({"selected_skills": ["code-review"]}, [original]),  # type: ignore[arg-type]
    )

    assert isinstance(request.messages[0], SystemMessage)
    assert "most severe first" in request.messages[0].content
    assert "release-notes" not in request.messages[0].content
    assert request.messages[1] is original


def test_no_selection_leaves_the_request_untouched(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """An empty selection adds no system message."""
    transport = RecordingTransport(_skill_answers(**{"code-review": 0.0}))
    middleware = _middleware(clients, transport)
    original = HumanMessage("Hi.")

    request = middleware._add_selected_skills(
        _Request({"selected_skills": []}, [original]),  # type: ignore[arg-type]
    )

    assert request.messages == [original]


def test_unknown_selected_skill_is_ignored(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """State naming a skill that is not configured cannot inject content."""
    transport = RecordingTransport(_skill_answers(**{"code-review": 0.0}))
    middleware = _middleware(clients, transport)
    original = HumanMessage("Hi.")

    request = middleware._add_selected_skills(
        _Request({"selected_skills": ["not-configured", 7]}, [original]),  # type: ignore[arg-type]
    )

    assert request.messages == [original]


def _write_skill(root: Path, name: str, markdown: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def test_skill_loads_from_a_path(tmp_path: Path) -> None:
    """A `SKILL.md` path is read from disk."""
    path = _write_skill(tmp_path, "code-review", REVIEW_MD)

    middleware = SkillsMiddleware(skills=[path])

    assert set(middleware.skills) == {"code-review"}


def test_skill_name_must_match_its_directory(tmp_path: Path) -> None:
    """The Agent Skills layout requires the directory to match the name."""
    path = _write_skill(tmp_path, "wrong-directory", REVIEW_MD)

    with pytest.raises(ValueError, match="must match its parent directory"):
        SkillsMiddleware(skills=[path])


def test_non_skill_filename_is_rejected(tmp_path: Path) -> None:
    """Only `SKILL.md` files are loaded."""
    path = tmp_path / "code-review" / "OTHER.md"
    path.parent.mkdir(parents=True)
    path.write_text(REVIEW_MD, encoding="utf-8")

    with pytest.raises(ValueError, match=r"must point to a SKILL\.md file"):
        SkillsMiddleware(skills=[path])


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    """A path that does not exist fails with a clear error."""
    with pytest.raises(ValueError, match="Unable to resolve"):
        SkillsMiddleware(skills=[tmp_path / "gone" / "SKILL.md"])


def test_path_outside_the_root_is_rejected(tmp_path: Path) -> None:
    """`skills_root` confines path sources to one directory."""
    library = tmp_path / "library"
    library.mkdir()
    outside = _write_skill(tmp_path / "elsewhere", "code-review", REVIEW_MD)

    with pytest.raises(ValueError, match="resolves outside the configured skills root"):
        SkillsMiddleware(skills=[outside], skills_root=library)


def test_symlink_escaping_the_root_is_rejected(tmp_path: Path) -> None:
    """Containment is enforced after resolving symlinks."""
    library = tmp_path / "library"
    library.mkdir()
    real = _write_skill(tmp_path / "secrets", "code-review", REVIEW_MD)
    link_dir = library / "code-review"
    link_dir.mkdir()
    link = link_dir / "SKILL.md"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="resolves outside the configured skills root"):
        SkillsMiddleware(skills=[link], skills_root=library)


def test_path_inside_the_root_is_accepted(tmp_path: Path) -> None:
    """A path within the root loads normally."""
    library = tmp_path / "library"
    path = _write_skill(library, "code-review", REVIEW_MD)

    middleware = SkillsMiddleware(skills=[path], skills_root=library)

    assert set(middleware.skills) == {"code-review"}


def test_oversized_skill_file_is_rejected(tmp_path: Path) -> None:
    """Files past the size limit are rejected rather than truncated."""
    path = _write_skill(tmp_path, "code-review", REVIEW_MD)
    path.write_text("---\nname: code-review\n" + "x" * (10 * 1024 * 1024))

    with pytest.raises(ValueError, match="exceeds the 10 MiB size limit"):
        SkillsMiddleware(skills=[path])


def test_invalid_utf8_skill_file_is_rejected(tmp_path: Path) -> None:
    """A file that is not UTF-8 fails with a clear error."""
    path = _write_skill(tmp_path, "code-review", REVIEW_MD)
    path.write_bytes(b"---\nname: code-review\n\xff\xfe\n---\n")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        SkillsMiddleware(skills=[path])


def test_skill_objects_are_accepted() -> None:
    """A pre-built `Skill` needs no parsing."""
    skill = Skill(name="code-review", description="Review diffs.", content="body")

    middleware = SkillsMiddleware(skills=[skill])

    assert middleware.skills["code-review"] is skill
