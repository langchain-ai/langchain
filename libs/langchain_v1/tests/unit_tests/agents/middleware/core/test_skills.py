from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages.content import TextContentBlock

from langchain.agents import create_agent
from langchain.agents.middleware import SkillsMiddleware
from langchain.agents.middleware._utils import append_to_system_message
from langchain.agents.middleware.skill import (
    MAX_SKILL_LOAD_WARNING_LENGTH,
    SkillMetadata,
    SkillSource,
    _derive_source_label,
    _format_skill_annotations,
    _parse_allowed_tools,
    _source_path,
    _truncate_skill_load_warning,
    _validate_metadata,
    _validate_skill_name,
    _validate_tuple_source,
    load_skill_content,
)
from langchain.agents.protocol import FileDownloadResponse
from tests.unit_tests.agents.model import FakeToolCallingModel


class _SkillsBackend:
    def ls(self, path: str) -> list[dict[str, object]]:
        assert path == "/skills/"
        return [
            {
                "path": "/skills/web-research/",
                "is_dir": True,
            }
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        assert paths == ["/skills/web-research/SKILL.md"]
        return [
            FileDownloadResponse(
                path="/skills/web-research/SKILL.md",
                content=(
                    b"---\nname: web-research\ndescription: Research the web\n---\n# Web Research\n"
                ),
                error=None,
            )
        ]

    async def als(self, path: str) -> list[dict[str, object]]:
        return self.ls(path)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self.download_files(paths)


class _FakeRequest:
    def __init__(self, *, state: dict[str, object], system_message: SystemMessage | None) -> None:
        self.state = state
        self.system_message = system_message

    def override(self, **kwargs: object) -> "_FakeRequest":
        return _FakeRequest(
            state=self.state,
            system_message=cast(
                "SystemMessage | None",
                kwargs.get("system_message", self.system_message),
            ),
        )


def test_skills_middleware_is_publicly_exported() -> None:
    assert SkillsMiddleware.__name__ == "SkillsMiddleware"


def test_before_agent_loads_metadata_from_backend() -> None:
    middleware = SkillsMiddleware(backend=_SkillsBackend(), sources=["/skills/"])

    result = middleware.before_agent(state={}, runtime=SimpleNamespace(), config={})

    assert result is not None
    assert result["skills_metadata"] == [
        {
            "name": "web-research",
            "description": "Research the web",
            "path": "/skills/web-research/SKILL.md",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


def test_modify_request_appends_skill_list_to_system_message() -> None:
    middleware = SkillsMiddleware(backend=_SkillsBackend(), sources=["/skills/"])
    original_request = _FakeRequest(
        state={
            "skills_metadata": [
                {
                    "name": "web-research",
                    "description": "Research the web",
                    "path": "/skills/web-research/SKILL.md",
                    "metadata": {},
                    "license": None,
                    "compatibility": None,
                    "allowed_tools": [],
                }
            ],
            "skills_load_errors": [],
        },
        system_message=None,
    )

    modified_request = middleware.modify_request(original_request)

    assert modified_request.system_message is not None
    assert isinstance(modified_request.system_message, SystemMessage)

    content_blocks = modified_request.system_message.content_blocks

    text_blocks = [
        cast("TextContentBlock", block)["text"]
        for block in content_blocks
        if cast("dict[str, object]", block).get("type") == "text"
    ]

    assert text_blocks
    text = text_blocks[0]
    assert text.startswith("## Skills System")
    assert "web-research" in text
    assert "/skills/web-research/SKILL.md" in text


def test_create_agent_injects_skills_into_model_prompt() -> None:
    middleware = SkillsMiddleware(backend=_SkillsBackend(), sources=["/skills/"])
    agent = create_agent(model=FakeToolCallingModel(), middleware=[middleware])

    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})

    messages = result["messages"]
    prompt_text = "\n".join(message.text for message in messages if hasattr(message, "text"))

    assert "## Skills System" in prompt_text
    assert "web-research" in prompt_text
    assert "/skills/web-research/SKILL.md" in prompt_text


def test_load_skill_content_reads_from_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "web-research"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("# skill\n", encoding="utf-8")

    assert load_skill_content(str(skill_path), allowed_roots=(root,)) == "# skill\n"


def test_load_skill_content_rejects_path_outside_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside all allowed"):
        load_skill_content(str(outside), allowed_roots=(root,))


def test_validate_tuple_source_accepts_valid_tuple() -> None:
    _validate_tuple_source(("/skills/user", "User"))


@pytest.mark.parametrize(
    "source",
    [
        (),
        ("only_path",),
        ("a", "b", "c"),
        (1, "User"),
        ("path", 1),
    ],
)
def test_validate_tuple_source_rejects_invalid_tuple(source: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="Invalid skill source"):
        _validate_tuple_source(source)


def test_source_path_from_string() -> None:
    assert _source_path("/skills/user") == "/skills/user"


def test_source_path_from_tuple() -> None:
    assert _source_path(("/skills/user", "User")) == "/skills/user"


def test_truncate_short_warning() -> None:
    text = "hello"
    assert _truncate_skill_load_warning(text) == text


def test_truncate_long_warning() -> None:
    text = "a" * (MAX_SKILL_LOAD_WARNING_LENGTH + 100)

    result = _truncate_skill_load_warning(text)

    assert len(result) == MAX_SKILL_LOAD_WARNING_LENGTH
    assert result.endswith("... [truncated]")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/skills/user", "User"),
        ("/built_in_skills", "Built-in"),
        ("/repo/.claude/skills", "Claude"),
        (("/repo/skills", "Project"), "Project"),
    ],
)
def test_derive_source_label(source: SkillSource, expected: str) -> None:
    assert _derive_source_label(source) == expected


def test_validate_skill_name_valid() -> None:
    valid, error = _validate_skill_name("web-search", "web-search")

    assert valid is True
    assert error == ""


@pytest.mark.parametrize(
    ("name", "directory"),
    [
        ("", ""),
        ("Web", "Web"),
        ("web--search", "web--search"),
        ("-web", "-web"),
        ("web-", "web-"),
        ("web search", "web search"),
        ("another", "web-search"),
    ],
)
def test_validate_skill_name_invalid(name: str, directory: str) -> None:
    valid, _ = _validate_skill_name(name, directory)

    assert valid is False


def test_parse_allowed_tools_string() -> None:
    assert _parse_allowed_tools(
        "bash grep python",
        "test_path",
    ) == ["bash", "grep", "python"]


def test_parse_allowed_tools_csv() -> None:
    assert _parse_allowed_tools(
        "bash,grep,python",
        "test_path",
    ) == ["bash", "grep", "python"]


def test_parse_allowed_tools_list() -> None:
    assert _parse_allowed_tools(
        ["bash", "grep", "python"],
        "test_path",
    ) == ["bash", "grep", "python"]


def test_parse_allowed_tools_invalid() -> None:
    assert _parse_allowed_tools(123, "test_path") == []


def test_validate_metadata_dict() -> None:
    result = _validate_metadata(
        {"a": 1, "b": True},
        "test_path",
    )

    assert result == {
        "a": "1",
        "b": "True",
    }


def test_validate_metadata_invalid() -> None:
    assert _validate_metadata("abc", "test_path") == {}


def test_format_skill_annotations_all_fields() -> None:
    skill: SkillMetadata = {
        "license": "MIT",
        "compatibility": "Python 3.12",
    }

    assert _format_skill_annotations(skill) == "License: MIT, Compatibility: Python 3.12"


def test_format_skill_annotations_empty() -> None:
    skill: SkillMetadata = {}

    assert _format_skill_annotations(skill) == ""


def test_append_to_system_message_with_none_creates_new_message() -> None:
    result = append_to_system_message(
        system_message=None,
        text="First instruction",
    )

    assert isinstance(result, SystemMessage)
    assert result.content_blocks == [
        {
            "type": "text",
            "text": "First instruction",
        }
    ]


def test_append_to_system_message_appends_text_to_existing_message() -> None:
    original_message = SystemMessage(
        content_blocks=[
            {
                "type": "text",
                "text": "Existing instruction",
            }
        ]
    )

    result = append_to_system_message(
        system_message=original_message,
        text="New instruction",
    )

    assert isinstance(result, SystemMessage)

    assert result.content_blocks == [
        {
            "type": "text",
            "text": "Existing instruction",
        },
        {
            "type": "text",
            "text": "\n\nNew instruction",
        },
    ]


def test_append_to_system_message_does_not_mutate_original_message() -> None:
    original_message = SystemMessage(
        content_blocks=[
            {
                "type": "text",
                "text": "Original",
            }
        ]
    )

    append_to_system_message(
        system_message=original_message,
        text="Added",
    )

    assert original_message.content_blocks == [
        {
            "type": "text",
            "text": "Original",
        }
    ]


def test_append_to_system_message_preserves_multiple_existing_blocks() -> None:
    original_message = SystemMessage(
        content_blocks=[
            {
                "type": "text",
                "text": "First",
            },
            {
                "type": "text",
                "text": "Second",
            },
        ]
    )

    result = append_to_system_message(
        system_message=original_message,
        text="Third",
    )

    assert result.content_blocks == [
        {
            "type": "text",
            "text": "First",
        },
        {
            "type": "text",
            "text": "Second",
        },
        {
            "type": "text",
            "text": "\n\nThird",
        },
    ]


def test_append_to_system_message_can_append_empty_text() -> None:
    original_message = SystemMessage(
        content_blocks=[
            {
                "type": "text",
                "text": "Existing",
            }
        ]
    )

    result = append_to_system_message(
        system_message=original_message,
        text="",
    )

    assert result.content_blocks[-1] == {
        "type": "text",
        "text": "\n\n",
    }
