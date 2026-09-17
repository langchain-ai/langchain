"""Experimental skills middleware powered by TypeSafe."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import yaml

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        PrivateStateAttr,
        TracePolicy,
        omit_payload,
    )
    from langgraph.runtime import Runtime
except ImportError as error:
    msg = (
        "SkillsMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import NotRequired, override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import ClassificationResponse, Noul

logger = logging.getLogger(__name__)

_SKILL_QUESTION_PREFIX = "skill::"
_MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class Skill(BaseModel):
    """An Agent Skills-compatible skill.

    Args:
        name: Unique lowercase skill name.
        description: What the skill does and when it should be used.
        content: Markdown instructions from the body of `SKILL.md`.
        license: License name or path to a bundled license file.
        compatibility: Environment requirements for using the skill.
        metadata: Additional string key-value metadata.
        allowed_tools: Space-separated tools pre-approved for the skill.
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    content: str
    license: str | None = None
    compatibility: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")

    @field_validator(
        "name",
        "description",
        "license",
        "compatibility",
        "allowed_tools",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: object) -> object:
        """Strip surrounding whitespace from frontmatter string fields."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        """Validate the Agent Skills name format."""
        if not _SKILL_NAME_PATTERN.fullmatch(name):
            msg = (
                "Skill names must contain only lowercase letters, numbers, and single "
                "hyphens, and cannot start or end with a hyphen."
            )
            raise ValueError(msg)
        return name

    @classmethod
    def from_markdown(cls, markdown: str, *, source: Path | None = None) -> Skill:
        """Parse a complete `SKILL.md` document.

        Args:
            markdown: YAML-frontmatter `SKILL.md` contents.
            source: Optional source path, used to validate the parent directory name.

        Returns:
            Parsed skill metadata and Markdown body.

        Raises:
            ValueError: If the document is too large, malformed, or violates the Agent
                Skills specification.
        """
        if len(markdown.encode("utf-8")) > _MAX_SKILL_FILE_SIZE:
            msg = "SKILL.md content exceeds the 10 MiB size limit."
            raise ValueError(msg)

        lines = markdown.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            msg = "SKILL.md must begin with YAML frontmatter delimited by `---`."
            raise ValueError(msg)

        try:
            end = next(
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration as error:
            msg = "SKILL.md frontmatter is missing its closing `---` delimiter."
            raise ValueError(msg) from error

        try:
            frontmatter = yaml.safe_load("".join(lines[1:end]))
        except yaml.YAMLError as error:
            msg = "SKILL.md contains invalid YAML frontmatter."
            raise ValueError(msg) from error
        if not isinstance(frontmatter, dict):
            msg = "SKILL.md frontmatter must be a YAML mapping."
            raise TypeError(msg)

        data = dict(frontmatter)
        data["content"] = "".join(lines[end + 1 :])
        skill = cls.model_validate(data)
        if source is not None and source.parent.name != skill.name:
            msg = (
                f"Skill name {skill.name!r} must match its parent directory "
                f"{source.parent.name!r}."
            )
            raise ValueError(msg)
        return skill


SkillSource = Skill | Path | str
"""A skill object, path to `SKILL.md`, or complete `SKILL.md` contents."""


class _SkillsState(AgentState):
    """Agent state used to persist the selected skill for one run."""

    selected_skills: NotRequired[Annotated[list[str], PrivateStateAttr]]


def _load_skill(source: SkillSource) -> Skill:
    """Normalize a supported skill source."""
    if isinstance(source, Skill):
        return source
    if isinstance(source, Path):
        logical_path = source.expanduser()
        try:
            resolved = logical_path.resolve(strict=True)
            if not resolved.is_file() or logical_path.name != "SKILL.md":
                msg = f"Skill path must point to a SKILL.md file: {source}."
                raise ValueError(msg)
            if resolved.stat().st_size > _MAX_SKILL_FILE_SIZE:
                msg = "SKILL.md content exceeds the 10 MiB size limit."
                raise ValueError(msg)
            markdown = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            msg = f"Unable to read SKILL.md from {source}."
            raise ValueError(msg) from error
        return Skill.from_markdown(markdown, source=logical_path)
    return Skill.from_markdown(source)


class SkillsMiddleware(AgentMiddleware[_SkillsState]):
    """Inject the TypeSafe-selected skill into model-request messages.

    The middleware performs one classification on the latest human message before an
    agent run. TypeSafe evaluates every skill independently, so any number of relevant
    skills can be selected. The complete roster stays out of the model prompt, and only
    selected skills' instructions are prepended to model-request messages.

    Skill sources are trusted configuration. Selected instructions are added as a
    system message and can direct agent behavior. Do not load user-controlled
    `SKILL.md` content.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Install the `experimental` extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        skills: Skills represented as `Skill` objects, paths to `SKILL.md`, or complete
            `SKILL.md` strings.
        relevance_threshold: Minimum relevance probability required to add each skill.

    Raises:
        ValueError: If the skill roster is empty, contains duplicate names, or includes
            an invalid source.

    Example:
        ```python
        from pathlib import Path

        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import SkillsMiddleware

        skills = SkillsMiddleware(
            skills=[Path("skills/code-review/SKILL.md")]
        )
        agent = create_agent(model, middleware=[skills])
        ```
    """

    state_schema = _SkillsState  # type: ignore[assignment]
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        skills: Sequence[SkillSource],
        relevance_threshold: float = 0.3,
    ) -> None:
        """Initialize the skills middleware."""
        super().__init__()
        if not 0 <= relevance_threshold <= 1:
            msg = "`relevance_threshold` must be between 0 and 1."
            raise ValueError(msg)
        loaded_skills = [_load_skill(source) for source in skills]
        if not loaded_skills:
            msg = "At least one skill is required."
            raise ValueError(msg)
        duplicates = {
            skill.name
            for skill in loaded_skills
            if sum(item.name == skill.name for item in loaded_skills) > 1
        }
        if duplicates:
            msg = f"Skill names must be unique; duplicates: {sorted(duplicates)}."
            raise ValueError(msg)

        self.skills = {skill.name: skill for skill in loaded_skills}
        self.relevance_threshold = relevance_threshold
        self.classifier = TypeSafeClassifier(
            questions={
                f"{_SKILL_QUESTION_PREFIX}{skill.name}": Noul(
                    instructions=(
                        "Does this skill directly apply to the user's latest request? "
                        f"Skill: {skill.name}. Description: {skill.description}"
                    )
                )
                for skill in loaded_skills
            },
        )

    def _classification_input(self, state: _SkillsState) -> HumanMessage | None:
        """Return the latest human message from agent state."""
        return next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, HumanMessage)
            ),
            None,
        )

    def _selected_skills(self, response: ClassificationResponse) -> list[str]:
        """Return every skill whose independent relevance score passes the threshold."""
        return [
            name
            for name in self.skills
            if (answer := response.nouls.get(f"{_SKILL_QUESTION_PREFIX}{name}"))
            is not None
            and answer.noul >= self.relevance_threshold
        ]

    def _classification_config(self) -> RunnableConfig:
        """Return tracing metadata for the internal classification call."""
        return {"metadata": {"lc_source": "typesafe_skills"}}

    @override
    def before_agent(
        self,
        state: _SkillsState,
        runtime: Runtime[Any],
    ) -> dict[str, list[str]]:
        """Select skills for the latest human request."""
        del runtime
        classifier_input = self._classification_input(state)
        if classifier_input is None:
            return {"selected_skills": []}
        try:
            response = self.classifier.invoke(
                classifier_input,
                config=self._classification_config(),
            )
            selected = self._selected_skills(response)
        except Exception:
            logger.exception(
                "TypeSafe skill classification failed; no skills were added."
            )
            selected = []
        return {"selected_skills": selected}

    @override
    async def abefore_agent(
        self,
        state: _SkillsState,
        runtime: Runtime[Any],
    ) -> dict[str, list[str]]:
        """Select skills for the latest human request asynchronously."""
        del runtime
        classifier_input = self._classification_input(state)
        if classifier_input is None:
            return {"selected_skills": []}
        try:
            response = await self.classifier.ainvoke(
                classifier_input,
                config=self._classification_config(),
            )
            selected = self._selected_skills(response)
        except Exception:
            logger.exception(
                "TypeSafe skill classification failed; no skills were added."
            )
            selected = []
        return {"selected_skills": selected}

    def _add_selected_skills(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Return a request with all selected skills prepended to its messages."""
        selected = request.state.get("selected_skills", [])
        if not isinstance(selected, list):
            return request
        skill_messages = [
            SystemMessage(
                content=f"<skill name={name!r}>\n{self.skills[name].content}\n</skill>"
            )
            for name in selected
            if isinstance(name, str) and name in self.skills
        ]
        if not skill_messages:
            return request
        return request.override(messages=[*skill_messages, *request.messages])

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Add selected skills to a synchronous model call."""
        return handler(self._add_selected_skills(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Add selected skills to an asynchronous model call."""
        return await handler(self._add_selected_skills(request))


__all__ = ["Skill", "SkillSource", "SkillsMiddleware"]
