"""Skills middleware powered by TypeSafe."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typesafe_sdk as ts
import yaml
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from typing_extensions import NotRequired, override

from langchain_typesafe._classify import TypeSafeClassifier, log_classification_failure

logger = logging.getLogger(__name__)

_SKILL_QUESTION_PREFIX = "skill::"
_MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_FILENAME = "SKILL.md"


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
            TypeError: If the frontmatter is not a YAML mapping.
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
    """Agent state used to persist the selected skills for one run."""

    selected_skills: NotRequired[Annotated[list[str], PrivateStateAttr]]


def _read_skill_file(path: Path, root: Path | None) -> str:
    """Read a `SKILL.md` file, enforcing containment and a size limit.

    The file is opened once and both the containment check and the size limit are
    applied to that open handle, so a path that changes between checking and reading
    cannot be used to read something else.

    Args:
        path: Path to a `SKILL.md` file.
        root: Directory the file must resolve inside, or `None` to allow any path.

    Returns:
        The file's contents.

    Raises:
        ValueError: If the path is not a `SKILL.md` file, resolves outside `root`, or
            exceeds the size limit.
    """
    logical_path = path.expanduser()
    if logical_path.name != _SKILL_FILENAME:
        msg = f"Skill path must point to a {_SKILL_FILENAME} file: {path}."
        raise ValueError(msg)
    try:
        resolved = logical_path.resolve(strict=True)
    except OSError as error:
        msg = f"Unable to resolve {_SKILL_FILENAME} path: {path}."
        raise ValueError(msg) from error
    if root is not None:
        resolved_root = root.expanduser().resolve()
        if not resolved.is_relative_to(resolved_root):
            msg = (
                f"Skill path {path} resolves outside the configured skills root {root}."
            )
            raise ValueError(msg)
    try:
        with resolved.open("rb") as handle:
            if not resolved.is_file():
                msg = f"Skill path must point to a regular file: {path}."
                raise ValueError(msg)
            # Read one byte past the limit so an oversized file is rejected rather
            # than silently truncated.
            raw = handle.read(_MAX_SKILL_FILE_SIZE + 1)
    except OSError as error:
        msg = f"Unable to read {_SKILL_FILENAME} from {path}."
        raise ValueError(msg) from error
    if len(raw) > _MAX_SKILL_FILE_SIZE:
        msg = "SKILL.md content exceeds the 10 MiB size limit."
        raise ValueError(msg)
    try:
        return raw.decode("utf-8")
    except UnicodeError as error:
        msg = f"{_SKILL_FILENAME} at {path} is not valid UTF-8."
        raise ValueError(msg) from error


def _load_skill(source: SkillSource, root: Path | None) -> Skill:
    """Normalize a supported skill source.

    Args:
        source: A skill, a path to `SKILL.md`, or `SKILL.md` contents.
        root: Directory that path sources must resolve inside.

    Returns:
        The loaded skill.

    Raises:
        ValueError: If the source cannot be read or is not a valid skill.
    """
    if isinstance(source, Skill):
        return source
    if isinstance(source, Path):
        markdown = _read_skill_file(source, root)
        return Skill.from_markdown(markdown, source=source.expanduser())
    return Skill.from_markdown(source)


class SkillsMiddleware(AgentMiddleware[_SkillsState]):
    """Inject TypeSafe-selected skills into model-request messages.

    The middleware performs one classification on the latest human message before an
    agent run. TypeSafe evaluates every skill independently, so any number of
    relevant skills can be selected. The complete roster stays out of the model
    prompt, and only the selected skills' instructions are prepended to
    model-request messages.

    !!! warning

        This middleware is experimental. Its API may change without notice.

        Skill sources are trusted configuration. Selected content is added as a
        system message and can direct agent behavior, so never load a `SKILL.md`
        a user can write. Pass `skills_root` to confine path sources to one
        directory.

    Args:
        skills: Skills represented as `Skill` objects, paths to `SKILL.md`, or
            complete `SKILL.md` strings.
        relevance_threshold: Minimum relevance probability required to add a skill.
        skills_root: Directory that every path source must resolve inside. Paths
            resolving outside it are rejected, which stops a symlink or `..`
            component from pulling in a file outside the skill library.
        api_key: TypeSafe API key. If omitted, reads `TYPESAFE_API_KEY`.
        base_url: Root URL for the TypeSafe API.
        model: TypeSafe model used for the relevance decision.
        timeout: Timeout in seconds for the relevance request.
        retry: Retry policy for the relevance request.
        client: Optional synchronous TypeSafe client.
        async_client: Optional asynchronous TypeSafe client.

    Raises:
        ValueError: If the skill roster is empty, contains duplicate names, includes
            an invalid source, or `relevance_threshold` is outside `[0, 1]`.

    ??? example "Select skills from a skill library"

        ```python
        from pathlib import Path

        from langchain.agents import create_agent
        from langchain_typesafe import SkillsMiddleware

        library = Path("skills")
        skills = SkillsMiddleware(
            skills=[library / "code-review" / "SKILL.md"],
            skills_root=library,
        )
        agent = create_agent(model, middleware=[skills])
        ```
    """

    state_schema = _SkillsState  # type: ignore[assignment]
    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Keep the classified request payload out of middleware traces."""

    def __init__(
        self,
        *,
        skills: Sequence[SkillSource],
        relevance_threshold: float = 0.3,
        skills_root: Path | None = None,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retry: ts.RetryPolicy | None = None,
        client: ts.TypeSafeClient | None = None,
        async_client: ts.AsyncTypeSafeClient | None = None,
    ) -> None:
        """Initialize the skills middleware."""
        super().__init__()
        if not 0 <= relevance_threshold <= 1:
            msg = "`relevance_threshold` must be between 0 and 1."
            raise ValueError(msg)
        loaded_skills = [_load_skill(source, skills_root) for source in skills]
        if not loaded_skills:
            msg = "At least one skill is required."
            raise ValueError(msg)
        duplicates = sorted(
            {
                skill.name
                for skill in loaded_skills
                if sum(item.name == skill.name for item in loaded_skills) > 1
            }
        )
        if duplicates:
            msg = f"Skill names must be unique; duplicates: {duplicates}."
            raise ValueError(msg)

        self.skills = {skill.name: skill for skill in loaded_skills}
        self.relevance_threshold = relevance_threshold
        self._classifier = TypeSafeClassifier(
            {
                f"{_SKILL_QUESTION_PREFIX}{skill.name}": ts.Noul(
                    instructions=(
                        "Does this skill directly apply to the user's latest request? "
                        f"Skill: {skill.name}. Description: {skill.description}"
                    )
                )
                for skill in loaded_skills
            },
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            retry=retry,
            client=client,
            async_client=async_client,
        )

    def _classification_state(self, state: _SkillsState) -> dict[str, Any] | None:
        """Return the latest human message as classifiable state."""
        message = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if message is None:
            return None
        return convert_to_openai_messages(message)

    def _selected_skills(self, response: ts.SystemOneResponse) -> list[str]:
        """Return every skill whose independent relevance score passes the threshold."""
        return [
            name
            for name in self.skills
            if (answer := response.nouls.get(f"{_SKILL_QUESTION_PREFIX}{name}"))
            is not None
            and answer.noul >= self.relevance_threshold
        ]

    @override
    def before_agent(
        self,
        state: _SkillsState,
        runtime: Runtime[Any],
    ) -> dict[str, list[str]]:
        """Select skills for the latest human request."""
        del runtime
        classifier_state = self._classification_state(state)
        if classifier_state is None:
            return {"selected_skills": []}
        try:
            selected = self._selected_skills(
                self._classifier.classify(classifier_state)
            )
        except Exception as error:  # noqa: BLE001 - skills must not break the agent
            log_classification_failure(logger, error, "no skills were added")
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
        classifier_state = self._classification_state(state)
        if classifier_state is None:
            return {"selected_skills": []}
        try:
            response = await self._classifier.aclassify(classifier_state)
            selected = self._selected_skills(response)
        except Exception as error:  # noqa: BLE001 - skills must not break the agent
            log_classification_failure(logger, error, "no skills were added")
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
