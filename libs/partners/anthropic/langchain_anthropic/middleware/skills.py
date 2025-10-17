"""Anthropic Skills middleware.

Requires:
    - langchain: For agent middleware framework
    - langchain-anthropic: For ChatAnthropic model (already a dependency)
"""

import hashlib
import os
import zipfile
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from warnings import warn

from langchain_anthropic.chat_models import ChatAnthropic

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "ClaudeSkillsMiddleware requires 'langchain' to be installed. "
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


# Pre-built Anthropic skill IDs
ANTHROPIC_SKILLS = {"pptx", "xlsx", "docx", "pdf"}

# Required beta headers for Skills functionality
REQUIRED_BETAS = [
    "code-execution-2025-08-25",
    "skills-2025-10-02",
    "files-api-2025-04-14",
]


class SkillConfig:
    """Configuration for a single skill.

    Args:
        skill_id: The skill identifier. For Anthropic pre-built skills, use
            `'pptx'`, `'xlsx'`, `'docx'`, or `'pdf'`. For custom skills, use
            the ID from the `/v1/skills` endpoint.
        type: The skill type. `'anthropic'` for pre-built skills or `'custom'`
            for user-uploaded skills.
        version: The skill version. Use `'latest'` for the most recent version,
            or specify a date-based version for Anthropic skills (e.g., `'20251002'`)
            or epoch timestamp for custom skills.
    """

    def __init__(
        self,
        skill_id: str,
        type: Literal["anthropic", "custom"] = "anthropic",  # noqa: A002
        version: str = "latest",
    ) -> None:
        """Initialize skill configuration."""
        self.skill_id = skill_id
        self.type = type
        self.version = version

    def to_dict(self) -> dict[str, str]:
        """Convert skill configuration to API format.

        Returns:
            Dictionary with skill configuration for API requests.
        """
        return {
            "type": self.type,
            "skill_id": self.skill_id,
            "version": self.version,
        }


class LocalSkillConfig(SkillConfig):
    """Configuration for a skill loaded from local files.

    This class handles loading skills from the local filesystem and uploading
    them to the Anthropic API. Skills are expected to be in a directory containing
    a `SKILL.md` file with YAML frontmatter.

    Args:
        path: Path to the skill directory or zip file containing the skill.
            The directory must contain a `SKILL.md` file with required frontmatter.
        display_title: Optional display title for the skill when uploaded.
            If not provided, the name from the SKILL.md frontmatter will be used.
        auto_upload: If `True`, automatically upload the skill when the middleware
            is initialized. If `False`, you must manually upload via the
            `upload_skill` method.

    Example:
        ```python
        from langchain_anthropic.middleware import (
            ClaudeSkillsMiddleware,
            LocalSkillConfig,
        )

        # Create middleware with local skill
        middleware = ClaudeSkillsMiddleware(
            skills=[
                "pptx",  # Pre-built skill
                LocalSkillConfig(path="./my-custom-skill"),  # Local skill
            ]
        )
        ```
    """

    def __init__(
        self,
        path: str | Path,
        *,
        display_title: str | None = None,
        auto_upload: bool = True,
    ) -> None:
        """Initialize local skill configuration.

        Args:
            path: Path to the skill directory or zip file.
            display_title: Optional display title for the skill.
            auto_upload: Whether to automatically upload the skill.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is not a valid skill directory or zip file.
        """
        self.path = Path(path)
        self.display_title = display_title
        self.auto_upload = auto_upload
        self._uploaded_skill_id: str | None = None
        self._content_hash: str | None = None

        # Validate path exists
        if not self.path.exists():
            msg = f"Skill path does not exist: {self.path}"
            raise FileNotFoundError(msg)

        # Validate it's a directory or zip file
        if not (self.path.is_dir() or self.path.suffix == ".zip"):
            msg = f"Skill path must be a directory or .zip file, got: {self.path}"
            raise ValueError(msg)

        # For directories, validate SKILL.md exists
        if self.path.is_dir():
            skill_md = self.path / "SKILL.md"
            if not skill_md.exists():
                msg = f"Skill directory must contain a SKILL.md file: {self.path}"
                raise ValueError(msg)

        # Initialize parent class with placeholder values
        # These will be set after upload
        super().__init__(
            skill_id="",  # Will be set after upload
            type="custom",
            version="latest",
        )

    def _compute_content_hash(self) -> str:
        """Compute a hash of the skill content for caching.

        Returns:
            SHA256 hash of the skill content.
        """
        hasher = hashlib.sha256()

        if self.path.is_dir():
            # Hash all files in the directory
            for root, _dirs, files in os.walk(self.path):
                for file in sorted(files):  # Sort for consistent hashing
                    file_path = Path(root) / file
                    hasher.update(str(file_path.relative_to(self.path)).encode())
                    hasher.update(file_path.read_bytes())
        else:
            # Hash the zip file
            hasher.update(self.path.read_bytes())

        return hasher.hexdigest()

    def _create_zip_bytes(self) -> bytes:
        """Create a zip file from the skill directory or read existing zip.

        The zip must contain a top-level folder with all files inside it,
        as required by the Anthropic Skills API.

        Returns:
            Bytes of the zip file.
        """
        if self.path.suffix == ".zip":
            return self.path.read_bytes()

        # Create zip file in memory with top-level folder
        # The directory name becomes the top-level folder in the zip
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _dirs, files in os.walk(self.path):
                for file in files:
                    file_path = Path(root) / file
                    # Create path relative to parent, preserving directory name
                    rel_path = file_path.relative_to(self.path.parent)
                    zip_file.write(file_path, str(rel_path))

        return zip_buffer.getvalue()

    def upload_skill(self, client: Any) -> str:
        """Upload the skill to Anthropic API.

        Args:
            client: Anthropic client instance.

        Returns:
            The uploaded skill ID.

        Raises:
            ValueError: If upload fails.
        """
        # Check if already uploaded with same content
        content_hash = self._compute_content_hash()
        if self._uploaded_skill_id and self._content_hash == content_hash:
            return self._uploaded_skill_id

        # Create zip file
        zip_bytes = self._create_zip_bytes()

        # Upload to Anthropic
        try:
            response = client.beta.skills.create(
                files=[("skill.zip", zip_bytes, "application/zip")],
                display_title=self.display_title,
                betas=REQUIRED_BETAS,
            )

            # Store the skill ID and hash
            self._uploaded_skill_id = response.id
            self._content_hash = content_hash
            self.skill_id = response.id

            return response.id

        except Exception as e:
            msg = f"Failed to upload skill from {self.path}: {e}"
            raise ValueError(msg) from e

    @property
    def is_uploaded(self) -> bool:
        """Check if the skill has been uploaded.

        Returns:
            `True` if the skill has been uploaded, `False` otherwise.
        """
        return self._uploaded_skill_id is not None


class ClaudeSkillsMiddleware(AgentMiddleware):
    """Skills Middleware for Claude.

    Enables Claude to use Skills - specialized capabilities for document processing,
    data manipulation, and other domain-specific tasks. Skills are folders containing
    instructions, scripts, and resources that Claude loads when relevant.

    This middleware:

    - Adds skills to model requests via the `container` parameter
    - Injects required beta headers for Skills functionality
    - Automatically adds code execution tool if not present (required for Skills)
    - Supports both Anthropic pre-built skills and custom skills
    - Handles uploading local skills to the Anthropic API

    Requires both 'langchain' and 'langchain-anthropic' packages to be installed.

    Learn more about Claude Skills
    [here](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview).

    Example:
        Basic usage with pre-built skills:

        ```python
        from langchain_anthropic.middleware import ClaudeSkillsMiddleware

        # Enable PowerPoint and Excel skills
        skills_middleware = ClaudeSkillsMiddleware(skills=["pptx", "xlsx"])

        # Use with an agent
        agent = create_react_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
            tools=[...],
            middleware=[skills_middleware],
        )
        ```

        Advanced usage with custom skills from API:

        ```python
        from langchain_anthropic.middleware import (
            ClaudeSkillsMiddleware,
            SkillConfig,
        )

        # Mix pre-built and custom skills
        skills_middleware = ClaudeSkillsMiddleware(
            skills=[
                "pptx",  # Pre-built skill
                SkillConfig(
                    skill_id="custom-skill-123",
                    type="custom",
                    version="latest",
                ),
            ]
        )
        ```

        Using local skill files:

        ```python
        from pathlib import Path
        from langchain_anthropic.middleware import (
            ClaudeSkillsMiddleware,
            LocalSkillConfig,
        )

        # Load skill from local directory
        skills_middleware = ClaudeSkillsMiddleware(
            skills=[
                "xlsx",  # Pre-built skill
                LocalSkillConfig(
                    path="./my-custom-skill",  # Directory with SKILL.md
                    display_title="My Custom Skill",
                    auto_upload=True,  # Automatically upload when first used
                ),
            ]
        )
        ```
    """

    def __init__(
        self,
        skills: list[str | SkillConfig | LocalSkillConfig],
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
        code_execution_tool_name: str = "code_execution",
    ) -> None:
        """Initialize the middleware with skills configuration.

        Args:
            skills: List of skills to enable. Can be:
                - Skill ID strings for Anthropic pre-built skills
                  (`'pptx'`, `'xlsx'`, `'docx'`, `'pdf'`)
                - `SkillConfig` objects for custom skills or fine-grained control
                - `LocalSkillConfig` objects for skills loaded from local files
            unsupported_model_behavior: Behavior when a non-Anthropic model is used.
                - `'ignore'`: Skip skills injection silently
                - `'warn'`: Log a warning and skip skills injection
                - `'raise'`: Raise an error
            code_execution_tool_name: Name of the code execution tool. If not present
                in the request, it will be automatically added.

        Raises:
            ValueError: If more than 8 skills are provided (API limit) or if
                invalid skill configuration is detected.
        """
        if len(skills) > 8:
            msg = (
                "Anthropic API supports a maximum of 8 skills per request, "
                f"but {len(skills)} were provided."
            )
            raise ValueError(msg)

        self.skills = self._normalize_skills(skills)
        self.unsupported_model_behavior = unsupported_model_behavior
        self.code_execution_tool_name = code_execution_tool_name

    def _normalize_skills(
        self, skills: list[str | SkillConfig | LocalSkillConfig]
    ) -> list[SkillConfig | LocalSkillConfig]:
        """Normalize skill configurations.

        Args:
            skills: List of skill IDs, SkillConfig objects, or LocalSkillConfig objects.

        Returns:
            List of SkillConfig or LocalSkillConfig objects.
        """
        normalized = []
        for skill in skills:
            if isinstance(skill, str):
                # Infer type based on skill_id
                skill_type: Literal["anthropic", "custom"] = (
                    "anthropic" if skill in ANTHROPIC_SKILLS else "custom"
                )
                normalized.append(
                    SkillConfig(skill_id=skill, type=skill_type, version="latest")
                )
            else:
                normalized.append(skill)
        return normalized

    def _should_apply_skills(self, request: ModelRequest) -> bool:
        """Check if skills should be applied to the request.

        Args:
            request: The model request to check.

        Returns:
            `True` if skills should be applied, `False` otherwise.

        Raises:
            ValueError: If model is unsupported and behavior is set to `'raise'`.
        """
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "ClaudeSkillsMiddleware only supports Anthropic models, "
                f"not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        return True

    def _ensure_code_execution_tool(self, request: ModelRequest) -> None:
        """Ensure code execution tool is present in the request.

        Skills require the code execution tool to be enabled. If not present,
        this method will automatically add it.

        Args:
            request: The model request to check and modify.
        """
        tools = request.tools or []

        # Check if code execution tool exists
        has_code_execution = any(
            (
                isinstance(tool, dict)
                and tool.get("name") == self.code_execution_tool_name
            )
            or (isinstance(tool, str) and tool == self.code_execution_tool_name)
            or (hasattr(tool, "name") and tool.name == self.code_execution_tool_name)
            for tool in tools
        )

        if not has_code_execution:
            # Automatically add code execution tool
            tools = list(tools)
            tools.append(
                {
                    "type": "code_execution_20250825",
                    "name": self.code_execution_tool_name,
                }
            )
            request.tools = tools

    def _upload_local_skills(self, request: ModelRequest) -> None:
        """Upload any local skills that need uploading.

        Args:
            request: The model request to get Anthropic client from.

        Raises:
            ValueError: If upload fails or if model doesn't have a client.
        """
        # Check if there are any local skills that need uploading
        has_local_skills = any(
            isinstance(skill, LocalSkillConfig)
            and skill.auto_upload
            and not skill.is_uploaded
            for skill in self.skills
        )

        if not has_local_skills:
            return

        # Get Anthropic client from the model
        if not isinstance(request.model, ChatAnthropic):
            return

        # Get the underlying client
        if not hasattr(request.model, "_client"):
            msg = "ChatAnthropic model does not have a _client attribute"
            raise ValueError(msg)

        client = request.model._client  # noqa: SLF001

        # Upload any local skills that need it
        for skill in self.skills:
            if (
                isinstance(skill, LocalSkillConfig)
                and skill.auto_upload
                and not skill.is_uploaded
            ):
                skill.upload_skill(client)

    def _inject_skills_config(self, request: ModelRequest) -> None:
        """Inject skills configuration into the request.

        Args:
            request: The model request to modify.
        """
        # Add required beta headers
        model_settings = request.model_settings or {}
        existing_betas = model_settings.get("betas", [])

        # Merge with required betas, avoiding duplicates
        all_betas = list(set(existing_betas + REQUIRED_BETAS))
        model_settings["betas"] = all_betas

        # Create or update container configuration
        container = model_settings.get("container", {})
        container["skills"] = [skill.to_dict() for skill in self.skills]
        model_settings["container"] = container

        # Update request
        request.model_settings = model_settings

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Modify the model request to add skills configuration.

        Args:
            request: The model request to potentially modify.
            handler: The handler to execute the model request.

        Returns:
            The model response from the handler.

        Raises:
            ValueError: If the model is unsupported and behavior is set to `'raise'`.
        """
        if not self._should_apply_skills(request):
            return handler(request)

        self._ensure_code_execution_tool(request)
        self._upload_local_skills(request)
        self._inject_skills_config(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Modify the model request to add skills configuration (async version).

        Args:
            request: The model request to potentially modify.
            handler: The async handler to execute the model request.

        Returns:
            The model response from the handler.

        Raises:
            ValueError: If the model is unsupported and behavior is set to `'raise'`.
        """
        if not self._should_apply_skills(request):
            return await handler(request)

        self._ensure_code_execution_tool(request)
        self._upload_local_skills(request)
        self._inject_skills_config(request)
        return await handler(request)
