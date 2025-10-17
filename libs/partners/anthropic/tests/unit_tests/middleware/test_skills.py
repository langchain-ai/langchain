"""Tests for Anthropic Skills middleware."""

import tempfile
import warnings
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_anthropic.middleware import (
    ClaudeSkillsMiddleware,
    LocalSkillConfig,
    SkillConfig,
)


class FakeToolCallingModel(BaseChatModel):
    """Fake model for testing middleware."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async top level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"


def test_skill_config_initialization() -> None:
    """Test SkillConfig initialization and to_dict method."""
    # Test with default values (Anthropic skill)
    skill = SkillConfig(skill_id="pptx")
    assert skill.skill_id == "pptx"
    assert skill.type == "anthropic"
    assert skill.version == "latest"
    assert skill.to_dict() == {
        "type": "anthropic",
        "skill_id": "pptx",
        "version": "latest",
    }

    # Test with custom skill and specific version
    skill = SkillConfig(
        skill_id="custom-123",
        type="custom",
        version="1234567890",
    )
    assert skill.skill_id == "custom-123"
    assert skill.type == "custom"
    assert skill.version == "1234567890"
    assert skill.to_dict() == {
        "type": "custom",
        "skill_id": "custom-123",
        "version": "1234567890",
    }


def test_claude_skills_middleware_initialization() -> None:
    """Test ClaudeSkillsMiddleware initialization."""
    # Test with string skill IDs (Anthropic pre-built skills)
    middleware = ClaudeSkillsMiddleware(skills=["pptx", "xlsx"])
    assert len(middleware.skills) == 2
    assert middleware.skills[0].skill_id == "pptx"
    assert middleware.skills[0].type == "anthropic"
    assert middleware.skills[1].skill_id == "xlsx"
    assert middleware.skills[1].type == "anthropic"

    # Test with SkillConfig objects
    middleware = ClaudeSkillsMiddleware(
        skills=[
            SkillConfig(skill_id="pptx", type="anthropic", version="20251002"),
            SkillConfig(skill_id="custom-123", type="custom", version="latest"),
        ]
    )
    assert len(middleware.skills) == 2
    assert middleware.skills[0].version == "20251002"
    assert middleware.skills[1].type == "custom"

    # Test with mixed skills
    middleware = ClaudeSkillsMiddleware(
        skills=[
            "pptx",
            SkillConfig(skill_id="custom-456", type="custom"),
        ]
    )
    assert len(middleware.skills) == 2
    assert middleware.skills[0].skill_id == "pptx"
    assert middleware.skills[1].skill_id == "custom-456"


def test_claude_skills_middleware_max_skills_validation() -> None:
    """Test that ClaudeSkillsMiddleware validates maximum skill count."""
    # Should raise error when more than 8 skills are provided
    with pytest.raises(
        ValueError,
        match="Anthropic API supports a maximum of 8 skills per request",
    ):
        ClaudeSkillsMiddleware(skills=["pptx"] * 9)


def test_claude_skills_middleware_code_execution_auto_added() -> None:
    """Test that code execution tool is automatically added if missing."""
    middleware = ClaudeSkillsMiddleware(skills=["pptx"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Request without code execution tool
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Create a presentation")],
        system_prompt=None,
        tool_choice=None,
        tools=[],  # No code execution tool initially
        response_format=None,
        state={"messages": [HumanMessage("Create a presentation")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Should automatically add code_execution tool
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)

    # Verify code_execution was added to tools
    assert len(fake_request.tools) == 1
    tool = fake_request.tools[0]
    assert isinstance(tool, dict)
    assert tool["name"] == "code_execution"
    assert tool["type"] == "code_execution_20250825"


def test_claude_skills_middleware_basic_functionality() -> None:
    """Test ClaudeSkillsMiddleware basic functionality."""
    middleware = ClaudeSkillsMiddleware(skills=["pptx", "xlsx"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Create a presentation")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Create a presentation")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)

    # Check that model_settings were properly configured
    assert "betas" in fake_request.model_settings
    assert "code-execution-2025-08-25" in fake_request.model_settings["betas"]
    assert "skills-2025-10-02" in fake_request.model_settings["betas"]
    assert "files-api-2025-04-14" in fake_request.model_settings["betas"]

    # Check that container with skills was added
    assert "container" in fake_request.model_settings
    assert "skills" in fake_request.model_settings["container"]
    skills = fake_request.model_settings["container"]["skills"]
    assert len(skills) == 2
    assert skills[0]["skill_id"] == "pptx"
    assert skills[1]["skill_id"] == "xlsx"


def test_claude_skills_middleware_preserves_existing_betas() -> None:
    """Test that middleware preserves existing beta headers."""
    middleware = ClaudeSkillsMiddleware(skills=["pptx"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Create a presentation")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Create a presentation")]},
        runtime=cast(Runtime, object()),
        model_settings={"betas": ["existing-beta-header"]},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    middleware.wrap_model_call(fake_request, mock_handler)

    # Check that existing betas are preserved
    betas = fake_request.model_settings["betas"]
    assert "existing-beta-header" in betas
    assert "code-execution-2025-08-25" in betas
    assert "skills-2025-10-02" in betas
    assert "files-api-2025-04-14" in betas


def test_claude_skills_middleware_unsupported_model() -> None:
    """Test ClaudeSkillsMiddleware with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="raise"
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test raise behavior
    with pytest.raises(
        ValueError,
        match="ClaudeSkillsMiddleware only supports Anthropic models",
    ):
        middleware.wrap_model_call(fake_request, mock_handler)

    # Test warn behavior
    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="warn"
    )

    with warnings.catch_warnings(record=True) as w:
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert "ClaudeSkillsMiddleware only supports Anthropic models" in str(
            w[-1].message
        )

    # Test ignore behavior
    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="ignore"
    )
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


def test_claude_skills_middleware_custom_tool_name() -> None:
    """Test middleware with custom code execution tool name."""
    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], code_execution_tool_name="custom_code_exec"
    )

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Request with custom tool name
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Create a presentation")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "custom_exec", "name": "custom_code_exec"}],
        response_format=None,
        state={"messages": [HumanMessage("Create a presentation")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Should work with custom tool name
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_claude_skills_middleware_async() -> None:
    """Test ClaudeSkillsMiddleware async path."""
    middleware = ClaudeSkillsMiddleware(skills=["pptx", "docx"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Create a document")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Create a document")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)

    # Verify skills were configured
    assert "container" in fake_request.model_settings
    skills = fake_request.model_settings["container"]["skills"]
    assert len(skills) == 2
    assert skills[0]["skill_id"] == "pptx"
    assert skills[1]["skill_id"] == "docx"


async def test_claude_skills_middleware_async_unsupported_model() -> None:
    """Test ClaudeSkillsMiddleware async path with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="raise"
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test raise behavior
    with pytest.raises(
        ValueError,
        match="ClaudeSkillsMiddleware only supports Anthropic models",
    ):
        await middleware.awrap_model_call(fake_request, mock_handler)

    # Test warn behavior
    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="warn"
    )

    with warnings.catch_warnings(record=True) as w:
        result = await middleware.awrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert "ClaudeSkillsMiddleware only supports Anthropic models" in str(
            w[-1].message
        )

    # Test ignore behavior
    middleware = ClaudeSkillsMiddleware(
        skills=["pptx"], unsupported_model_behavior="ignore"
    )
    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_claude_skills_middleware_async_code_execution_auto_added() -> None:
    """Test that async path automatically adds code execution tool."""
    middleware = ClaudeSkillsMiddleware(skills=["xlsx"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Request without code execution tool
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Process spreadsheet")],
        system_prompt=None,
        tool_choice=None,
        tools=[],  # No code execution tool initially
        response_format=None,
        state={"messages": [HumanMessage("Process spreadsheet")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Should automatically add code_execution tool
    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)

    # Verify code_execution was added to tools
    assert len(fake_request.tools) == 1
    tool = fake_request.tools[0]
    assert isinstance(tool, dict)
    assert tool["name"] == "code_execution"
    assert tool["type"] == "code_execution_20250825"


def test_claude_skills_middleware_all_pre_built_skills() -> None:
    """Test middleware with all pre-built Anthropic skills."""
    middleware = ClaudeSkillsMiddleware(skills=["pptx", "xlsx", "docx", "pdf"])

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Process documents")],
        system_prompt=None,
        tool_choice=None,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        response_format=None,
        state={"messages": [HumanMessage("Process documents")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)

    # Verify all skills are configured
    skills = fake_request.model_settings["container"]["skills"]
    assert len(skills) == 4
    skill_ids = {skill["skill_id"] for skill in skills}
    assert skill_ids == {"pptx", "xlsx", "docx", "pdf"}
    # All should be anthropic type with latest version by default
    for skill in skills:
        assert skill["type"] == "anthropic"
        assert skill["version"] == "latest"


def test_local_skill_config_initialization() -> None:
    """Test LocalSkillConfig initialization and validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "my-skill"
        skill_dir.mkdir()

        # Create a valid SKILL.md file
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: My Custom Skill
description: A test skill for unit testing
---

# My Custom Skill

This is a test skill."""
        )

        # Test successful initialization
        skill = LocalSkillConfig(path=skill_dir, display_title="Test Skill")
        assert skill.path == skill_dir
        assert skill.display_title == "Test Skill"
        assert skill.type == "custom"
        assert skill.version == "latest"
        assert skill.auto_upload is True
        assert not skill.is_uploaded

        # Test with auto_upload=False
        skill = LocalSkillConfig(path=skill_dir, auto_upload=False)
        assert skill.auto_upload is False


def test_local_skill_config_missing_path() -> None:
    """Test LocalSkillConfig with non-existent path."""
    with pytest.raises(FileNotFoundError, match="Skill path does not exist"):
        LocalSkillConfig(path="/nonexistent/path")


def test_local_skill_config_invalid_path_type() -> None:
    """Test LocalSkillConfig with invalid path type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a regular file instead of a directory
        invalid_path = Path(temp_dir) / "invalid.txt"
        invalid_path.write_text("not a skill")

        with pytest.raises(
            ValueError,
            match=r"Skill path must be a directory or \.zip file",
        ):
            LocalSkillConfig(path=invalid_path)


def test_local_skill_config_missing_skill_md() -> None:
    """Test LocalSkillConfig with directory missing SKILL.md."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "incomplete-skill"
        skill_dir.mkdir()

        with pytest.raises(
            ValueError,
            match=r"Skill directory must contain a SKILL\.md file",
        ):
            LocalSkillConfig(path=skill_dir)


def test_local_skill_config_zip_file() -> None:
    """Test LocalSkillConfig with zip file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a zip file
        import zipfile

        zip_path = Path(temp_dir) / "skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "SKILL.md",
                """---
name: Zipped Skill
description: A skill from a zip file
---

# Zipped Skill""",
            )

        # Should initialize successfully
        skill = LocalSkillConfig(path=zip_path)
        assert skill.path == zip_path
        assert skill.type == "custom"


def test_local_skill_config_content_hash() -> None:
    """Test that content hash is computed correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "my-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: Hash Test
description: Testing hash computation
---

# Hash Test"""
        )

        skill = LocalSkillConfig(path=skill_dir)
        hash1 = skill._compute_content_hash()

        # Hash should be consistent
        hash2 = skill._compute_content_hash()
        assert hash1 == hash2

        # Modify content
        skill_md.write_text(
            """---
name: Hash Test Modified
description: Modified content
---

# Modified"""
        )

        # Hash should change
        hash3 = skill._compute_content_hash()
        assert hash1 != hash3


def test_local_skill_config_create_zip() -> None:
    """Test creating zip from skill directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "my-skill"
        skill_dir.mkdir()

        # Create multiple files
        (skill_dir / "SKILL.md").write_text("# Test Skill")
        (skill_dir / "helper.py").write_text("def helper(): pass")

        # Create subdirectory with file
        subdir = skill_dir / "scripts"
        subdir.mkdir()
        (subdir / "script.py").write_text("print('hello')")

        skill = LocalSkillConfig(path=skill_dir)
        zip_bytes = skill._create_zip_bytes()

        # Verify it's valid zip
        import zipfile
        from io import BytesIO

        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            # Zip should contain top-level folder with all files inside
            assert "my-skill/SKILL.md" in names
            assert "my-skill/helper.py" in names
            assert "my-skill/scripts/script.py" in names


def test_local_skill_config_upload() -> None:
    """Test uploading a local skill."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")

        skill = LocalSkillConfig(path=skill_dir)

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "skill-123"
        mock_client.beta.skills.create.return_value = mock_response

        # Upload
        skill_id = skill.upload_skill(mock_client)

        assert skill_id == "skill-123"
        assert skill.skill_id == "skill-123"
        assert skill.is_uploaded

        # Verify upload was called
        mock_client.beta.skills.create.assert_called_once()

        # Upload again with same content - should not call API again
        skill_id2 = skill.upload_skill(mock_client)
        assert skill_id2 == "skill-123"
        assert mock_client.beta.skills.create.call_count == 1  # Still just 1 call


def test_claude_skills_middleware_with_local_skill() -> None:
    """Test ClaudeSkillsMiddleware with LocalSkillConfig."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill")

        # Create middleware with local skill (auto_upload=False)
        local_skill = LocalSkillConfig(path=skill_dir, auto_upload=False)
        middleware = ClaudeSkillsMiddleware(skills=["pptx", local_skill])

        # Mock chat model with _client
        mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
        mock_client = MagicMock()
        mock_chat_anthropic._client = mock_client

        # Manually upload the skill first
        mock_response = MagicMock()
        mock_response.id = "local-skill-456"
        mock_client.beta.skills.create.return_value = mock_response
        local_skill.upload_skill(mock_client)

        fake_request = ModelRequest(
            model=mock_chat_anthropic,
            messages=[HumanMessage("Use skills")],
            system_prompt=None,
            tool_choice=None,
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            response_format=None,
            state={"messages": [HumanMessage("Use skills")]},
            runtime=cast(Runtime, object()),
            model_settings={},
        )

        def mock_handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[AIMessage(content="mock response")])

        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)

        # Verify both skills are configured
        skills = fake_request.model_settings["container"]["skills"]
        assert len(skills) == 2
        skill_ids = {skill["skill_id"] for skill in skills}
        assert "pptx" in skill_ids
        assert "local-skill-456" in skill_ids


def test_claude_skills_middleware_auto_upload() -> None:
    """Test that middleware auto-uploads local skills when auto_upload=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "auto-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Auto Upload Test")

        # Create middleware with auto_upload=True (default)
        local_skill = LocalSkillConfig(path=skill_dir)
        middleware = ClaudeSkillsMiddleware(skills=[local_skill])

        # Mock chat model with _client
        mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
        mock_client = MagicMock()
        mock_chat_anthropic._client = mock_client

        mock_response = MagicMock()
        mock_response.id = "auto-uploaded-789"
        mock_client.beta.skills.create.return_value = mock_response

        fake_request = ModelRequest(
            model=mock_chat_anthropic,
            messages=[HumanMessage("Test")],
            system_prompt=None,
            tool_choice=None,
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            response_format=None,
            state={"messages": [HumanMessage("Test")]},
            runtime=cast(Runtime, object()),
            model_settings={},
        )

        def mock_handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[AIMessage(content="mock response")])

        # Should auto-upload during wrap_model_call
        assert not local_skill.is_uploaded
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert local_skill.is_uploaded
        assert local_skill.skill_id == "auto-uploaded-789"

        # Verify upload was called
        mock_client.beta.skills.create.assert_called_once()
