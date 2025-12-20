"""Unit tests for xAI builtin tools conversion utilities."""

from langchain_xai.utils.builtin_tools import convert_standard_to_xai


def test_convert_web_search() -> None:
    """Test converting web_search tool."""
    tool = {"type": "web_search"}
    result = convert_standard_to_xai(tool)
    assert result == {"type": "web_search"}


def test_convert_code_execution() -> None:
    """Test converting code_execution tool."""
    tool = {"type": "code_execution"}
    result = convert_standard_to_xai(tool)
    assert result == {"type": "code_interpreter"}


def test_convert_x_search() -> None:
    """Test converting x_search tool (unique to xAI)."""
    tool = {"type": "x_search"}
    result = convert_standard_to_xai(tool)
    assert result == {"type": "x_search"}


def test_convert_unsupported_tool() -> None:
    """Test converting tool not supported by xAI."""
    # web_fetch is Anthropic-only
    tool = {"type": "web_fetch"}
    result = convert_standard_to_xai(tool)
    assert result is None


def test_convert_memory() -> None:
    """Test converting memory tool (Anthropic-only)."""
    tool = {"type": "memory"}
    result = convert_standard_to_xai(tool)
    assert result is None


def test_convert_file_search() -> None:
    """Test converting file_search tool (OpenAI-only)."""
    tool = {"type": "file_search"}
    result = convert_standard_to_xai(tool)
    assert result is None


def test_convert_image_generation() -> None:
    """Test converting image_generation tool (OpenAI-only)."""
    tool = {"type": "image_generation"}
    result = convert_standard_to_xai(tool)
    assert result is None


def test_convert_text_editor() -> None:
    """Test converting text_editor tool (Anthropic-only)."""
    tool = {"type": "text_editor"}
    result = convert_standard_to_xai(tool)
    assert result is None


def test_convert_bash() -> None:
    """Test converting bash tool (Anthropic-only)."""
    tool = {"type": "bash"}
    result = convert_standard_to_xai(tool)
    assert result is None

