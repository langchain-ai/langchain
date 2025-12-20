"""Unit tests for Anthropic builtin tools conversion utilities."""

from langchain_anthropic.utils.builtin_tools import convert_standard_to_anthropic


def test_convert_web_search() -> None:
    """Test converting web_search tool."""
    tool = {"type": "web_search"}
    result = convert_standard_to_anthropic(tool)
    assert result == {"type": "web_search_20250305", "name": "web_search"}


def test_convert_web_search_with_max_uses() -> None:
    """Test converting web_search with max_uses."""
    tool = {"type": "web_search", "max_uses": 5}
    result = convert_standard_to_anthropic(tool)
    assert result == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
    }


def test_convert_web_search_with_user_location() -> None:
    """Test converting web_search with user_location."""
    tool = {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
        },
    }
    result = convert_standard_to_anthropic(tool)
    assert result == {
        "type": "web_search_20250305",
        "name": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
        },
    }


def test_convert_code_execution() -> None:
    """Test converting code_execution tool."""
    tool = {"type": "code_execution"}
    result = convert_standard_to_anthropic(tool)
    assert result == {"type": "code_execution_20250825", "name": "code_execution"}


def test_convert_web_fetch() -> None:
    """Test converting web_fetch tool."""
    tool = {"type": "web_fetch"}
    result = convert_standard_to_anthropic(tool)
    assert result == {"type": "web_fetch_20250910", "name": "web_fetch"}


def test_convert_web_fetch_with_max_uses() -> None:
    """Test converting web_fetch with max_uses."""
    tool = {"type": "web_fetch", "max_uses": 3}
    result = convert_standard_to_anthropic(tool)
    assert result == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 3,
    }


def test_convert_memory() -> None:
    """Test converting memory tool."""
    tool = {"type": "memory"}
    result = convert_standard_to_anthropic(tool)
    assert result == {"type": "memory_20250818", "name": "memory"}


def test_convert_text_editor() -> None:
    """Test converting text_editor tool."""
    tool = {"type": "text_editor"}
    result = convert_standard_to_anthropic(tool)
    assert result == {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }


def test_convert_bash() -> None:
    """Test converting bash tool."""
    tool = {"type": "bash"}
    result = convert_standard_to_anthropic(tool)
    assert result == {"type": "bash_20250124", "name": "bash"}


def test_convert_unsupported_tool() -> None:
    """Test converting tool not supported by Anthropic."""
    # file_search is OpenAI-only
    tool = {"type": "file_search"}
    result = convert_standard_to_anthropic(tool)
    assert result is None


def test_convert_image_generation() -> None:
    """Test converting image_generation tool (OpenAI-only)."""
    tool = {"type": "image_generation"}
    result = convert_standard_to_anthropic(tool)
    assert result is None


def test_convert_x_search() -> None:
    """Test converting x_search tool (xAI-only)."""
    tool = {"type": "x_search"}
    result = convert_standard_to_anthropic(tool)
    assert result is None

