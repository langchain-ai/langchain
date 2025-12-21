"""Unit tests for OpenAI builtin tools conversion utilities."""

from langchain_openai.utils.builtin_tools import convert_standard_to_openai


def test_convert_web_search() -> None:
    """Test converting web_search tool."""
    tool = {"type": "web_search"}
    result = convert_standard_to_openai(tool)
    assert result == {"type": "web_search"}


def test_convert_web_search_with_user_location() -> None:
    """Test converting web_search with user_location."""
    tool = {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "country": "US",
        },
    }
    result = convert_standard_to_openai(tool)
    assert result == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "country": "US",
        },
    }


def test_convert_code_execution() -> None:
    """Test converting code_execution tool."""
    tool = {"type": "code_execution"}
    result = convert_standard_to_openai(tool)
    assert result == {"type": "code_interpreter", "container": {"type": "auto"}}


def test_convert_code_execution_with_container() -> None:
    """Test converting code_execution with custom container."""
    tool = {"type": "code_execution", "container": {"type": "custom", "id": "abc"}}
    result = convert_standard_to_openai(tool)
    assert result == {
        "type": "code_interpreter",
        "container": {"type": "custom", "id": "abc"},
    }


def test_convert_file_search() -> None:
    """Test converting file_search tool."""
    tool = {"type": "file_search"}
    result = convert_standard_to_openai(tool)
    assert result == {"type": "file_search"}


def test_convert_file_search_with_vector_stores() -> None:
    """Test converting file_search with vector_store_ids."""
    tool = {"type": "file_search", "vector_store_ids": ["vs_123", "vs_456"]}
    result = convert_standard_to_openai(tool)
    assert result == {"type": "file_search", "vector_store_ids": ["vs_123", "vs_456"]}


def test_convert_image_generation() -> None:
    """Test converting image_generation tool."""
    tool = {"type": "image_generation"}
    result = convert_standard_to_openai(tool)
    assert result == {"type": "image_generation"}


def test_convert_unsupported_tool() -> None:
    """Test converting tool not supported by OpenAI."""
    # web_fetch is Anthropic-only
    tool = {"type": "web_fetch"}
    result = convert_standard_to_openai(tool)
    assert result is None


def test_convert_memory_tool() -> None:
    """Test converting memory tool (Anthropic-only)."""
    tool = {"type": "memory"}
    result = convert_standard_to_openai(tool)
    assert result is None


def test_convert_text_editor_tool() -> None:
    """Test converting text_editor tool (Anthropic-only)."""
    tool = {"type": "text_editor"}
    result = convert_standard_to_openai(tool)
    assert result is None


def test_convert_bash_tool() -> None:
    """Test converting bash tool (Anthropic-only)."""
    tool = {"type": "bash"}
    result = convert_standard_to_openai(tool)
    assert result is None
