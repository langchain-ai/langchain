"""Testing that declaring custom tools using pydantic v2 works."""

from langchain import _PYDANTIC_MAJOR_VERSION
from langchain.tools.base import tool
import pytest

if _PYDANTIC_MAJOR_VERSION != 2:
    pytest.skip(
        "Unit tests for testing compatibility with pydantic major version 2",
        allow_module_level=True,
    )


def test_custom_tool_pydantic_v2() -> None:
    """Test that custom tools can be declared using pydantic v2."""

    @tool()
    def speak(what: str) -> str:
        """Return what was said backwards."""
        return what[::-1]

    assert speak("hello") == "olleh"
