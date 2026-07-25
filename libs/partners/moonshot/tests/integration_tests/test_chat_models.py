"""Integration tests for `ChatMoonshot`.

These tests require a real MOONSHOT_API_KEY to be set in the environment.
They can also be run by pointing to an OpenAI-compatible endpoint (e.g. DeepSeek)
by setting MOONSHOT_API_BASE to https://api.deepseek.com/v1 and MOONSHOT_API_KEY
to a valid DeepSeek API key.
"""

from __future__ import annotations

import os

import pytest

moonshot_api_key = os.environ.get("MOONSHOT_API_KEY")
moonshot_api_base = os.environ.get("MOONSHOT_API_BASE", "")

# Detect backend and use appropriate model name
if "deepseek" in moonshot_api_base.lower():
    MODEL_NAME = "deepseek-v4-flash"
else:
    MODEL_NAME = "moonshot-v1-8k"


@pytest.mark.skipif(not moonshot_api_key, reason="MOONSHOT_API_KEY not set")
def test_invoke() -> None:
    """Test basic invocation."""
    from langchain_moonshot import ChatMoonshot

    model = ChatMoonshot(model=MODEL_NAME)
    response = model.invoke("Say 'hello' in one word")
    assert response.content


@pytest.mark.skipif(not moonshot_api_key, reason="MOONSHOT_API_KEY not set")
def test_stream() -> None:
    """Test streaming."""
    from langchain_moonshot import ChatMoonshot

    model = ChatMoonshot(model=MODEL_NAME)
    chunks = list(model.stream("Count 1 2 3"))
    assert len(chunks) > 1


@pytest.mark.skipif(not moonshot_api_key, reason="MOONSHOT_API_KEY not set")
def test_tool_calling() -> None:
    """Test tool calling."""
    from pydantic import BaseModel, Field

    from langchain_moonshot import ChatMoonshot

    class GetWeather(BaseModel):
        """Get the current weather in a given location"""
        location: str = Field(description="City name")

    model = ChatMoonshot(model=MODEL_NAME)
    model_with_tools = model.bind_tools([GetWeather])
    response = model_with_tools.invoke("What is the weather in Shanghai?")
    assert response.tool_calls or response.content


@pytest.mark.skipif(not moonshot_api_key, reason="MOONSHOT_API_KEY not set")
def test_structured_output() -> None:
    """Test structured output."""
    from pydantic import BaseModel, Field

    from langchain_moonshot import ChatMoonshot

    class Person(BaseModel):
        """Personal information"""
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    model = ChatMoonshot(model=MODEL_NAME)
    structured = model.with_structured_output(Person)
    result = structured.invoke("I am Alice, 25 years old")
    assert result["name"] == "Alice" or result.name == "Alice"
