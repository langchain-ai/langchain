import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class WeatherBaseModel(BaseModel):
    """Weather response."""

    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


def get_weather(city: str) -> str:  # noqa: ARG001
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75°F."


@pytest.mark.requires("langchain_openai")
@pytest.mark.vcr
@pytest.mark.parametrize("use_responses_api", [False, True])
def test_inference_to_native_output(use_responses_api: bool) -> None:
    """Test that native output is inferred when a model supports it."""
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": "gpt-4.1-mini", "use_responses_api": use_responses_api}

    if "OPENAI_API_KEY" not in os.environ:
        model_kwargs["api_key"] = "foo"

    model = ChatOpenAI(**model_kwargs)

    agent = create_agent(
        model,
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool "
            "once, then use the WeatherReport tool to generate the final response."
        ),
        tools=[get_weather],
        response_format=WeatherBaseModel,
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather in Boston?")]})

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 4

    assert [m.type for m in response["messages"]] == [
        "human",  # "What's the weather?"
        "ai",  # "What's the weather?"
        "tool",  # "The weather is sunny and 75°F."
        "ai",  # structured response
    ]


@pytest.mark.requires("langchain_openai")
@pytest.mark.vcr
@pytest.mark.parametrize("use_responses_api", [False, True])
def test_inference_to_tool_output(use_responses_api: bool) -> None:
    """Test that tool output is inferred when a model supports it."""
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": "gpt-4.1-mini", "use_responses_api": use_responses_api}

    if "OPENAI_API_KEY" not in os.environ:
        model_kwargs["api_key"] = "foo"

    model = ChatOpenAI(**model_kwargs)

    agent = create_agent(
        model,
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool "
            "once, then use the WeatherReport tool to generate the final response."
        ),
        tools=[get_weather],
        response_format=ToolStrategy(WeatherBaseModel),
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 5

    assert [m.type for m in response["messages"]] == [
        "human",  # "What's the weather?"
        "ai",  # "What's the weather?"
        "tool",  # "The weather is sunny and 75°F."
        "ai",  # structured response
        "tool",  # artificial tool message
    ]
