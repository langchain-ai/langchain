r"""Test response_format for langchain-openai.

If tests fail, cassettes may need to be re-recorded.

To re-record cassettes:

1. Delete existing cassettes (`rm tests/cassettes/test_inference_to_*.yaml.gz`)
2. Re run the tests with a valid OPENAI_API_KEY in your environment:
```bash
OPENAI_API_KEY=... uv run python -m pytest tests/unit_tests/agents/test_response_format_integration.py
```

The cassettes are compressed. To read them:
```bash
gunzip -c "tests/cassettes/test_inference_to_native_output[True].yaml.gz" | \
    yq -o json . | \
    jq '.requests[].body |= (gsub("\n";"") | @base64d | fromjson) |
        .responses[].body.string |= (gsub("\n";"") | @base64d | fromjson)'
```

Or, in  Python:
```python
import json

from langchain_tests.conftest import CustomPersister, CustomSerializer

def bytes_encoder(obj):
    return obj.decode("utf-8", errors="replace")

path = "tests/cassettes/test_inference_to_native_output[True].yaml.gz"

requests, responses = CustomPersister().load_cassette(path, CustomSerializer())
assert len(requests) == len(responses)
for request, response in list(zip(requests, responses)):
    print("------ REQUEST ------")
    req = request._to_dict()
    req["body"] = json.loads(req["body"])
    print(json.dumps(req, indent=2, default=bytes_encoder))
    print("\n\n ------ RESPONSE ------")
    resp = response
    print(json.dumps(resp, indent=2, default=bytes_encoder))
print("\n\n")
```
"""  # noqa: E501

import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy


class WeatherBaseModel(BaseModel):
    """Weather response."""

    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75°F."


@pytest.mark.requires("langchain_openai")
@pytest.mark.vcr
@pytest.mark.parametrize("use_responses_api", [False, True])
def test_inference_to_native_output(*, use_responses_api: bool) -> None:
    """Test that native output is inferred when a model supports it."""
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": "gpt-5", "use_responses_api": use_responses_api}

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
def test_inference_to_tool_output(*, use_responses_api: bool) -> None:
    """Test that tool output is inferred when a model supports it."""
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": "gpt-5", "use_responses_api": use_responses_api}

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


@pytest.mark.requires("langchain_openai")
@pytest.mark.vcr
@pytest.mark.parametrize("use_responses_api", [False, True])
def test_strict_mode(*, use_responses_api: bool) -> None:
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": "gpt-5", "use_responses_api": use_responses_api}

    if "OPENAI_API_KEY" not in os.environ:
        model_kwargs["api_key"] = "foo"

    model = ChatOpenAI(**model_kwargs)

    # spy on _get_request_payload to check that `strict` is enabled
    original_method = model._get_request_payload
    payloads = []

    def capture_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_method(*args, **kwargs)
        payloads.append(result)
        return result

    with patch.object(model, "_get_request_payload", side_effect=capture_payload):
        agent = create_agent(
            model,
            tools=[get_weather],
            response_format=ProviderStrategy(WeatherBaseModel, strict=True),
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather in Boston?")]})

        assert len(payloads) == 2
        if use_responses_api:
            assert payloads[-1]["text"]["format"]["strict"]
        else:
            assert payloads[-1]["response_format"]["json_schema"]["strict"]

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
