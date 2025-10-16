"""Test suite for create_agent with structured output response_format permutations."""

import json

import pytest

from dataclasses import dataclass
from typing import Union, Sequence, Any, Callable
from collections.abc import Awaitable

from langchain_core.messages import HumanMessage, AIMessage as CoreAIMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from tests.unit_tests.agents.model import FakeToolCallingModel
from langchain.tools import BaseTool


# Test data models
class WeatherBaseModel(BaseModel):
    """Weather response."""

    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


@dataclass
class WeatherDataclass:
    """Weather response."""

    temperature: float
    condition: str


class WeatherTypedDict(TypedDict):
    """Weather response."""

    temperature: float
    condition: str


weather_json_schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "description": "Temperature in fahrenheit"},
        "condition": {"type": "string", "description": "Weather condition"},
    },
    "title": "weather_schema",
    "required": ["temperature", "condition"],
}


class LocationResponse(BaseModel):
    city: str = Field(description="The city name")
    country: str = Field(description="The country name")


class LocationTypedDict(TypedDict):
    city: str
    country: str


location_json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "The city name"},
        "country": {"type": "string", "description": "The country name"},
    },
    "title": "location_schema",
    "required": ["city", "country"],
}


@tool
def get_weather() -> str:
    """Get the weather."""

    return "The weather is sunny and 75°F."


@tool
def get_location() -> str:
    """Get the current location."""

    return "You are in New York, USA."


# Standardized test data
WEATHER_DATA = {"temperature": 75.0, "condition": "sunny"}
LOCATION_DATA = {"city": "New York", "country": "USA"}

# Standardized expected responses
EXPECTED_WEATHER_PYDANTIC = WeatherBaseModel(**WEATHER_DATA)
EXPECTED_WEATHER_DATACLASS = WeatherDataclass(**WEATHER_DATA)
EXPECTED_WEATHER_DICT: WeatherTypedDict = {"temperature": 75.0, "condition": "sunny"}
EXPECTED_LOCATION = LocationResponse(**LOCATION_DATA)
EXPECTED_LOCATION_DICT: LocationTypedDict = {"city": "New York", "country": "USA"}


class TestResponseFormatAsModel:
    def test_pydantic_model(self) -> None:
        """Test response_format as Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=WeatherBaseModel)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
        assert len(response["messages"]) == 5

    def test_dataclass(self) -> None:
        """Test response_format as dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherDataclass",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=WeatherDataclass)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DATACLASS
        assert len(response["messages"]) == 5

    def test_typed_dict(self) -> None:
        """Test response_format as TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherTypedDict",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=WeatherTypedDict)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 5

    def test_json_schema(self) -> None:
        """Test response_format as JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=weather_json_schema)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 5


class TestResponseFormatAsToolStrategy:
    def test_pydantic_model(self) -> None:
        """Test response_format as ToolStrategy with Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=ToolStrategy(WeatherBaseModel))
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
        assert len(response["messages"]) == 5

    def test_dataclass(self) -> None:
        """Test response_format as ToolStrategy with dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherDataclass",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=ToolStrategy(WeatherDataclass))
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DATACLASS
        assert len(response["messages"]) == 5

    def test_typed_dict(self) -> None:
        """Test response_format as ToolStrategy with TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherTypedDict",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(model, [get_weather], response_format=ToolStrategy(WeatherTypedDict))
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 5

    def test_json_schema(self) -> None:
        """Test response_format as ToolStrategy with JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model, [get_weather], response_format=ToolStrategy(weather_json_schema)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 5

    def test_union_of_json_schemas(self) -> None:
        """Test response_format as ToolStrategy with union of JSON schemas."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [get_weather, get_location],
            response_format=ToolStrategy({"oneOf": [weather_json_schema, location_json_schema]}),
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 5

        # Test with LocationResponse
        tool_calls_location = [
            [{"args": {}, "id": "1", "name": "get_location"}],
            [
                {
                    "name": "location_schema",
                    "id": "2",
                    "args": LOCATION_DATA,
                }
            ],
        ]

        model_location = FakeToolCallingModel(tool_calls=tool_calls_location)

        agent_location = create_agent(
            model_location,
            [get_weather, get_location],
            response_format=ToolStrategy({"oneOf": [weather_json_schema, location_json_schema]}),
        )
        response_location = agent_location.invoke({"messages": [HumanMessage("Where am I?")]})

        assert response_location["structured_response"] == EXPECTED_LOCATION_DICT
        assert len(response_location["messages"]) == 5

    def test_union_of_types(self) -> None:
        """Test response_format as ToolStrategy with Union of various types."""
        # Test with WeatherBaseModel
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": WEATHER_DATA,
                }
            ],
        ]

        model = FakeToolCallingModel[Union[WeatherBaseModel, LocationResponse]](
            tool_calls=tool_calls
        )

        agent = create_agent(
            model,
            [get_weather, get_location],
            response_format=ToolStrategy(Union[WeatherBaseModel, LocationResponse]),
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
        assert len(response["messages"]) == 5

        # Test with LocationResponse
        tool_calls_location = [
            [{"args": {}, "id": "1", "name": "get_location"}],
            [
                {
                    "name": "LocationResponse",
                    "id": "2",
                    "args": LOCATION_DATA,
                }
            ],
        ]

        model_location = FakeToolCallingModel(tool_calls=tool_calls_location)

        agent_location = create_agent(
            model_location,
            [get_weather, get_location],
            response_format=ToolStrategy(Union[WeatherBaseModel, LocationResponse]),
        )
        response_location = agent_location.invoke({"messages": [HumanMessage("Where am I?")]})

        assert response_location["structured_response"] == EXPECTED_LOCATION
        assert len(response_location["messages"]) == 5

    def test_multiple_structured_outputs_error_without_retry(self) -> None:
        """Test that MultipleStructuredOutputsError is raised when model returns multiple structured tool calls without retry."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": WEATHER_DATA,
                },
                {
                    "name": "LocationResponse",
                    "id": "2",
                    "args": LOCATION_DATA,
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                Union[WeatherBaseModel, LocationResponse],
                handle_errors=False,
            ),
        )

        with pytest.raises(
            MultipleStructuredOutputsError,
            match=".*WeatherBaseModel.*LocationResponse.*",
        ):
            agent.invoke({"messages": [HumanMessage("Give me weather and location")]})

    def test_multiple_structured_outputs_with_retry(self) -> None:
        """Test that retry handles multiple structured output tool calls."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": WEATHER_DATA,
                },
                {
                    "name": "LocationResponse",
                    "id": "2",
                    "args": LOCATION_DATA,
                },
            ],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "3",
                    "args": WEATHER_DATA,
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                Union[WeatherBaseModel, LocationResponse],
                handle_errors=True,
            ),
        )

        response = agent.invoke({"messages": [HumanMessage("Give me weather")]})

        # HumanMessage, AIMessage, ToolMessage, ToolMessage, AI, ToolMessage
        assert len(response["messages"]) == 6
        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC

    def test_structured_output_parsing_error_without_retry(self) -> None:
        """Test that StructuredOutputParsingError is raised when tool args fail to parse without retry."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": {"invalid": "data"},
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                WeatherBaseModel,
                handle_errors=False,
            ),
        )

        with pytest.raises(
            StructuredOutputValidationError,
            match=".*WeatherBaseModel.*",
        ):
            agent.invoke({"messages": [HumanMessage("What's the weather?")]})

    def test_structured_output_parsing_error_with_retry(self) -> None:
        """Test that retry handles parsing errors for structured output."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": {"invalid": "data"},
                },
            ],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": WEATHER_DATA,
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                WeatherBaseModel,
                handle_errors=(StructuredOutputValidationError,),
            ),
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage
        assert len(response["messages"]) == 5
        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC

    def test_retry_with_custom_function(self) -> None:
        """Test retry with custom message generation."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": WEATHER_DATA,
                },
                {
                    "name": "LocationResponse",
                    "id": "2",
                    "args": LOCATION_DATA,
                },
            ],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "3",
                    "args": WEATHER_DATA,
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        def custom_message(exception: Exception) -> str:
            if isinstance(exception, MultipleStructuredOutputsError):
                return "Custom error: Multiple outputs not allowed"
            return "Custom error"

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                Union[WeatherBaseModel, LocationResponse],
                handle_errors=custom_message,
            ),
        )

        response = agent.invoke({"messages": [HumanMessage("Give me weather")]})

        # HumanMessage, AIMessage, ToolMessage, ToolMessage, AI, ToolMessage
        assert len(response["messages"]) == 6
        assert response["messages"][2].content == "Custom error: Multiple outputs not allowed"
        assert response["messages"][3].content == "Custom error: Multiple outputs not allowed"
        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC

    def test_retry_with_custom_string_message(self) -> None:
        """Test retry with custom static string message."""
        tool_calls = [
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "1",
                    "args": {"invalid": "data"},
                },
            ],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": WEATHER_DATA,
                },
            ],
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        agent = create_agent(
            model,
            [],
            response_format=ToolStrategy(
                WeatherBaseModel,
                handle_errors="Please provide valid weather data with temperature and condition.",
            ),
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert len(response["messages"]) == 5
        assert (
            response["messages"][2].content
            == "Please provide valid weather data with temperature and condition."
        )
        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC


class TestResponseFormatAsProviderStrategy:
    def test_pydantic_model(self) -> None:
        """Test response_format as ProviderStrategy with Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
        ]

        model = FakeToolCallingModel[WeatherBaseModel](
            tool_calls=tool_calls, structured_response=EXPECTED_WEATHER_PYDANTIC, model_name="gpt-4.1"
        )

        agent = create_agent(
            model, [get_weather], response_format=ProviderStrategy(WeatherBaseModel)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
        assert len(response["messages"]) == 4

    def test_dataclass(self) -> None:
        """Test response_format as ProviderStrategy with dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
        ]

        model = FakeToolCallingModel[WeatherDataclass](
            tool_calls=tool_calls, structured_response=EXPECTED_WEATHER_DATACLASS, model_name="gpt-4.1"
        )

        agent = create_agent(
            model, [get_weather], response_format=ProviderStrategy(WeatherDataclass)
        )
        response = agent.invoke(
            {"messages": [HumanMessage("What's the weather?")]},
        )

        assert response["structured_response"] == EXPECTED_WEATHER_DATACLASS
        assert len(response["messages"]) == 4

    def test_typed_dict(self) -> None:
        """Test response_format as ProviderStrategy with TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
        ]

        model = FakeToolCallingModel[WeatherTypedDict](
            tool_calls=tool_calls, structured_response=EXPECTED_WEATHER_DICT, model_name="gpt-4.1"
        )

        agent = create_agent(
            model, [get_weather], response_format=ProviderStrategy(WeatherTypedDict)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 4

    def test_json_schema(self) -> None:
        """Test response_format as ProviderStrategy with JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
        ]

        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=EXPECTED_WEATHER_DICT, model_name="gpt-4.1"
        )

        agent = create_agent(
            model, [get_weather], response_format=ProviderStrategy(weather_json_schema)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == EXPECTED_WEATHER_DICT
        assert len(response["messages"]) == 4


class TestDynamicModelWithResponseFormat:
    """Test response_format with middleware that modifies the model."""

    def test_middleware_model_swap_provider_to_tool_strategy(self) -> None:
        """Test that strategy resolution is deferred until after middleware modifies the model.

        Verifies that when a raw schema is provided, `_supports_provider_strategy` is called
        on the middleware-modified model (not the original), ensuring the correct strategy is
        selected based on the final model's capabilities.
        """
        from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        # Custom model that we'll use to test whether the provider strategy is applied
        # correctly at runtime. Use a model_name that supports provider strategy.
        class CustomModel(GenericFakeChatModel):
            model_name: str = "gpt-4.1"
            tool_bindings: list[Any] = []

            def bind_tools(
                self,
                tools: Sequence[Union[dict[str, Any], type[BaseModel], Callable, BaseTool]],
                **kwargs: Any,
            ) -> Runnable[LanguageModelInput, BaseMessage]:
                # Record every tool binding event.
                self.tool_bindings.append(tools)
                return self

        model = CustomModel(
            messages=iter(
                [
                    # Simulate model returning structured output directly
                    # (this is what provider strategy would do)
                    json.dumps(WEATHER_DATA),
                ]
            )
        )

        # Create middleware that swaps the model in the request
        class ModelSwappingMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], CoreAIMessage],
            ) -> CoreAIMessage:
                # Replace the model with our custom test model
                request.model = model
                return handler(request)

        # Use raw Pydantic model (not wrapped in ToolStrategy or ProviderStrategy)
        # This should auto-detect strategy based on model capabilities
        agent = create_agent(
            model=model,
            tools=[],
            # Raw schema - should auto-detect strategy
            response_format=WeatherBaseModel,
            middleware=[ModelSwappingMiddleware()],
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # Verify successful parsing of JSON as structured output via ProviderStrategy
        assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
        # Two messages: Human input message and AI response with JSON content
        assert len(response["messages"]) == 2
        ai_message = response["messages"][1]
        assert isinstance(ai_message, AIMessage)
        # ProviderStrategy doesn't use tool calls - it parses content directly
        assert ai_message.tool_calls == []
        assert ai_message.content == json.dumps(WEATHER_DATA)


def test_union_of_types() -> None:
    """Test response_format as ProviderStrategy with Union (if supported)."""
    tool_calls = [
        [{"args": {}, "id": "1", "name": "get_weather"}],
        [
            {
                "name": "WeatherBaseModel",
                "id": "2",
                "args": WEATHER_DATA,
            }
        ],
    ]

    model = FakeToolCallingModel[Union[WeatherBaseModel, LocationResponse]](
        tool_calls=tool_calls, structured_response=EXPECTED_WEATHER_PYDANTIC
    )

    agent = create_agent(
        model,
        [get_weather, get_location],
        response_format=ToolStrategy(Union[WeatherBaseModel, LocationResponse]),
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

    assert response["structured_response"] == EXPECTED_WEATHER_PYDANTIC
    assert len(response["messages"]) == 5


def test_provider_strategy_strict_only_for_openai() -> None:
    """Test that strict=True is set for OpenAI and Grok models in ProviderStrategy."""
    from langchain.agents.structured_output import ProviderStrategy

    # Create a mock OpenAI model with model_name
    class MockOpenAIModel:
        model_name: str = "gpt-4.1"

    # Create a mock Grok/X.AI model with model_name
    class MockGrokModel:
        model_name: str = "grok-beta"

    provider_strategy = ProviderStrategy(WeatherBaseModel)

    # Test OpenAI model: should include strict=True
    openai_model = MockOpenAIModel()
    openai_kwargs = provider_strategy.to_model_kwargs(model=openai_model)
    assert "strict" in openai_kwargs
    assert openai_kwargs["strict"] is True
    assert "response_format" in openai_kwargs

    # Test Grok model: should include strict=True (Grok requires strict)
    grok_model = MockGrokModel()
    grok_kwargs = provider_strategy.to_model_kwargs(model=grok_model)
    assert "strict" in grok_kwargs
    assert grok_kwargs["strict"] is True
    assert "response_format" in grok_kwargs


def test_provider_strategy_validation() -> None:
    """Test that ProviderStrategy validates provider support at agent invocation time."""
    from langchain.agents.structured_output import ProviderStrategy

    # Create a mock model from an unsupported provider (e.g., Anthropic)
    # Use a model_name that doesn't match any supported patterns
    class MockAnthropicModel(FakeToolCallingModel):
        model_name: str = "claude-3-5-sonnet-20241022"

    # Test unsupported provider: should raise ValueError when invoking agent
    anthropic_model = MockAnthropicModel(tool_calls=[[]])
    agent = create_agent(anthropic_model, [], response_format=ProviderStrategy(WeatherBaseModel))
    with pytest.raises(ValueError, match="ProviderStrategy does not support this model"):
        agent.invoke({"messages": [HumanMessage("What's the weather?")]})
