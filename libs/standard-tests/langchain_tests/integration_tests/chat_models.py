from typing import Any, Literal, Optional

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_tests.unit_tests.chat_models import (
    ChatModelTests,
)


def _get_joke_class(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> Any:
    """
    :private:
    """

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    def validate_joke(result: Any) -> bool:
        return isinstance(result, Joke)

    class JokeDict(TypedDict):
        """Joke to tell user."""

        setup: Annotated[str, ..., "question to set up a joke"]
        punchline: Annotated[str, ..., "answer to resolve the joke"]

    def validate_joke_dict(result: Any) -> bool:
        return all(key in ["setup", "punchline"] for key in result.keys())

    if schema_type == "pydantic":
        return Joke, validate_joke

    elif schema_type == "typeddict":
        return JokeDict, validate_joke_dict

    elif schema_type == "json_schema":
        return Joke.model_json_schema(), validate_joke_dict
    else:
        raise ValueError("Invalid schema type")


class _TestCallbackHandler(BaseCallbackHandler):
    options: list[Optional[dict]]

    def __init__(self) -> None:
        super().__init__()
        self.options = []

    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.options.append(options)


class _MagicFunctionSchema(BaseModel):
    input: int = Field(..., gt=-1000, lt=1000)


@tool(args_schema=_MagicFunctionSchema)
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


@tool
def magic_function_no_args() -> int:
    """Calculates a magic function."""
    return 5


def _validate_tool_call_message(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function"
    assert tool_call["args"] == {"input": 3}
    assert tool_call["id"] is not None
    assert tool_call["type"] == "tool_call"


def _validate_tool_call_message_no_args(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function_no_args"
    assert tool_call["args"] == {}
    assert tool_call["id"] is not None
    assert tool_call["type"] == "tool_call"


class ChatModelIntegrationTests(ChatModelTests):
    """Base class for chat model integration tests.

    Test subclasses must implement the ``chat_model_class`` and
    ``chat_model_params`` properties to specify what model to test and its
    initialization parameters.

    Example:

    .. code-block:: python

        from typing import Type

        from langchain_tests.integration_tests import ChatModelIntegrationTests
        from my_package.chat_models import MyChatModel


        class TestMyChatModelIntegration(ChatModelIntegrationTests):
            @property
            def chat_model_class(self) -> Type[MyChatModel]:
                # Return the chat model class to test here
                return MyChatModel

            @property
            def chat_model_params(self) -> dict:
                # Return initialization parameters for the model.
                return {"model": "model-001", "temperature": 0}

    .. note::
          API references for individual test methods include troubleshooting tips.


    Test subclasses must implement the following two properties:

    chat_model_class
        The chat model class to test, e.g., ``ChatParrotLink``.

        Example:

        .. code-block:: python

            @property
            def chat_model_class(self) -> Type[ChatParrotLink]:
                return ChatParrotLink

    chat_model_params
        Initialization parameters for the chat model.

        Example:

        .. code-block:: python

            @property
            def chat_model_params(self) -> dict:
                return {"model": "bird-brain-001", "temperature": 0}

    In addition, test subclasses can control what features are tested (such as tool
    calling or multi-modality) by selectively overriding the following properties.
    Expand to see details:

    .. dropdown:: has_tool_calling

        Boolean property indicating whether the chat model supports tool calling.

        By default, this is determined by whether the chat model's `bind_tools` method
        is overridden. It typically does not need to be overridden on the test class.

        Example override:

        .. code-block:: python

            @property
            def has_tool_calling(self) -> bool:
                return True

    .. dropdown:: tool_choice_value

        Value to use for tool choice when used in tests.

        Some tests for tool calling features attempt to force tool calling via a
        `tool_choice` parameter. A common value for this parameter is "any". Defaults
        to `None`.

        Note: if the value is set to "tool_name", the name of the tool used in each
        test will be set as the value for `tool_choice`.

        Example:

        .. code-block:: python

            @property
            def tool_choice_value(self) -> Optional[str]:
                return "any"

    .. dropdown:: has_structured_output

        Boolean property indicating whether the chat model supports structured
        output.

        By default, this is determined by whether the chat model's
        ``with_structured_output`` method is overridden. If the base implementation is
        intended to be used, this method should be overridden.

        See: https://python.langchain.com/docs/concepts/structured_outputs/

        Example:

        .. code-block:: python

            @property
            def has_structured_output(self) -> bool:
                return True

    .. dropdown:: structured_output_kwargs

        Dict property that can be used to specify additional kwargs for
        ``with_structured_output``. Useful for testing different models.

        Example:

        .. code-block:: python

            @property
            def structured_output_kwargs(self) -> dict:
                return {"method": "function_calling"}

    .. dropdown:: supports_json_mode

        Boolean property indicating whether the chat model supports JSON mode in
        ``with_structured_output``.

        See: https://python.langchain.com/docs/concepts/structured_outputs/#json-mode

        Example:

        .. code-block:: python

            @property
            def supports_json_mode(self) -> bool:
                return True

    .. dropdown:: supports_image_inputs

        Boolean property indicating whether the chat model supports image inputs.
        Defaults to ``False``.

        If set to ``True``, the chat model will be tested using content blocks of the
        form

        .. code-block:: python

            [
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]

        See https://python.langchain.com/docs/concepts/multimodality/

        Example:

        .. code-block:: python

            @property
            def supports_image_inputs(self) -> bool:
                return True

    .. dropdown:: supports_video_inputs

        Boolean property indicating whether the chat model supports image inputs.
        Defaults to ``False``. No current tests are written for this feature.

    .. dropdown:: returns_usage_metadata

        Boolean property indicating whether the chat model returns usage metadata
        on invoke and streaming responses.

        ``usage_metadata`` is an optional dict attribute on AIMessages that track input
        and output tokens: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html

        Example:

        .. code-block:: python

            @property
            def returns_usage_metadata(self) -> bool:
                return False

    .. dropdown:: supports_anthropic_inputs

        Boolean property indicating whether the chat model supports Anthropic-style
        inputs.

        These inputs might feature "tool use" and "tool result" content blocks, e.g.,

        .. code-block:: python

            [
                {"type": "text", "text": "Hmm let me think about that"},
                {
                    "type": "tool_use",
                    "input": {"fav_color": "green"},
                    "id": "foo",
                    "name": "color_picker",
                },
            ]

        If set to ``True``, the chat model will be tested using content blocks of this
        form.

        Example:

        .. code-block:: python

            @property
            def supports_anthropic_inputs(self) -> bool:
                return False

    .. dropdown:: supports_image_tool_message

        Boolean property indicating whether the chat model supports ToolMessages
        that include image content, e.g.,

        .. code-block:: python

            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
                tool_call_id="1",
                name="random_image",
            )

        If set to ``True``, the chat model will be tested with message sequences that
        include ToolMessages of this form.

        Example:

        .. code-block:: python

            @property
            def supports_image_tool_message(self) -> bool:
                return False

    .. dropdown:: supported_usage_metadata_details

        Property controlling what usage metadata details are emitted in both invoke
        and stream.

        ``usage_metadata`` is an optional dict attribute on AIMessages that track input
        and output tokens: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html

        It includes optional keys ``input_token_details`` and ``output_token_details``
        that can track usage details associated with special types of tokens, such as
        cached, audio, or reasoning.

        Only needs to be overridden if these details are supplied.
    """

    @property
    def standard_chat_model_params(self) -> dict:
        """:private:"""
        return {}

    def test_agent_loop(self, model: BaseChatModel) -> None:
        """Test that the model supports a simple ReAct agent loop. This test is skipped
        if the ``has_tool_calling`` property on the test class is set to False.

        This test is optional and should be skipped if the model does not support
        tool calling (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that ``bind_tools`` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            Check also that all required information (e.g., tool calling identifiers)
            from AIMessage objects is propagated correctly to model payloads.

            This test may fail if the chat model does not consistently generate tool
            calls in response to an appropriate query. In these cases you can ``xfail``
            the test:

            .. code-block:: python

                @pytest.mark.xfail(reason=("Does not support tool_choice."))
                def test_agent_loop(self, model: BaseChatModel) -> None:
                    super().test_agent_loop(model)

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        @tool
        def get_weather(location: str) -> str:
            """Call to surf the web."""
            return "It's sunny."

        agent = create_react_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": "What is the weather in San Francisco, CA?"})
        messages = result["messages"]
        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage) and messages[1].tool_calls
        assert isinstance(messages[2], ToolMessage)
        assert isinstance(messages[3], AIMessage)
