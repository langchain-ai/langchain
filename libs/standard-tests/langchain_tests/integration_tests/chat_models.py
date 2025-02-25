import base64
import json
from typing import Any, List, Literal, Optional, cast
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
    tool_example_to_messages,
)
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from typing_extensions import Annotated, TypedDict

from langchain_tests.unit_tests.chat_models import (
    ChatModelTests,
)
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


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
    metadatas: list[Optional[dict]]

    def __init__(self) -> None:
        super().__init__()
        self.metadatas = []

    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.metadatas.append(metadata)


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

    def test_invoke(self, model: BaseChatModel) -> None:
        """Test to verify that `model.invoke(simple_message)` works.

        This should pass for all integrations.

        .. dropdown:: Troubleshooting

            If this test fails, you should make sure your _generate method
            does not raise any exceptions, and that it returns a valid
            :class:`~langchain_core.outputs.chat_result.ChatResult` like so:

            .. code-block:: python

                return ChatResult(
                    generations=[ChatGeneration(
                        message=AIMessage(content="Output text")
                    )]
                )
        """
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    async def test_ainvoke(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.ainvoke(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "natively async" implementation, but rather that the model can be used
        in an async context.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
            because `ainvoke` has a default implementation that calls `invoke` in an
            async context.

            If that test passes but not this one, you should make sure your _agenerate
            method does not raise any exceptions, and that it returns a valid
            :class:`~langchain_core.outputs.chat_result.ChatResult` like so:

            .. code-block:: python

                return ChatResult(
                    generations=[ChatGeneration(
                        message=AIMessage(content="Output text")
                    )]
                )
        """
        result = await model.ainvoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_stream(self, model: BaseChatModel) -> None:
        """Test to verify that `model.stream(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "streaming" implementation, but rather that the model can be used in a
        streaming context.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
            because `stream` has a default implementation that calls `invoke` and yields
            the result as a single chunk.

            If that test passes but not this one, you should make sure your _stream
            method does not raise any exceptions, and that it yields valid
            :class:`~langchain_core.outputs.chat_generation.ChatGenerationChunk`
            objects like so:

            .. code-block:: python

                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="chunk text")
                )
        """
        num_tokens = 0
        for token in model.stream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            num_tokens += len(token.content)
        assert num_tokens > 0

    async def test_astream(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.astream(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "natively async" or "streaming" implementation, but rather that the model can
        be used in an async streaming context.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_stream`.
            and
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`.
            because `astream` has a default implementation that calls `_stream` in an
            async context if it is implemented, or `ainvoke` and yields the result as a
            single chunk if not.

            If those tests pass but not this one, you should make sure your _astream
            method does not raise any exceptions, and that it yields valid
            :class:`~langchain_core.outputs.chat_generation.ChatGenerationChunk`
            objects like so:

            .. code-block:: python

                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="chunk text")
                )
        """
        num_tokens = 0
        async for token in model.astream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            num_tokens += len(token.content)
        assert num_tokens > 0

    def test_batch(self, model: BaseChatModel) -> None:
        """Test to verify that `model.batch([messages])` works.

        This should pass for all integrations. Tests the model's ability to process
        multiple prompts in a single batch.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because `batch` has a default implementation that calls `invoke` for each
            message in the batch.

            If that test passes but not this one, you should make sure your `batch`
            method does not raise any exceptions, and that it returns a list of valid
            :class:`~langchain_core.messages.AIMessage` objects.
        """
        batch_results = model.batch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    async def test_abatch(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.abatch([messages])` works.

        This should pass for all integrations. Tests the model's ability to process
        multiple prompts in a single batch asynchronously.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_batch`
            and
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`
            because `abatch` has a default implementation that calls `ainvoke` for each
            message in the batch.

            If those tests pass but not this one, you should make sure your `abatch`
            method does not raise any exceptions, and that it returns a list of valid
            :class:`~langchain_core.messages.AIMessage` objects.
        """
        batch_results = await model.abatch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    def test_conversation(self, model: BaseChatModel) -> None:
        """Test to verify that the model can handle multi-turn conversations.

        This should pass for all integrations. Tests the model's ability to process
        a sequence of alternating human and AI messages as context for generating
        the next response.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because this test also uses `model.invoke()`.

            If that test passes but not this one, you should verify that:
            1. Your model correctly processes the message history
            2. The model maintains appropriate context from previous messages
            3. The response is a valid :class:`~langchain_core.messages.AIMessage`
        """
        messages = [
            HumanMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]
        result = model.invoke(messages)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        """
        Test to verify that the model can handle double-message conversations.

        This should pass for all integrations. Tests the model's ability to process
        a sequence of double-system, double-human, and double-ai messages as context
        for generating the next response.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because this test also uses `model.invoke()`.

            Second, debug
            :meth:`~langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_conversation`
            because this test is the "basic case" without double messages.

            If that test passes those but not this one, you should verify that:
            1. Your model API can handle double messages, or the integration should
               merge messages before sending them to the API.
            2. The response is a valid :class:`~langchain_core.messages.AIMessage`
        """
        messages = [
            SystemMessage("hello"),
            SystemMessage("hello"),
            HumanMessage("hello"),
            HumanMessage("hello"),
            AIMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]
        result = model.invoke(messages)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_usage_metadata(self, model: BaseChatModel) -> None:
        """Test to verify that the model returns correct usage metadata.

        This test is optional and should be skipped if the model does not return
        usage metadata (see Configuration below).

        .. dropdown:: Configuration

            By default, this test is run.
            To disable this feature, set `returns_usage_metadata` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def returns_usage_metadata(self) -> bool:
                        return False

            This test can also check the format of specific kinds of usage metadata
            based on the `supported_usage_metadata_details` property. This property
            should be configured as follows with the types of tokens that the model
            supports tracking:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supported_usage_metadata_details(self) -> dict:
                        return {
                            "invoke": [
                                "audio_input",
                                "audio_output",
                                "reasoning_output",
                                "cache_read_input",
                                "cache_creation_input",
                            ],
                            "stream": [
                                "audio_input",
                                "audio_output",
                                "reasoning_output",
                                "cache_read_input",
                                "cache_creation_input",
                            ],
                        }


        .. dropdown:: Troubleshooting

            If this test fails, first verify that your model returns
            :class:`~langchain_core.messages.ai.UsageMetadata` dicts
            attached to the returned AIMessage object in `_generate`:

            .. code-block:: python

                return ChatResult(
                    generations=[ChatGeneration(
                        message=AIMessage(
                            content="Output text",
                            usage_metadata={
                                "input_tokens": 350,
                                "output_tokens": 240,
                                "total_tokens": 590,
                                "input_token_details": {
                                    "audio": 10,
                                    "cache_creation": 200,
                                    "cache_read": 100,
                                },
                                "output_token_details": {
                                    "audio": 10,
                                    "reasoning": 200,
                                }
                            }
                        )
                    )]
                )
        """
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert result.usage_metadata is not None
        assert isinstance(result.usage_metadata["input_tokens"], int)
        assert isinstance(result.usage_metadata["output_tokens"], int)
        assert isinstance(result.usage_metadata["total_tokens"], int)

        if "audio_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_audio_input()
            assert msg.usage_metadata is not None
            assert msg.usage_metadata["input_token_details"] is not None
            assert isinstance(msg.usage_metadata["input_token_details"]["audio"], int)
            assert msg.usage_metadata["input_tokens"] >= sum(
                (v or 0)  # type: ignore[misc]
                for v in msg.usage_metadata["input_token_details"].values()
            )
        if "audio_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_audio_output()
            assert msg.usage_metadata is not None
            assert msg.usage_metadata["output_token_details"] is not None
            assert isinstance(msg.usage_metadata["output_token_details"]["audio"], int)
            assert int(msg.usage_metadata["output_tokens"]) >= sum(
                (v or 0)  # type: ignore[misc]
                for v in msg.usage_metadata["output_token_details"].values()
            )
        if "reasoning_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_reasoning_output()
            assert msg.usage_metadata is not None
            assert msg.usage_metadata["output_token_details"] is not None
            assert isinstance(
                msg.usage_metadata["output_token_details"]["reasoning"],
                int,
            )
            assert msg.usage_metadata["output_tokens"] >= sum(
                (v or 0)  # type: ignore[misc]
                for v in msg.usage_metadata["output_token_details"].values()
            )
        if "cache_read_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_read_input()
            assert msg.usage_metadata is not None
            assert msg.usage_metadata["input_token_details"] is not None
            assert isinstance(
                msg.usage_metadata["input_token_details"]["cache_read"],
                int,
            )
            assert msg.usage_metadata["input_tokens"] >= sum(
                (v or 0)  # type: ignore[misc]
                for v in msg.usage_metadata["input_token_details"].values()
            )
        if "cache_creation_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_creation_input()
            assert msg.usage_metadata is not None
            assert msg.usage_metadata["input_token_details"] is not None
            assert isinstance(
                msg.usage_metadata["input_token_details"]["cache_creation"],
                int,
            )
            assert msg.usage_metadata["input_tokens"] >= sum(
                (v or 0)  # type: ignore[misc]
                for v in msg.usage_metadata["input_token_details"].values()
            )

    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        """
        Test to verify that the model returns correct usage metadata in streaming mode.

        .. dropdown:: Configuration

            By default, this test is run.
            To disable this feature, set `returns_usage_metadata` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def returns_usage_metadata(self) -> bool:
                        return False

            This test can also check the format of specific kinds of usage metadata
            based on the `supported_usage_metadata_details` property. This property
            should be configured as follows with the types of tokens that the model
            supports tracking:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supported_usage_metadata_details(self) -> dict:
                        return {
                            "invoke": [
                                "audio_input",
                                "audio_output",
                                "reasoning_output",
                                "cache_read_input",
                                "cache_creation_input",
                            ],
                            "stream": [
                                "audio_input",
                                "audio_output",
                                "reasoning_output",
                                "cache_read_input",
                                "cache_creation_input",
                            ],
                        }

        .. dropdown:: Troubleshooting

            If this test fails, first verify that your model yields
            :class:`~langchain_core.messages.ai.UsageMetadata` dicts
            attached to the returned AIMessage object in `_stream`
            that sum up to the total usage metadata.

            Note that `input_tokens` should only be included on one of the chunks
            (typically the first or the last chunk), and the rest should have 0 or None
            to avoid counting input tokens multiple times.

            `output_tokens` typically count the number of tokens in each chunk, not the
            sum. This test will pass as long as the sum of `output_tokens` across all
            chunks is not 0.

            .. code-block:: python

                yield ChatResult(
                    generations=[ChatGeneration(
                        message=AIMessage(
                            content="Output text",
                            usage_metadata={
                                "input_tokens": (
                                    num_input_tokens if is_first_chunk else 0
                                ),
                                "output_tokens": 11,
                                "total_tokens": (
                                    11+num_input_tokens if is_first_chunk else 11
                                ),
                                "input_token_details": {
                                    "audio": 10,
                                    "cache_creation": 200,
                                    "cache_read": 100,
                                },
                                "output_token_details": {
                                    "audio": 10,
                                    "reasoning": 200,
                                }
                            }
                        )
                    )]
                )
        """
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")
        full: Optional[AIMessageChunk] = None
        for chunk in model.stream("Write me 2 haikus. Only include the haikus."):
            assert isinstance(chunk, AIMessageChunk)
            # only one chunk is allowed to set usage_metadata.input_tokens
            # if multiple do, it's likely a bug that will result in overcounting
            # input tokens
            if full and full.usage_metadata and full.usage_metadata["input_tokens"]:
                assert (
                    not chunk.usage_metadata or not chunk.usage_metadata["input_tokens"]
                ), (
                    "Only one chunk should set input_tokens,"
                    " the rest should be 0 or None"
                )
            full = chunk if full is None else cast(AIMessageChunk, full + chunk)

        assert isinstance(full, AIMessageChunk)
        assert full.usage_metadata is not None
        assert isinstance(full.usage_metadata["input_tokens"], int)
        assert isinstance(full.usage_metadata["output_tokens"], int)
        assert isinstance(full.usage_metadata["total_tokens"], int)

        if "audio_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_audio_input(stream=True)
            assert isinstance(msg.usage_metadata["input_token_details"]["audio"], int)  # type: ignore[index]
        if "audio_output" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_audio_output(stream=True)
            assert isinstance(msg.usage_metadata["output_token_details"]["audio"], int)  # type: ignore[index]
        if "reasoning_output" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_reasoning_output(stream=True)
            assert isinstance(
                msg.usage_metadata["output_token_details"]["reasoning"],  # type: ignore[index]
                int,
            )
        if "cache_read_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_cache_read_input(stream=True)
            assert isinstance(
                msg.usage_metadata["input_token_details"]["cache_read"],  # type: ignore[index]
                int,
            )
        if "cache_creation_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_cache_creation_input(stream=True)
            assert isinstance(
                msg.usage_metadata["input_token_details"]["cache_creation"],  # type: ignore[index]
                int,
            )

    def test_stop_sequence(self, model: BaseChatModel) -> None:
        """Test that model does not fail when invoked with the ``stop`` parameter,
        which is a standard parameter for stopping generation at a certain token.

        More on standard parameters here: https://python.langchain.com/docs/concepts/chat_models/#standard-parameters

        This should pass for all integrations.

        .. dropdown:: Troubleshooting

            If this test fails, check that the function signature for ``_generate``
            (as well as ``_stream`` and async variants) accepts the ``stop`` parameter:

            .. code-block:: python

                def _generate(
                    self,
                    messages: List[BaseMessage],
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> ChatResult:
        """  # noqa: E501
        result = model.invoke("hi", stop=["you"])
        assert isinstance(result, AIMessage)

        custom_model = self.chat_model_class(
            **{
                **self.chat_model_params,
                "stop": ["you"],
            }
        )
        result = custom_model.invoke("hi")
        assert isinstance(result, AIMessage)

    def test_tool_calling(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls. This test is skipped if the
        ``has_tool_calling`` property on the test class is set to False.

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

            This test may fail if the chat model does not support a ``tool_choice``
            parameter. This parameter can be used to force a tool call. If
            ``tool_choice`` is not supported and the model consistently fails this
            test, you can ``xfail`` the test:

            .. code-block:: python

                @pytest.mark.xfail(reason=("Does not support tool_choice."))
                def test_tool_calling(self, model: BaseChatModel) -> None:
                    super().test_tool_calling(model)

            Otherwise, ensure that the ``tool_choice_value`` property is correctly
            specified on the test class.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "magic_function"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools([magic_function], tool_choice=tool_choice)

        # Test invoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message(result)

        # Test stream
        full: Optional[BaseMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls. This test is skipped if the
        ``has_tool_calling`` property on the test class is set to False.

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

            This test may fail if the chat model does not support a ``tool_choice``
            parameter. This parameter can be used to force a tool call. If
            ``tool_choice`` is not supported and the model consistently fails this
            test, you can ``xfail`` the test:

            .. code-block:: python

                @pytest.mark.xfail(reason=("Does not support tool_choice."))
                async def test_tool_calling_async(self, model: BaseChatModel) -> None:
                    await super().test_tool_calling_async(model)

            Otherwise, ensure that the ``tool_choice_value`` property is correctly
            specified on the test class.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "magic_function"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools([magic_function], tool_choice=tool_choice)

        # Test ainvoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = await model_with_tools.ainvoke(query)
        _validate_tool_call_message(result)

        # Test astream
        full: Optional[BaseMessageChunk] = None
        async for chunk in model_with_tools.astream(query):
            full = chunk if full is None else full + chunk  # type: ignore
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls for tools with no arguments.
        This test is skipped if the ``has_tool_calling`` property on the test class
        is set to False.

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
            chat model. It should correctly handle the case where a tool has no
            arguments.

            This test may fail if the chat model does not support a ``tool_choice``
            parameter. This parameter can be used to force a tool call. It may also
            fail if a provider does not support this form of tool. In these cases,
            you can ``xfail`` the test:

            .. code-block:: python

                @pytest.mark.xfail(reason=("Does not support tool_choice."))
                def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
                    super().test_tool_calling_with_no_arguments(model)

            Otherwise, ensure that the ``tool_choice_value`` property is correctly
            specified on the test class.
        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "magic_function_no_args"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools(
            [magic_function_no_args], tool_choice=tool_choice
        )
        query = "What is the value of magic_function_no_args()? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message_no_args(result)

        full: Optional[BaseMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore
        assert isinstance(full, AIMessage)
        _validate_tool_call_message_no_args(full)

    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls for tools that are derived from
        LangChain runnables. This test is skipped if the ``has_tool_calling`` property
        on the test class is set to False.

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

            This test may fail if the chat model does not support a ``tool_choice``
            parameter. This parameter can be used to force a tool call. If
            ``tool_choice`` is not supported and the model consistently fails this
            test, you can ``xfail`` the test:

            .. code-block:: python

                @pytest.mark.xfail(reason=("Does not support tool_choice."))
                def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
                    super().test_bind_runnables_as_tools(model)

            Otherwise, ensure that the ``tool_choice_value`` property is correctly
            specified on the test class.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        prompt = ChatPromptTemplate.from_messages(
            [("human", "Hello. Please respond in the style of {answer_style}.")]
        )
        llm = GenericFakeChatModel(messages=iter(["hello matey"]))
        chain = prompt | llm | StrOutputParser()
        tool_ = chain.as_tool(
            name="greeting_generator",
            description="Generate a greeting in a particular style of speaking.",
        )
        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "greeting_generator"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools([tool_], tool_choice=tool_choice)
        query = "Using the tool, generate a Pirate greeting."
        result = model_with_tools.invoke(query)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        tool_call = result.tool_calls[0]
        assert tool_call["args"].get("answer_style")
        assert tool_call["type"] == "tool_call"

    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        """Test to verify structured output is generated both on invoke and stream.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V2 models.
            ``langchain_core`` implements a utility function that will accommodate
            most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output
        """  # noqa: E501
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        schema, validation_function = _get_joke_class(schema_type)  # type: ignore[arg-type]
        chat = model.with_structured_output(schema, **self.structured_output_kwargs)
        mock_callback = MagicMock()
        mock_callback.on_chat_model_start = MagicMock()

        invoke_callback = _TestCallbackHandler()

        result = chat.invoke(
            "Tell me a joke about cats.", config={"callbacks": [invoke_callback]}
        )
        validation_function(result)

        assert len(invoke_callback.metadatas) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(invoke_callback.metadatas[0], dict)
        assert isinstance(
            invoke_callback.metadatas[0]["structured_output_format"]["schema"], dict
        )
        assert invoke_callback.metadatas[0]["structured_output_format"][
            "schema"
        ] == convert_to_openai_tool(schema)

        stream_callback = _TestCallbackHandler()

        for chunk in chat.stream(
            "Tell me a joke about cats.", config={"callbacks": [stream_callback]}
        ):
            validation_function(chunk)
        assert chunk

        assert len(stream_callback.metadatas) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(stream_callback.metadatas[0], dict)
        assert isinstance(
            stream_callback.metadatas[0]["structured_output_format"]["schema"], dict
        )
        assert stream_callback.metadatas[0]["structured_output_format"][
            "schema"
        ] == convert_to_openai_tool(schema)

    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> None:
        """Test to verify structured output is generated both on invoke and stream.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V2 models.
            ``langchain_core`` implements a utility function that will accommodate
            most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output
        """  # noqa: E501
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        schema, validation_function = _get_joke_class(schema_type)  # type: ignore[arg-type]

        chat = model.with_structured_output(schema, **self.structured_output_kwargs)
        ainvoke_callback = _TestCallbackHandler()

        result = await chat.ainvoke(
            "Tell me a joke about cats.", config={"callbacks": [ainvoke_callback]}
        )
        validation_function(result)

        assert len(ainvoke_callback.metadatas) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(ainvoke_callback.metadatas[0], dict)
        assert isinstance(
            ainvoke_callback.metadatas[0]["structured_output_format"]["schema"], dict
        )
        assert ainvoke_callback.metadatas[0]["structured_output_format"][
            "schema"
        ] == convert_to_openai_tool(schema)

        astream_callback = _TestCallbackHandler()

        async for chunk in chat.astream(
            "Tell me a joke about cats.", config={"callbacks": [astream_callback]}
        ):
            validation_function(chunk)
        assert chunk

        assert len(astream_callback.metadatas) == 1, (
            "Expected on_chat_model_start to be called once"
        )

        assert isinstance(astream_callback.metadatas[0], dict)
        assert isinstance(
            astream_callback.metadatas[0]["structured_output_format"]["schema"], dict
        )
        assert astream_callback.metadatas[0]["structured_output_format"][
            "schema"
        ] == convert_to_openai_tool(schema)

    @pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Test requires pydantic 2.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        """Test to verify we can generate structured output using
        pydantic.v1.BaseModel.

        pydantic.v1.BaseModel is available in the pydantic 2 package.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V1 models.
            ``langchain_core`` implements a utility function that will accommodate
            most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output
        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        class Joke(BaseModelV1):  # Uses langchain_core.pydantic_v1.BaseModel
            """Joke to tell user."""

            setup: str = FieldV1(description="question to set up a joke")
            punchline: str = FieldV1(description="answer to resolve the joke")

        # Pydantic class
        chat = model.with_structured_output(Joke, **self.structured_output_kwargs)
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, Joke)

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, Joke)

        # Schema
        chat = model.with_structured_output(
            Joke.schema(), **self.structured_output_kwargs
        )
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)  # for mypy
        assert set(chunk.keys()) == {"setup", "punchline"}

    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        """Test to verify we can generate structured output that includes optional
        parameters.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles Pydantic V2 models with optional parameters.
            ``langchain_core`` implements a utility function that will accommodate
            most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output
        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        # Pydantic
        class Joke(BaseModel):
            """Joke to tell user."""

            setup: str = Field(description="question to set up a joke")
            punchline: Optional[str] = Field(
                default=None, description="answer to resolve the joke"
            )

        chat = model.with_structured_output(Joke, **self.structured_output_kwargs)
        setup_result = chat.invoke(
            "Give me the setup to a joke about cats, no punchline."
        )
        assert isinstance(setup_result, Joke)

        joke_result = chat.invoke("Give me a joke about cats, include the punchline.")
        assert isinstance(joke_result, Joke)

        # Schema
        chat = model.with_structured_output(
            Joke.model_json_schema(), **self.structured_output_kwargs
        )
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)

        # TypedDict
        class JokeDict(TypedDict):
            """Joke to tell user."""

            setup: Annotated[str, ..., "question to set up a joke"]
            punchline: Annotated[Optional[str], None, "answer to resolve the joke"]

        chat = model.with_structured_output(JokeDict, **self.structured_output_kwargs)
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)

    def test_json_mode(self, model: BaseChatModel) -> None:
        """Test structured output via `JSON mode. <https://python.langchain.com/docs/concepts/structured_outputs/#json-mode>`_

        This test is optional and should be skipped if the model does not support
        the JSON mode feature (see Configuration below).

        .. dropdown:: Configuration

            To disable this test, set ``supports_json_mode`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supports_json_mode(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output
        """  # noqa: E501
        if not self.supports_json_mode:
            pytest.skip("Test requires json mode support.")

        from pydantic import BaseModel as BaseModelProper
        from pydantic import Field as FieldProper

        class Joke(BaseModelProper):
            """Joke to tell user."""

            setup: str = FieldProper(description="question to set up a joke")
            punchline: str = FieldProper(description="answer to resolve the joke")

        # Pydantic class
        # Type ignoring since the interface only officially supports pydantic 1
        # or pydantic.v1.BaseModel but not pydantic.BaseModel from pydantic 2.
        # We'll need to do a pass updating the type signatures.
        chat = model.with_structured_output(Joke, method="json_mode")  # type: ignore[arg-type]
        msg = (
            "Tell me a joke about cats. Return the result as a JSON with 'setup' and "
            "'punchline' keys. Return nothing other than JSON."
        )
        result = chat.invoke(msg)
        assert isinstance(result, Joke)

        for chunk in chat.stream(msg):
            assert isinstance(chunk, Joke)

        # Schema
        chat = model.with_structured_output(
            Joke.model_json_schema(), method="json_mode"
        )
        result = chat.invoke(msg)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in chat.stream(msg):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)  # for mypy
        assert set(chunk.keys()) == {"setup", "punchline"}

    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that message histories are compatible with string tool contents
        (e.g. OpenAI format). If a model passes this test, it should be compatible
        with messages generated from providers following OpenAI format.

        This test should be skipped if the model does not support tool calling
        (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model can correctly handle message histories that include AIMessage objects with ``""`` content.
            2. The ``tool_calls`` attribute on AIMessage objects is correctly handled and passed to the model in an appropriate format.
            3. The model can correctly handle ToolMessage objects with string content and arbitrary string values for ``tool_call_id``.

            You can ``xfail`` the test if tool calling is implemented but this format
            is not supported.

            .. code-block:: python

                @pytest.mark.xfail(reason=("Not implemented."))
                def test_tool_message_histories_string_content(self, *args: Any) -> None:
                    super().test_tool_message_histories_string_content(*args)
        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": "1", "b": "2"}

        messages_string_content = [
            HumanMessage("What is 1 + 2"),
            # string content (e.g. OpenAI)
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                json.dumps({"result": 3}),
                name=function_name,
                tool_call_id="abc123",
            ),
        ]
        result_string_content = model_with_tools.invoke(messages_string_content)
        assert isinstance(result_string_content, AIMessage)

    def test_tool_message_histories_list_content(
        self,
        model: BaseChatModel,
        my_adder_tool: BaseTool,
    ) -> None:
        """Test that message histories are compatible with list tool contents
        (e.g. Anthropic format).

        These message histories will include AIMessage objects with "tool use" and
        content blocks, e.g.,

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

        This test should be skipped if the model does not support tool calling
        (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model can correctly handle message histories that include AIMessage objects with list content.
            2. The ``tool_calls`` attribute on AIMessage objects is correctly handled and passed to the model in an appropriate format.
            3. The model can correctly handle ToolMessage objects with string content and arbitrary string values for ``tool_call_id``.

            You can ``xfail`` the test if tool calling is implemented but this format
            is not supported.

            .. code-block:: python

                @pytest.mark.xfail(reason=("Not implemented."))
                def test_tool_message_histories_list_content(self, *args: Any) -> None:
                    super().test_tool_message_histories_list_content(*args)
        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": 1, "b": 2}

        messages_list_content = [
            HumanMessage("What is 1 + 2"),
            # List content (e.g., Anthropic)
            AIMessage(
                [
                    {"type": "text", "text": "some text"},
                    {
                        "type": "tool_use",
                        "id": "abc123",
                        "name": function_name,
                        "input": function_args,
                    },
                ],
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                json.dumps({"result": 3}),
                name=function_name,
                tool_call_id="abc123",
            ),
        ]
        result_list_content = model_with_tools.invoke(messages_list_content)
        assert isinstance(result_list_content, AIMessage)

    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that the model can process few-shot examples with tool calls.

        These are represented as a sequence of messages of the following form:

        - ``HumanMessage`` with string content;
        - ``AIMessage`` with the ``tool_calls`` attribute populated;
        - ``ToolMessage`` with string content;
        - ``AIMessage`` with string content (an answer);
        - ``HuamnMessage`` with string content (a follow-up question).

        This test should be skipped if the model does not support tool calling
        (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            This test uses a utility function in ``langchain_core`` to generate a
            sequence of messages representing "few-shot" examples: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.tool_example_to_messages.html

            If this test fails, check that the model can correctly handle this
            sequence of messages.

            You can ``xfail`` the test if tool calling is implemented but this format
            is not supported.

            .. code-block:: python

                @pytest.mark.xfail(reason=("Not implemented."))
                def test_structured_few_shot_examples(self, *args: Any) -> None:
                    super().test_structured_few_shot_examples(*args)
        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool], tool_choice="any")
        function_result = json.dumps({"result": 3})

        tool_schema = my_adder_tool.args_schema
        assert tool_schema is not None
        few_shot_messages = tool_example_to_messages(
            "What is 1 + 2",
            [tool_schema(a=1, b=2)],
            tool_outputs=[function_result],
            ai_response=function_result,
        )

        messages = few_shot_messages + [HumanMessage("What is 3 + 4")]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process image inputs.

        This test should be skipped (see Configuration below) if the model does not
        support image inputs These will take the form of messages with OpenAI-style
        image content blocks:

        .. code-block:: python

            [
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]

        See https://python.langchain.com/docs/concepts/multimodality/

        .. dropdown:: Configuration

            To disable this test, set ``supports_image_inputs`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supports_image_inputs(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with image content blocks in OpenAI format, including base64-encoded
            images. Otherwise, set the ``supports_image_inputs`` property to False.
        """
        if not self.supports_image_inputs:
            return
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        model.invoke([message])

    def test_image_tool_message(self, model: BaseChatModel) -> None:
        """Test that the model can process ToolMessages with image inputs.

        This test should be skipped if the model does not support messages of the
        form:

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

        This test can be skipped by setting the ``supports_image_tool_message`` property
        to False (see Configuration below).

        .. dropdown:: Configuration

            To disable this test, set ``supports_image_tool_message`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supports_image_tool_message(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with image content blocks in ToolMessages, including base64-encoded
            images. Otherwise, set the ``supports_image_tool_message`` property to
            False.
        """
        if not self.supports_image_tool_message:
            return
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        messages = [
            HumanMessage("get a random image using the tool and describe the weather"),
            AIMessage(
                [],
                tool_calls=[
                    {"type": "tool_call", "id": "1", "name": "random_image", "args": {}}
                ],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
                tool_call_id="1",
                name="random_image",
            ),
        ]

        def random_image() -> str:
            """Return a random image."""
            return ""

        model.bind_tools([random_image]).invoke(messages)

    def test_anthropic_inputs(self, model: BaseChatModel) -> None:
        """Test that model can process Anthropic-style message histories.

        These message histories will include ``AIMessage`` objects with ``tool_use``
        content blocks, e.g.,

        .. code-block:: python

            AIMessage(
                [
                    {"type": "text", "text": "Hmm let me think about that"},
                    {
                        "type": "tool_use",
                        "input": {"fav_color": "green"},
                        "id": "foo",
                        "name": "color_picker",
                    },
                ]
            )

        as well as ``HumanMessage`` objects containing ``tool_result`` content blocks:

        .. code-block:: python

            HumanMessage(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "foo",
                        "content": [
                            {
                                "type": "text",
                                "text": "green is a great pick! that's my sister's favorite color",  # noqa: E501
                            }
                        ],
                        "is_error": False,
                    },
                    {"type": "text", "text": "what's my sister's favorite color"},
                ]
            )

        This test should be skipped if the model does not support messages of this
        form (or doesn't support tool calling generally). See Configuration below.

        .. dropdown:: Configuration

            To disable this test, set ``supports_anthropic_inputs`` to False in your
            test class:

            .. code-block:: python

                class TestMyChatModelIntegration(ChatModelIntegrationTests):
                    @property
                    def supports_anthropic_inputs(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model can correctly handle message histories that include message objects with list content.
            2. The ``tool_calls`` attribute on AIMessage objects is correctly handled and passed to the model in an appropriate format.
            3. HumanMessages with "tool_result" content blocks are correctly handled.

            Otherwise, if Anthropic tool call and result formats are not supported,
            set the ``supports_anthropic_inputs`` property to False.
        """  # noqa: E501
        if not self.supports_anthropic_inputs:
            return

        class color_picker(BaseModelV1):
            """Input your fav color and get a random fact about it."""

            fav_color: str

        human_content: List[dict] = [
            {
                "type": "text",
                "text": "what's your favorite color in this image",
            },
        ]
        if self.supports_image_inputs:
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            human_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                }
            )
        messages = [
            SystemMessage("you're a good assistant"),
            HumanMessage(human_content),  # type: ignore[arg-type]
            AIMessage(
                [
                    {"type": "text", "text": "Hmm let me think about that"},
                    {
                        "type": "tool_use",
                        "input": {"fav_color": "green"},
                        "id": "foo",
                        "name": "color_picker",
                    },
                ]
            ),
            HumanMessage(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "foo",
                        "content": [
                            {
                                "type": "text",
                                "text": "green is a great pick! that's my sister's favorite color",  # noqa: E501
                            }
                        ],
                        "is_error": False,
                    },
                    {"type": "text", "text": "what's my sister's favorite color"},
                ]
            ),
        ]
        model.bind_tools([color_picker]).invoke(messages)

    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that ToolMessage with ``status="error"`` can be handled.

        These messages may take the form:

        .. code-block:: python

            ToolMessage(
                "Error: Missing required argument 'b'.",
                name="my_adder_tool",
                tool_call_id="abc123",
                status="error",
            )

        If possible, the ``status`` field should be parsed and passed appropriately
        to the model.

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

            If this test fails, check that the ``status`` field on ``ToolMessage``
            objects is either ignored or passed to the model appropriately.

            Otherwise, ensure that the ``tool_choice_value`` property is correctly
            specified on the test class.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        messages = [
            HumanMessage("What is 1 + 2"),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "my_adder_tool",
                        "args": {"a": 1},
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                "Error: Missing required argument 'b'.",
                name="my_adder_tool",
                tool_call_id="abc123",
                status="error",
            ),
        ]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

    def test_message_with_name(self, model: BaseChatModel) -> None:
        """Test that HumanMessage with values for the ``name`` field can be handled.

        These messages may take the form:

        .. code-block:: python

            HumanMessage("hello", name="example_user")

        If possible, the ``name`` field should be parsed and passed appropriately
        to the model. Otherwise, it should be ignored.

        .. dropdown:: Troubleshooting

            If this test fails, check that the ``name`` field on ``HumanMessage``
            objects is either ignored or passed to the model appropriately.
        """
        result = model.invoke([HumanMessage("hello", name="example_user")])
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def invoke_with_audio_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        raise NotImplementedError()

    def invoke_with_audio_output(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        raise NotImplementedError()

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        raise NotImplementedError()

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        raise NotImplementedError()

    def invoke_with_cache_creation_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        raise NotImplementedError()
