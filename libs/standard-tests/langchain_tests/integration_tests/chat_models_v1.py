"""Integration tests for v1 chat models.

This module provides comprehensive integration tests for the new messages and standard
content block system introduced in ``langchain_core.v1.messages`` and
``langchain_core.messages.content_blocks``.
"""

import base64
import json
from typing import Annotated, Any, Literal, Optional, TypedDict, Union, cast
from unittest.mock import MagicMock

import httpx
import langchain_core.messages.content_blocks as types
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.content_blocks import (
    AudioContentBlock,
    Citation,
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
    FileContentBlock,
    ImageContentBlock,
    InvalidToolCall,
    NonStandardContentBlock,
    PlainTextContentBlock,
    ReasoningContentBlock,
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    VideoContentBlock,
    WebSearchCall,
    WebSearchResult,
    create_audio_block,
    create_file_block,
    create_image_block,
    create_non_standard_block,
    create_plaintext_block,
    create_text_block,
    create_tool_call,
    create_video_block,
    is_reasoning_block,
    is_text_block,
    is_tool_call_block,
)
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    tool_example_to_messages,
)
from langchain_core.v1.chat_models import BaseChatModel
from langchain_core.v1.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]
from vcr.cassette import Cassette

from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1Tests
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION

# Content block type definitions for testing
ContentBlock = Union[
    TextContentBlock,
    ImageContentBlock,
    VideoContentBlock,
    AudioContentBlock,
    PlainTextContentBlock,
    FileContentBlock,
    ReasoningContentBlock,
    NonStandardContentBlock,
    ToolCall,
    InvalidToolCall,
    ToolCallChunk,
    WebSearchCall,
    WebSearchResult,
    Citation,
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
]


def _get_test_image_base64() -> str:
    """Get a small test image as base64 for testing."""
    # 1x1 pixel transparent PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501


def _get_test_audio_base64() -> str:
    """Get a small test audio file as base64 for testing."""
    # Minimal WAV file (1 second of silence)
    return (
        "UklGRjIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQ4AAAAAAAAAAAAAAAAAAA=="
    )


def _get_test_video_base64() -> str:
    """Get a small test video file as base64 for testing."""
    # Minimal valid video file would be much larger; for testing we use a placeholder
    return "PLACEHOLDER_VIDEO_DATA"


def _get_joke_class(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> Any:
    """:private:"""

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
        return all(key in ["setup", "punchline"] for key in result)

    if schema_type == "pydantic":
        return Joke, validate_joke

    if schema_type == "typeddict":
        return JokeDict, validate_joke_dict

    if schema_type == "json_schema":
        return Joke.model_json_schema(), validate_joke_dict
    msg = "Invalid schema type"
    raise ValueError(msg)


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
def magic_function(_input: int) -> int:
    """Applies a magic function to an input."""
    return _input + 2


@tool
def magic_function_no_args() -> int:
    """Calculates a magic function."""
    return 5


def _validate_tool_call_message(message: AIMessage) -> None:
    """Validate that a message contains tool calls in content blocks format."""

    if isinstance(message.content, list):
        tool_call_blocks = [
            block
            for block in message.content
            if isinstance(block, dict) and is_tool_call_block(block)
        ]
        assert len(tool_call_blocks) >= 1

        for tool_call in tool_call_blocks:
            # Ensure each tool call has the required fields
            assert "name" in tool_call
            assert "args" in tool_call
            assert "id" in tool_call
    # (No fallback, since the tools attribute makes the same search as the list
    # comprehension above)


def _validate_tool_call_message_no_args(message: AIMessage) -> None:
    """Validate that a message contains a single tool call with no arguments.

    Used for testing tool calls without arguments, such as
    ``magic_function_no_args``.
    """
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function_no_args"
    assert tool_call["args"] == {}
    assert tool_call["id"] is not None


@tool
def unicode_customer(customer_name: str, description: str) -> str:
    """Tool for creating a customer with a name containing Unicode characters.

    Args:
        customer_name: The customer's name in their native language.
        description: Description of the customer.

    Returns:
        A confirmation message about the customer creation.
    """
    return f"Created customer: {customer_name} - {description}"


class ChatModelV1IntegrationTests(ChatModelV1Tests):
    """Base class for v1 chat model integration tests.

    TODO: verify this entire docstring!

    Test subclasses must implement the ``chat_model_class`` and
    ``chat_model_params`` properties to specify what model to test and its
    initialization parameters.

    Example:

    .. code-block:: python

        from typing import Type

        from langchain_tests.integration_tests import ChatModelV1IntegrationTests
        from my_package.chat_models import MyChatModel


        class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
            @property
            def chat_model_class(self) -> Type[MyV1ChatModel]:
                # Return the chat model class to test here
                return MyChatModel

            @property
            def chat_model_params(self) -> dict:
                # Return initialization parameters for the v1 model.
                return {"model": "model-001", "temperature": 0}

    .. note::
          API references for individual test methods include troubleshooting tips.


    Test subclasses **must** implement the following two properties:

    chat_model_class
        The chat model class to test, e.g., ``ChatParrotLinkV1``.

        Example:

        .. code-block:: python

            @property
            def chat_model_class(self) -> Type[ChatParrotLinkV1]:
                return ChatParrotLinkV1

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

        TODO

    .. dropdown:: tool_choice_value

        TODO

    .. dropdown:: has_tool_choice

        TODO

    .. dropdown:: has_structured_output

        TODO

    .. dropdown:: structured_output_kwargs

        TODO

    .. dropdown:: supports_json_mode

        TODO

    .. dropdown:: returns_usage_metadata

        TODO

    .. dropdown:: supports_anthropic_inputs

        TODO

    .. dropdown:: supports_image_tool_message

        TODO

    .. dropdown:: supported_usage_metadata_details

        TODO

    .. dropdown:: enable_vcr_tests

        Property controlling whether to enable select tests that rely on
        `VCR <https://vcrpy.readthedocs.io/en/latest/>`_ caching of HTTP calls, such
        as benchmarking tests.

        To enable these tests, follow these steps:

        1. Override the ``enable_vcr_tests`` property to return ``True``:

            .. code-block:: python

                @property
                def enable_vcr_tests(self) -> bool:
                    return True

        2. Configure VCR to exclude sensitive headers and other information from cassettes.

            .. important::
                VCR will by default record authentication headers and other sensitive
                information in cassettes. Read below for how to configure what
                information is recorded in cassettes.

            To add configuration to VCR, add a ``conftest.py`` file to the ``tests/``
            directory and implement the ``vcr_config`` fixture there.

            ``langchain-tests`` excludes the headers ``'authorization'``,
            ``'x-api-key'``, and ``'api-key'`` from VCR cassettes. To pick up this
            configuration, you will need to add ``conftest.py`` as shown below. You can
            also exclude additional headers, override the default exclusions, or apply
            other customizations to the VCR configuration. See example below:

            .. code-block:: python
                :caption: tests/conftest.py

                import pytest
                from langchain_tests.conftest import _base_vcr_config as _base_vcr_config

                _EXTRA_HEADERS = [
                    # Specify additional headers to redact
                    ("user-agent", "PLACEHOLDER"),
                ]


                def remove_response_headers(response: dict) -> dict:
                    # If desired, remove or modify headers in the response.
                    response["headers"] = {}
                    return response


                @pytest.fixture(scope="session")
                def vcr_config(_base_vcr_config: dict) -> dict:  # noqa: F811
                    \"\"\"Extend the default configuration from langchain_tests.\"\"\"
                    config = _base_vcr_config.copy()
                    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
                    config["before_record_response"] = remove_response_headers

                    return config

            .. dropdown:: Compressing cassettes

                ``langchain-tests`` includes a custom VCR serializer that compresses
                cassettes using gzip. To use it, register the ``yaml.gz`` serializer
                to your VCR fixture and enable this serializer in the config. See
                example below:

                .. code-block:: python
                    :caption: tests/conftest.py

                    import pytest
                    from langchain_tests.conftest import CustomPersister, CustomSerializer
                    from langchain_tests.conftest import _base_vcr_config as _base_vcr_config
                    from vcr import VCR

                    _EXTRA_HEADERS = [
                        # Specify additional headers to redact
                        ("user-agent", "PLACEHOLDER"),
                    ]


                    def remove_response_headers(response: dict) -> dict:
                        # If desired, remove or modify headers in the response.
                        response["headers"] = {}
                        return response


                    @pytest.fixture(scope="session")
                    def vcr_config(_base_vcr_config: dict) -> dict:  # noqa: F811
                        \"\"\"Extend the default configuration from langchain_tests.\"\"\"
                        config = _base_vcr_config.copy()
                        config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
                        config["before_record_response"] = remove_response_headers
                        # New: enable serializer and set file extension
                        config["serializer"] = "yaml.gz"
                        config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

                        return config


                    def pytest_recording_configure(config: dict, vcr: VCR) -> None:
                        vcr.register_persister(CustomPersister())
                        vcr.register_serializer("yaml.gz", CustomSerializer())


                You can inspect the contents of the compressed cassettes (e.g., to
                ensure no sensitive information is recorded) using

                .. code-block:: bash

                    gunzip -k /path/to/tests/cassettes/TestClass_test.yaml.gz

                or by using the serializer:

                .. code-block:: python

                    from langchain_tests.conftest import CustomPersister, CustomSerializer

                    cassette_path = "/path/to/tests/cassettes/TestClass_test.yaml.gz"
                    requests, responses = CustomPersister().load_cassette(path, CustomSerializer())

        3. Run tests to generate VCR cassettes.

            Example:

            .. code-block:: bash

                uv run python -m pytest tests/integration_tests/test_chat_models.py::TestMyModel::test_stream_time

            This will generate a VCR cassette for the test in
            ``tests/integration_tests/cassettes/``.

            .. important::
                You should inspect the generated cassette to ensure that it does not
                contain sensitive information. If it does, you can modify the
                ``vcr_config`` fixture to exclude headers or modify the response
                before it is recorded.

            You can then commit the cassette to your repository. Subsequent test runs
            will use the cassette instead of making HTTP calls.

    """  # noqa: E501

    @property
    def standard_chat_model_params(self) -> dict:
        """:private:"""
        return {}

    def test_invoke(self, model: BaseChatModel) -> None:
        """Test to verify that ``model.invoke(simple_message)`` works.

        A model should be able to produce a non-empty ``AIMessage`` in response to
        ``"Hello"``. The message should at least contain a ``TextContentBlock`` with
        text populated.

        .. important::
            This should pass for all integrations!

        .. dropdown:: Troubleshooting

            TODO

        """
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

        text_contentblock = result.content[0]
        assert is_text_block(text_contentblock)

    async def test_ainvoke(self, model: BaseChatModel) -> None:
        """Test to verify that ``await model.ainvoke(simple_message)`` works.

        A model should be able to produce a non-empty ``AIMessage`` in response to
        ``"Hello"``. The message should at least contain a ``TextContentBlock`` with
        text populated.

        .. important::
            This should pass for all integrations!

        Passing this test does not indicate a "natively async" implementation, but
        rather that the model can be used in an async context.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_invoke`.
            because ``ainvoke`` has a default implementation that calls ``invoke`` in an
            async context.

        """
        result = await model.ainvoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

        text_contentblock = result.content[0]
        assert is_text_block(text_contentblock)

    def test_stream(self, model: BaseChatModel) -> None:
        """Test to verify that ``model.stream(simple_message)`` works.

        .. important::
            This should pass for all integrations!

        Passing this test does not indicate a "streaming" implementation, but rather
        that the model can be used in a streaming context. For instance, a model
        that yields at least one chunk in response to ``"Hello"``.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_invoke`.
            because ``stream`` has a default implementation that calls ``invoke`` and
            yields the result as a single chunk.

        """
        num_chunks = 0
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, list)
            num_chunks += 1
        assert num_chunks > 0

    async def test_astream(self, model: BaseChatModel) -> None:
        """Test to verify that ``await model.astream(simple_message)`` works.

        .. important::
            This should pass for all integrations!

        Passing this test does not indicate a "natively async" or "streaming"
        implementation, but rather that the model can be used in an async streaming
        context.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_stream`.
            and
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_ainvoke`.
            because ``astream`` has a default implementation that calls ``_stream`` in
            an async context if it is implemented, or ``ainvoke`` and yields the result
            as a single ``AIMessageChunk`` chunk if not.

        """
        num_chunks = 0
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, list)
            num_chunks += 1
        assert num_chunks > 0

    def test_batch(self, model: BaseChatModel) -> None:
        """Test to verify that ``model.batch([messages])`` works.

        .. important::
            This should pass for all integrations!

        Tests the model's ability to process multiple prompts in a single batch. We
        expect that the ``TextContentBlock`` of each response is populated with text.

        Passing this test does not indicate a "natively batching" or "batching"
        implementation, but rather that the model can be used in a batching context. For
        instance, your model may internally call ``invoke`` for each message in the
        batch, even if the model provider does not support batching natively.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_invoke`
            because ``batch`` has a default implementation that calls ``invoke`` for
            each message in the batch.

            If that test passes but not this one, you should make sure your ``batch``
            method does not raise any exceptions, and that it returns a list of valid
            :class:`~langchain_core.v1.messages.AIMessage` objects.

        """
        batch_results = model.batch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert len(result.content) > 0
            assert isinstance(result.text, str)
            assert len(result.text) > 0

    async def test_abatch(self, model: BaseChatModel) -> None:
        """Test to verify that ``await model.abatch([messages])`` works.

        .. important::
            This should pass for all integrations!

        Tests the model's ability to process multiple prompts in a single batch
        asynchronously. We expect that the ``TextContentBlock`` of each response is
        populated with text.

        Passing this test does not indicate a "natively batching" or "batching"
        implementation, but rather that the model can be used in a batching context. For
        instance, your model may internally call ``ainvoke`` for each message in the
        batch, even if the model provider does not support batching natively.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_batch`
            and
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_ainvoke`
            because ``abatch`` has a default implementation that calls ``ainvoke`` for
            each message in the batch.

            If those tests pass but not this one, you should make sure your ``abatch``
            method does not raise any exceptions, and that it returns a list of valid
            :class:`~langchain_core.v1.messages.AIMessage` objects.

        """
        batch_results = await model.abatch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert len(result.content) > 0
            assert isinstance(result.text, str)
            assert len(result.text) > 0

    def test_conversation(self, model: BaseChatModel) -> None:
        """Test to verify that the model can handle multi-turn conversations.

        .. important::
            This should pass for all integrations!

        Tests the model's ability to process a sequence of alternating human and AI
        messages as context for generating the next response. We expect that the
        ``TextContentBlock`` of each response is populated with text.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_invoke`
            because this test also uses ``model.invoke()``.

            If that test passes but not this one, you should verify that:
            1. Your model correctly processes the message history
            2. The model maintains appropriate context from previous messages
            3. The response is a valid :class:`~langchain_core.v1.messages.AIMessage`

        """
        messages = [
            HumanMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]
        result = model.invoke(messages)  # type: ignore[arg-type]
        assert result is not None
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        """Test to verify that the model can handle double-message conversations.

        .. important::
            This should pass for all integrations!

        Tests the model's ability to process a sequence of double-system, double-human,
        and double-ai messages as context for generating the next response. We expect
        that the ``TextContentBlock`` of each response is populated with text.

        .. dropdown:: Troubleshooting

            First, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_invoke`
            because this test also uses ``model.invoke()``.

            Second, debug
            :meth:`~langchain_tests.integration_tests.chat_models_v1.ChatModelV1IntegrationTests.test_conversation`
            because this test is the "basic case" without double messages.

            If that test passes those but not this one, you should verify that:
            1. Your model API can handle double messages, or the integration should merge messages before sending them to the API.
            2. The response is a valid :class:`~langchain_core.v1.messages.AIMessage`

        """  # noqa: E501
        messages = [
            SystemMessage("hello"),
            SystemMessage("hello"),
            HumanMessage("hello"),
            HumanMessage("hello"),
            AIMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]
        result = model.invoke(messages)  # type: ignore[arg-type]
        assert result is not None
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_usage_metadata(self, model: BaseChatModel) -> None:
        """Test to verify that the model returns correct usage metadata.

        This test is optional and should be skipped if the model does not return
        usage metadata (see Configuration below).

        .. versionchanged:: 0.3.17

            Additionally check for the presence of ``model_name`` in the response
            metadata, which is needed for usage tracking in callback handlers.

        .. dropdown:: Configuration

            By default, this test is run.

            To disable this feature, set the ``returns_usage_metadata`` property to
            ``False`` in your test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def returns_usage_metadata(self) -> bool:
                        return False

            This test can also check the format of specific kinds of usage metadata
            based on the ``supported_usage_metadata_details`` property. This property
            should be configured as follows with the types of tokens that the model
            supports tracking:

            TODO: check this!

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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

            TODO

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

        # Check model_name is in response_metadata
        # (Needed for langchain_core.callbacks.usage)
        model_name = result.response_metadata.get("model_name")
        assert isinstance(model_name, str)
        assert model_name != "", "model_name is empty"

        # TODO: check these
        # `input_tokens` is the total, possibly including other unclassified or
        # system-level tokens.
        if "audio_input" in self.supported_usage_metadata_details["invoke"]:
            # Checks if the specific chat model integration being tested has declared
            # that it supports reporting token counts specifically for `audio_input`
            msg = self.invoke_with_audio_input()  # To be implemented in test subclass
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                input_token_details := usage_metadata.get("input_token_details")
            ) is not None
            assert isinstance(input_token_details.get("audio"), int)
            # Asserts that total input tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in input_token_details.values() if isinstance(v, int)
            )
            assert usage_metadata.get("input_tokens", 0) >= total_detailed_tokens
        if "audio_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_audio_output()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                output_token_details := usage_metadata.get("output_token_details")
            ) is not None
            assert isinstance(output_token_details.get("audio"), int)
            # Asserts that total output tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in output_token_details.values() if isinstance(v, int)
            )
            assert usage_metadata.get("output_tokens", 0) >= total_detailed_tokens
        if "reasoning_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_reasoning_output()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                output_token_details := usage_metadata.get("output_token_details")
            ) is not None
            assert isinstance(output_token_details.get("reasoning"), int)
            # Asserts that total output tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in output_token_details.values() if isinstance(v, int)
            )
            assert usage_metadata.get("output_tokens", 0) >= total_detailed_tokens
        if "cache_read_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_read_input()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                input_token_details := usage_metadata.get("input_token_details")
            ) is not None
            assert isinstance(input_token_details.get("cache_read"), int)
            # Asserts that total input tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in input_token_details.values() if isinstance(v, int)
            )
            assert usage_metadata.get("input_tokens", 0) >= total_detailed_tokens
        if "cache_creation_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_creation_input()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                input_token_details := usage_metadata.get("input_token_details")
            ) is not None
            assert isinstance(input_token_details.get("cache_creation"), int)
            # Asserts that total input tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in input_token_details.values() if isinstance(v, int)
            )
            assert usage_metadata.get("input_tokens", 0) >= total_detailed_tokens

    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        """Test usage metadata in streaming mode.

        Test to verify that the model returns correct usage metadata in streaming mode.

        .. versionchanged:: 0.3.17

            Additionally check for the presence of ``model_name`` in the response
            metadata, which is needed for usage tracking in callback handlers.

        .. dropdown:: Configuration

            By default, this test is run.
            To disable this feature, set ``returns_usage_metadata`` to ``False`` in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def returns_usage_metadata(self) -> bool:
                        return False

            This test can also check the format of specific kinds of usage metadata
            based on the ``supported_usage_metadata_details`` property. This property
            should be configured as follows with the types of tokens that the model
            supports tracking:

            TODO: check this!

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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

            TODO

        """
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")

        full: Optional[AIMessageChunk] = None
        for chunk in model.stream("Write me 2 haikus. Only include the haikus."):
            assert isinstance(chunk, AIMessageChunk)
            # Only one chunk is allowed to set usage_metadata.input_tokens
            # if multiple do, it's likely a bug that will result in overcounting
            # input tokens (since the total number of input tokens applies to the full
            # generation, not individual chunks)
            if full and full.usage_metadata and full.usage_metadata["input_tokens"]:
                assert (
                    not chunk.usage_metadata or not chunk.usage_metadata["input_tokens"]
                ), (
                    "Only one chunk should set input_tokens,"
                    " the rest should be 0 or None"
                )
            full = chunk if full is None else cast("AIMessageChunk", full + chunk)

        assert isinstance(full, AIMessageChunk)
        assert full.usage_metadata is not None
        assert isinstance(full.usage_metadata["input_tokens"], int)
        assert isinstance(full.usage_metadata["output_tokens"], int)
        assert isinstance(full.usage_metadata["total_tokens"], int)

        # Check model_name is in response_metadata
        # (Needed for langchain_core.callbacks.usage)
        model_name = full.response_metadata.get("model_name")
        assert isinstance(model_name, str)
        assert model_name != "", "model_name is empty"

        # TODO: check these
        if "audio_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_audio_input(stream=True)
            assert msg.usage_metadata is not None
            assert isinstance(
                msg.usage_metadata.get("input_token_details", {}).get("audio"), int
            )
        if "audio_output" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_audio_output(stream=True)
            assert msg.usage_metadata is not None
            assert isinstance(
                msg.usage_metadata.get("output_token_details", {}).get("audio"), int
            )
        if "reasoning_output" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_reasoning_output(stream=True)
            assert msg.usage_metadata is not None
            assert isinstance(
                msg.usage_metadata.get("output_token_details", {}).get("reasoning"), int
            )
        if "cache_read_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_cache_read_input(stream=True)
            assert msg.usage_metadata is not None
            assert isinstance(
                msg.usage_metadata.get("input_token_details", {}).get("cache_read"), int
            )
        if "cache_creation_input" in self.supported_usage_metadata_details["stream"]:
            msg = self.invoke_with_cache_creation_input(stream=True)
            assert msg.usage_metadata is not None
            assert isinstance(
                msg.usage_metadata.get("input_token_details", {}).get("cache_creation"),
                int,
            )

    def test_stop_sequence(self, model: BaseChatModel) -> None:
        """Test that model does not fail when invoked with the ``stop`` parameter,
        which is a standard parameter for stopping generation at a certain token.

        `More on standard parameters <https://python.langchain.com/docs/concepts/chat_models/#standard-parameters>`__

        .. important::
            This should pass for all integrations!

        .. dropdown:: Troubleshooting

            TODO

        """
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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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
                def test_tool_calling(self, model: BaseChatModelV1) -> None:
                    super().test_tool_calling(model)

            Otherwise, in the case that only one tool is bound, ensure that
            ``tool_choice`` supports the string ``'any'`` to force calling that tool.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        # Emit warning if tool_choice_value property is overridden

        # TODO remove since deprecated?
        # if inspect.getattr_static(
        #     self, "tool_choice_value"
        # ) is not inspect.getattr_static(
        #     ChatModelV1IntegrationTests, "tool_choice_value"
        # ):
        #     warn_deprecated(
        #         "0.3.15",
        #         message=(
        #             "`tool_choice_value` will be removed in version 0.3.20. If a "
        #             "model supports `tool_choice`, it should accept `tool_choice='any' "  # noqa: E501
        #             "and `tool_choice=<string name of tool>`. If the model does not "
        #             "support `tool_choice`, override the `supports_tool_choice` "
        #             "property to return `False`."
        #         ),
        #         removal="0.3.20",
        #     )

        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )
        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message(result)

        # Test stream()
        full: Optional[AIMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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
                async def test_tool_calling_async(self, model: BaseChatModelV1) -> None:
                    await super().test_tool_calling_async(model)

            Otherwise, in the case that only one tool is bound, ensure that
            ``tool_choice`` supports the string ``'any'`` to force calling that tool.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )
        query = "What is the value of magic_function(3)? Use the tool."
        result = await model_with_tools.ainvoke(query)
        _validate_tool_call_message(result)

        # Test astream()
        full: Optional[AIMessageChunk] = None
        async for chunk in model_with_tools.astream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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
                def test_bind_runnables_as_tools(self, model: BaseChatModelV1) -> None:
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

        if self.has_tool_choice:
            tool_choice: Optional[str] = "any"
        else:
            tool_choice = None

        model_with_tools = model.bind_tools([tool_], tool_choice=tool_choice)
        query = "Using the tool, generate a Pirate greeting."
        result = model_with_tools.invoke(query)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        tool_call = result.tool_calls[0]
        assert tool_call["args"].get(
            "answer_style"
        )  # TODO: do we need to handle if args is str?  # noqa: E501
        assert is_tool_call_block(tool_call)

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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            TODO: verify this!

            If this test fails, check that:

            1. The model can correctly handle message histories that include ``AIMessage`` objects with ``""`` ``TextContentBlock``s.
            2. The ``tool_calls`` attribute on ``AIMessage`` objects is correctly handled and passed to the model in an appropriate format.
            3. The model can correctly handle ``ToolMessage`` objects with string content and arbitrary string values for ``tool_call_id``.

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
            # String content (e.g. OpenAI)
            create_tool_call(function_name, function_args, id="abc123"),
            ToolMessage(
                json.dumps({"result": 3}), tool_call_id="abc123", status="success"
            ),
        ]
        result_string_content = model_with_tools.invoke(
            messages_string_content  # type: ignore[arg-type]
        )  # TODO
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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model can correctly handle message histories that include ``AIMessage`` objects with list content.
            2. The ``tool_calls`` attribute on ``AIMessage`` objects is correctly handled and passed to the model in an appropriate format.
            3. The model can correctly handle ``ToolMessage`` objects with string content and arbitrary string values for ``tool_call_id``.

            You can ``xfail`` the test if tool calling is implemented but this format
            is not supported.

            .. code-block:: python

                @pytest.mark.xfail(reason=("Not implemented."))
                def test_tool_message_histories_list_content(self, *args: Any) -> None:
                    super().test_tool_message_histories_list_content(*args)

        """  # noqa: E501
        pytest.xfail("Test not implemented yet.")

        # TODO
        # if not self.has_tool_calling:
        #     pytest.skip("Test requires tool calling.")

        # model_with_tools = model.bind_tools([my_adder_tool])
        # function_name = "my_adder_tool"
        # function_args = {"a": 1, "b": 2}

        # messages_list_content = [
        #     HumanMessage("What is 1 + 2"),
        #     # List content (e.g., Anthropic)
        #     AIMessage(
        #         [
        #             {"type": "text", "text": "some text"},
        #             {
        #                 "type": "tool_use",
        #                 "id": "abc123",
        #                 "name": function_name,
        #                 "input": function_args,
        #             },
        #         ],
        #         tool_calls=[
        #             {
        #                 "name": function_name,
        #                 "args": function_args,
        #                 "id": "abc123",
        #                 "type": "tool_call",
        #             },
        #         ],
        #     ),
        #     ToolMessage(
        #         json.dumps({"result": 3}),
        #         name=function_name,
        #         tool_call_id="abc123",
        #     ),
        # ]
        # result_list_content = model_with_tools.invoke(messages_list_content)
        # assert isinstance(result_list_content, AIMessage)

    def test_tool_choice(self, model: BaseChatModel) -> None:
        """Test that the model can force tool calling via the ``tool_choice``
        parameter. This test is skipped if the ``has_tool_choice`` property on the
        test class is set to False.

        This test is optional and should be skipped if the model does not support
        tool calling (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_choice`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_choice(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check whether the ``test_tool_calling`` test is passing.
            If it is not, refer to the troubleshooting steps in that test first.

            If ``test_tool_calling`` is passing, check that the underlying model
            supports forced tool calling. If it does, ``bind_tools`` should accept a
            ``tool_choice`` parameter that can be used to force a tool call.

            It should accept:

            1. The string ``'any'`` to force calling the bound tool, and,
            2. The string name of the tool to force calling that tool.

        """
        if not self.has_tool_choice or not self.has_tool_calling:
            pytest.skip("Test requires tool choice.")

        @tool
        def get_weather(location: str) -> str:
            """Get weather at a location."""
            return "It's sunny."

        for tool_choice in ["any", "magic_function"]:
            model_with_tools = model.bind_tools(
                [magic_function, get_weather], tool_choice=tool_choice
            )
            result = model_with_tools.invoke("Hello!")
            assert isinstance(result, AIMessage)
            assert result.tool_calls
            if tool_choice == "magic_function":
                assert result.tool_calls[0]["name"] == "magic_function"

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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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
                def test_tool_calling_with_no_arguments(self, model: BaseChatModelV1) -> None:
                    super().test_tool_calling_with_no_arguments(model)

            Otherwise, in the case that only one tool is bound, ensure that
            ``tool_choice`` supports the string ``'any'`` to force calling that tool.

        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function_no_args], tool_choice=tool_choice_value
        )
        query = "What is the value of magic_function_no_args()? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message_no_args(result)

        full: Optional[AIMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message_no_args(full)

    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that ``ToolMessage`` with ``status="error"`` can be handled.

        These messages may take the form:

        .. code-block:: python

            ToolMessage(
                content="Error: Missing required argument 'b'.",
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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the ``status`` field on ``ToolMessage``
            objects is either ignored or passed to the model appropriately.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        model_with_tools = model.bind_tools([my_adder_tool])
        messages = [
            HumanMessage("What is 1 + 2?"),
            create_tool_call(
                "my_adder_tool", {"a": 1}, id="abc123"
            ),  # Missing required argument 'b'
            ToolMessage(
                "Error: Missing required argument 'b'.",
                tool_call_id="abc123",
                status="error",
            ),
        ]
        result = model_with_tools.invoke(messages)  # type: ignore[arg-type]
        assert isinstance(result, AIMessage)

    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that the model can process few-shot examples with tool calls.

        These are represented as a sequence of messages of the following form:

        - ``HumanMessage`` with ``TextContentBlock`` content;
        - ``AIMessage`` with the ``tool_calls`` attribute populated;
        - ``ToolMessage`` with string content;
        - ``ToolMessage`` with content block content;
        - ``AIMessage`` with ``TextContentBlock`` content (an answer);
        - ``HumanMessage`` with ``TextContentBlock`` content (a follow-up question).

        This test should be skipped if the model does not support tool calling
        (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            This test uses `a utility function <https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.tool_example_to_messages.html>`__
            in ``langchain_core`` to generate a sequence of messages representing
            "few-shot" examples.

            If this test fails, check that the model can correctly handle this
            sequence of messages.

            You can ``xfail`` the test if tool calling is implemented but this format
            is not supported.

            .. code-block:: python

                @pytest.mark.xfail(reason=("Not implemented."))
                def test_structured_few_shot_examples(self, *args: Any) -> None:
                    super().test_structured_few_shot_examples(*args)

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        model_with_tools = model.bind_tools([my_adder_tool], tool_choice="any")
        function_result = json.dumps({"result": 3})

        tool_schema = my_adder_tool.args_schema
        assert isinstance(tool_schema, type)
        assert issubclass(tool_schema, BaseModel)
        # TODO verify this is correct
        few_shot_messages = tool_example_to_messages(
            "What is 1 + 2",
            [tool_schema(a=1, b=2)],
            tool_outputs=[function_result],
            ai_response=function_result,
        )

        messages = [*few_shot_messages, HumanMessage("What is 3 + 4")]
        result = model_with_tools.invoke(messages)  # type: ignore[arg-type]
        assert isinstance(result, AIMessage)

    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        """Test to verify structured output is generated both on ``invoke()`` and ``stream()``.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V2 models.

            ``langchain_core`` implements `a utility function <https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html>`__
            that will accommodate most formats.

            See `example implementation <https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output>`__
            of ``with_structured_output``.

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

        assert len(invoke_callback.options) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(invoke_callback.options[0], dict)
        assert isinstance(
            invoke_callback.options[0]["ls_structured_output_format"]["schema"], dict
        )
        assert invoke_callback.options[0]["ls_structured_output_format"][
            "schema"
        ] == convert_to_json_schema(schema)

        stream_callback = _TestCallbackHandler()

        for chunk in chat.stream(
            "Tell me a joke about cats.", config={"callbacks": [stream_callback]}
        ):
            validation_function(chunk)
        assert chunk

        assert len(stream_callback.options) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(stream_callback.options[0], dict)
        assert isinstance(
            stream_callback.options[0]["ls_structured_output_format"]["schema"], dict
        )
        assert stream_callback.options[0]["ls_structured_output_format"][
            "schema"
        ] == convert_to_json_schema(schema)

    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> None:
        """Test to verify structured output is generated both on ``invoke()`` and ``stream()``.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V2 models.

            ``langchain_core`` implements `a utility function <https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html>`__
            that will accommodate most formats.

            See `example implementation <https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output>`__
            of ``with_structured_output``.

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

        assert len(ainvoke_callback.options) == 1, (
            "Expected on_chat_model_start to be called once"
        )
        assert isinstance(ainvoke_callback.options[0], dict)
        assert isinstance(
            ainvoke_callback.options[0]["ls_structured_output_format"]["schema"], dict
        )
        assert ainvoke_callback.options[0]["ls_structured_output_format"][
            "schema"
        ] == convert_to_json_schema(schema)

        astream_callback = _TestCallbackHandler()

        async for chunk in chat.astream(
            "Tell me a joke about cats.", config={"callbacks": [astream_callback]}
        ):
            validation_function(chunk)
        assert chunk

        assert len(astream_callback.options) == 1, (
            "Expected on_chat_model_start to be called once"
        )

        assert isinstance(astream_callback.options[0], dict)
        assert isinstance(
            astream_callback.options[0]["ls_structured_output_format"]["schema"], dict
        )
        assert astream_callback.options[0]["ls_structured_output_format"][
            "schema"
        ] == convert_to_json_schema(schema)

    @pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Test requires pydantic 2.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        """Test to verify we can generate structured output using ``pydantic.v1.BaseModel``.

        ``pydantic.v1.BaseModel`` is available in the Pydantic 2 package.

        This test is optional and should be skipped if the model does not support
        structured output (see Configuration below).

        .. dropdown:: Configuration

            To disable structured output tests, set ``has_structured_output`` to False
            in your test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles both JSON Schema and Pydantic V1 models.

            ``langchain_core`` implements `a utility function <https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html>`__
            that will accommodate most formats.

            See `example implementation <https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output>`__
            of ``with_structured_output``.

        """  # noqa: E501
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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_structured_output(self) -> bool:
                        return False

            By default, ``has_structured_output`` is True if a model overrides the
            ``with_structured_output`` or ``bind_tools`` methods.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles Pydantic V2 models with optional parameters.

            ``langchain_core`` implements `a utility function <https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html>`__
            that will accommodate most formats.

            See `example implementation <https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output>`__
            of ``with_structured_output``.

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
        """Test structured output via `JSON mode. <https://python.langchain.com/docs/concepts/structured_outputs/#json-mode>`_.

        This test is optional and should be skipped if the model does not support
        the JSON mode feature (see Configuration below).

        .. dropdown:: Configuration

            To disable this test, set ``supports_json_mode`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def supports_json_mode(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            See `example implementation <https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output>`__
            of ``with_structured_output``.

        """
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
        chat = model.with_structured_output(Joke, method="json_mode")
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

    def test_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs.

        This test should be skipped (see Configuration below) if the model does not
        support PDF inputs. These will take the form:

        .. code-block:: python

            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 image data>",
                "mime_type": "application/pdf",
            }

        See https://python.langchain.com/docs/concepts/multimodality/

        .. dropdown:: Configuration

            To disable this test, set ``supports_pdf_inputs`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):

                    @property
                    def supports_pdf_inputs(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with pdf content blocks, including base64-encoded files. Otherwise, set
            the ``supports_pdf_inputs`` property to False.

        """
        pytest.xfail("Test not implemented yet.")

        # TODO
        # if not self.supports_pdf_inputs:
        #     pytest.skip("Model does not support PDF inputs.")
        # url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        # pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        # message = HumanMessage(
        #     [
        #         {
        #             "type": "text",
        #             "text": "Summarize this document:",
        #         },
        #         {
        #             "type": "file",
        #             "source_type": "base64",
        #             "mime_type": "application/pdf",
        #             "data": pdf_data,
        #         },
        #     ]
        # )
        # _ = model.invoke([message])

        # # Test OpenAI Chat Completions format
        # message = HumanMessage(
        #     [
        #         {
        #             "type": "text",
        #             "text": "Summarize this document:",
        #         },
        #         {
        #             "type": "file",
        #             "file": {
        #                 "filename": "test file.pdf",
        #                 "file_data": f"data:application/pdf;base64,{pdf_data}",
        #             },
        #         },
        #     ]
        # )
        # _ = model.invoke([message])

    def test_audio_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process audio inputs.

        This test should be skipped (see Configuration below) if the model does not
        support audio inputs. These will take the form:

        .. code-block:: python

            # AudioContentBlock
            {
                "type": "audio",
                "base64": "<base64 audio data>",
                "mime_type": "audio/wav",  # or appropriate mime-type
            }

        See https://python.langchain.com/docs/concepts/multimodality/

        .. dropdown:: Configuration

            To disable this test, set ``supports_audio_content_blocks`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):

                    @property
                    def supports_audio_content_blocks(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with audio content blocks. Otherwise, set the ``supports_audio_content_blocks``
            property to False.

        """  # noqa: E501
        if not self.supports_audio_content_blocks:
            pytest.skip("Model does not support AudioContentBlock inputs.")

        url = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Alcal%C3%A1_de_Henares_%28RPS_13-04-2024%29_canto_de_ruise%C3%B1or_%28Luscinia_megarhynchos%29_en_el_Soto_del_Henares.wav"
        audio_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        message = HumanMessage(
            [
                create_text_block("Describe this audio:"),
                create_audio_block(
                    base64=audio_data,
                    mime_type="audio/wav",
                ),
            ]
        )
        _ = model.invoke([message])

        # TODO?
        # Test OpenAI Chat Completions format
        # message = HumanMessage(
        #     [
        #         {
        #             "type": "text",
        #             "text": "Describe this audio:",
        #         },
        #         {
        #             "type": "input_audio",
        #             "input_audio": {"data": audio_data, "format": "wav"},
        #         },
        #     ]
        # )
        # _ = model.invoke([message])

    def test_audio_content_blocks_processing(self, model: BaseChatModel) -> None:
        """Test audio content block processing with transcription.

        TODO: expand docstring

        """
        if not self.supports_audio_content_blocks:
            pytest.skip("Model does not support audio inputs.")

        audio_block = create_audio_block(
            base64=_get_test_audio_base64(),
            mime_type="audio/wav",
        )
        text_block = create_text_block("Transcribe this audio file.")

        result = model.invoke([HumanMessage([text_block, audio_block])])

        assert isinstance(result, AIMessage)
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process image inputs.

        This test should be skipped (see Configuration below) if the model does not
        support image inputs. These will take the form:

        .. code-block:: python

            # ImageContentBlock
            {
                "type": "image",
                "base64": "<base64 audio data>",
                "mime_type": "image/png",  # or appropriate mime-type
            }

        TODO: verify this
        For backward-compatibility, we must also support OpenAI-style
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

            To disable this test, set ``supports_image_content_blocks`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def supports_image_content_blocks(self) -> bool:
                        return False

                    # Can also explicitly disable testing image URLs:
                    @property
                    def supports_image_urls(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with image content blocks, including base64-encoded images. Otherwise, set
            the ``supports_image_content_blocks`` property to False.

        """
        if not self.supports_image_content_blocks:
            pytest.skip("Model does not support image message.")

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        # TODO?
        # OpenAI format, base64 data
        # message = HumanMessage(
        #     content=[
        #         {"type": "text", "text": "describe the weather in this image"},
        #         {
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        #         },
        #     ],
        # )
        # _ = model.invoke([message])

        # Standard format, base64 data
        message = HumanMessage(
            [
                create_text_block("describe the weather in this image"),
                create_image_block(
                    base64=image_data,
                    mime_type="image/jpeg",
                ),
            ],
        )
        _ = model.invoke([message])

        # TODO?
        # Standard format, URL
        # if self.supports_image_urls:
        #     message = HumanMessage(
        #         content=[
        #             {"type": "text", "text": "describe the weather in this image"},
        #             {
        #                 "type": "image",
        #                 "source_type": "url",
        #                 "url": image_url,
        #             },
        #         ],
        #     )
        #     _ = model.invoke([message])

    def test_image_tool_message(self, model: BaseChatModel) -> None:
        """Test that the model can process ToolMessages with image inputs.

        TODO: is this needed?

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
            )

        containing image content blocks in OpenAI Chat Completions format, in addition
        to messages of the form:

        .. code-block:: python

            ToolMessage(
                content=[
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_data,
                        "mime_type": "image/jpeg",
                    },
                ],
                tool_call_id="1",
            )

        containing image content blocks in standard format.

        This test can be skipped by setting the ``supports_image_tool_message`` property
        to False (see Configuration below).

        .. dropdown:: Configuration

            To disable this test, set ``supports_image_tool_message`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def supports_image_tool_message(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that the model can correctly handle messages
            with image content blocks in ToolMessages, including base64-encoded
            images. Otherwise, set the ``supports_image_tool_message`` property to
            False.

        """
        pytest.xfail("Test not implemented yet.")

        # TODO
        # if not self.supports_image_tool_message:
        #     pytest.skip("Model does not support image tool message.")
        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        # image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        # # Support both OpenAI and standard formats
        # oai_format_message = ToolMessage(
        #     content=[
        #         {
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        #         },
        #     ],
        #     tool_call_id="1",
        #     name="random_image",
        # )

        # standard_format_message = ToolMessage(
        #     content=[
        #         {
        #             "type": "image",
        #             "source_type": "base64",
        #             "data": image_data,
        #             "mime_type": "image/jpeg",
        #         },
        #     ],
        #     tool_call_id="1",
        #     name="random_image",
        # )

        # for tool_message in [oai_format_message, standard_format_message]:
        #     messages = [
        #         HumanMessage(
        #             "get a random image using the tool and describe the weather"
        #         ),
        #         AIMessage(
        #             [],
        #             tool_calls=[
        #                 {
        #                     "type": "tool_call",
        #                     "id": "1",
        #                     "name": "random_image",
        #                     "args": {},
        #                 }
        #             ],
        #         ),
        #         tool_message,
        #     ]

        #     def random_image() -> str:
        #         """Return a random image."""
        #         return ""

        #     _ = model.bind_tools([random_image]).invoke(messages)

    def test_image_content_blocks_with_analysis(self, model: BaseChatModel) -> None:
        """Test image analysis using ``ImageContentBlock``s.

        TODO: expand docstring

        """
        if not self.supports_image_content_blocks:
            pytest.skip("Model does not support image inputs.")

        image_block = create_image_block(
            base64=_get_test_image_base64(),
            mime_type="image/png",
        )
        text_block = create_text_block("Analyze this image in detail.")

        result = model.invoke([HumanMessage([text_block, image_block])])

        assert isinstance(result, AIMessage)
        text_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and is_text_block(block)
        ]
        assert len(text_blocks) > 0
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_video_content_blocks(self, model: BaseChatModel) -> None:
        """Test video content block processing.

        TODO: expand docstring

        """
        if not self.supports_video_content_blocks:
            pytest.skip("Model does not support video inputs.")

        video_block = create_video_block(
            base64=_get_test_video_base64(),
            mime_type="video/mp4",
        )
        text_block = create_text_block("Describe what you see in this video.")

        result = model.invoke([HumanMessage([text_block, video_block])])

        assert isinstance(result, AIMessage)
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_anthropic_inputs(self, model: BaseChatModel) -> None:
        """Test that model can process Anthropic-style message histories.

        TODO?

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

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
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
        pytest.xfail("Test not implemented yet.")

        # TODO
        # if not self.supports_anthropic_inputs:
        #     pytest.skip("Model does not explicitly support Anthropic inputs.")

        # # Anthropic-format tool
        # color_picker = {
        #     "name": "color_picker",
        #     "input_schema": {
        #         "type": "object",
        #         "properties": {
        #             "fav_color": {"type": "string"},
        #         },
        #         "required": ["fav_color"],
        #     },
        #     "description": "Input your fav color and get a random fact about it.",
        #     "cache_control": {"type": "ephemeral"},
        # }

        # human_content: list[dict] = [
        #     {
        #         "type": "text",
        #         "text": "what's your favorite color in this image",
        #         "cache_control": {"type": "ephemeral"},
        #     },
        # ]
        # if self.supports_image_inputs:
        #     image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        #     image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")  # noqa: E501
        #     human_content.append(
        #         {
        #             "type": "image",
        #             "source": {
        #                 "type": "base64",
        #                 "media_type": "image/jpeg",
        #                 "data": image_data,
        #             },
        #         }
        #     )
        # messages = [
        #     SystemMessage("you're a good assistant"),
        #     HumanMessage(human_content),  # type: ignore[arg-type]
        #     AIMessage(
        #         [
        #             {"type": "text", "text": "Hmm let me think about that"},
        #             {
        #                 "type": "tool_use",
        #                 "input": {"fav_color": "green"},
        #                 "id": "foo",
        #                 "name": "color_picker",
        #             },
        #         ],
        #         tool_calls=[
        #             {
        #                 "name": "color_picker",
        #                 "args": {"fav_color": "green"},
        #                 "id": "foo",
        #                 "type": "tool_call",
        #             }
        #         ],
        #     ),
        #     ToolMessage("That's a great pick!", tool_call_id="foo"),
        # ]
        # response = model.bind_tools([color_picker]).invoke(messages)
        # assert isinstance(response, AIMessage)

        # # Test thinking blocks
        # messages = [
        #     HumanMessage(
        #         [
        #             {
        #                 "type": "text",
        #                 "text": "Hello",
        #             },
        #         ]
        #     ),
        #     AIMessage(
        #         [
        #             {
        #                 "type": "thinking",
        #                 "thinking": "I'm thinking...",
        #                 "signature": "abc123",
        #             },
        #             {
        #                 "type": "text",
        #                 "text": "Hello, how are you?",
        #             },
        #         ]
        #     ),
        #     HumanMessage(
        #         [
        #             {
        #                 "type": "text",
        #                 "text": "Well, thanks.",
        #             },
        #         ]
        #     ),
        # ]
        # response = model.invoke(messages)
        # assert isinstance(response, AIMessage)

    def test_message_with_name(self, model: BaseChatModel) -> None:
        """Test that ``HumanMessage`` with values for the ``name`` field can be handled.

        This test expects that the model with a non-empty ``TextContentBlock``.

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
        assert len(result.content) > 0
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_agent_loop(self, model: BaseChatModel) -> None:
        """Test that the model supports a simple ReAct agent loop. This test is skipped
        if the ``has_tool_calling`` property on the test class is set to False.

        This test is optional and should be skipped if the model does not support
        tool calling (see Configuration below).

        .. dropdown:: Configuration

            To disable tool calling tests, set ``has_tool_calling`` to False in your
            test class:

            .. code-block:: python

                class TestMyV1ChatModelIntegration(ChatModelV1IntegrationTests):
                    @property
                    def has_tool_calling(self) -> bool:
                        return False

        .. dropdown:: Troubleshooting

            If this test fails, check that ``bind_tools`` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            Check also that all required information (e.g., tool calling identifiers)
            from ``AIMessage`` objects is propagated correctly to model payloads.

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

        llm_with_tools = model.bind_tools([get_weather])
        input_message = HumanMessage("What is the weather in San Francisco, CA?")
        tool_call_message = llm_with_tools.invoke([input_message])
        assert isinstance(tool_call_message, AIMessage)
        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        tool_message = get_weather.invoke(tool_call)
        assert isinstance(tool_message, ToolMessage)
        response = llm_with_tools.invoke(
            [
                input_message,
                tool_call_message,
                tool_message,
            ]
        )
        assert isinstance(response, AIMessage)

    @pytest.mark.benchmark
    @pytest.mark.vcr
    def test_stream_time(
        self, model: BaseChatModel, benchmark: BenchmarkFixture, vcr: Cassette
    ) -> None:
        """Test that streaming does not introduce undue overhead.

        See ``enable_vcr_tests`` dropdown :class:`above <ChatModelV1IntegrationTests>`
        for more information.

        .. dropdown:: Configuration

            This test can be enabled or disabled using the ``enable_vcr_tests``
            property. For example, to disable the test, set this property to ``False``:

            .. code-block:: python

                @property
                def enable_vcr_tests(self) -> bool:
                    return False

            .. important::

                VCR will by default record authentication headers and other sensitive
                information in cassettes. See ``enable_vcr_tests`` dropdown
                :class:`above <ChatModelV1IntegrationTests>` for how to configure what
                information is recorded in cassettes.

        """
        if not self.enable_vcr_tests:
            pytest.skip("VCR not set up.")

        def _run() -> None:
            for _ in model.stream("Write a story about a cat."):
                pass

        if not vcr.responses:
            _run()
        else:
            benchmark(_run)

    def invoke_with_audio_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        # To be implemented in test subclass
        raise NotImplementedError

    def invoke_with_audio_output(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        # To be implemented in test subclass
        raise NotImplementedError

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        # To be implemented in test subclass
        raise NotImplementedError

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        # To be implemented in test subclass
        raise NotImplementedError

    def invoke_with_cache_creation_input(self, *, stream: bool = False) -> AIMessage:
        """:private:"""
        # To be implemented in test subclass
        raise NotImplementedError

    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: Optional[str] = None,
        force_tool_call: bool = True,
    ) -> None:
        """Generic integration test for Unicode characters in tool calls.

        Args:
            model: The chat model to test
            tool_choice: Tool choice parameter to pass to ``bind_tools()`` (provider-specific)
            force_tool_call: Whether to force a tool call (use ``tool_choice=True`` if None)

        Tests that Unicode characters in tool call arguments are preserved correctly,
        not escaped as ``\\uXXXX`` sequences.
        """  # noqa: E501
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling support.")

        # Configure tool choice based on provider capabilities
        if tool_choice is None and force_tool_call:
            tool_choice = "any"

        if tool_choice is not None:
            llm_with_tool = model.bind_tools(
                [unicode_customer], tool_choice=tool_choice
            )
        else:
            llm_with_tool = model.bind_tools([unicode_customer])

        # Test with Chinese characters
        msgs = [
            HumanMessage(
                "Create a customer named '你好啊集团' (Hello Group) - a Chinese "
                "technology company"
            )
        ]
        ai_msg = llm_with_tool.invoke(msgs)

        assert isinstance(ai_msg, AIMessage)
        assert isinstance(ai_msg.tool_calls, list)

        if force_tool_call:
            assert len(ai_msg.tool_calls) >= 1, (
                f"Expected at least 1 tool call, got {len(ai_msg.tool_calls)}"
            )

        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            assert tool_call["name"] == "unicode_customer"
            assert "args" in tool_call

            # Verify Unicode characters are properly handled
            args = tool_call["args"]
            assert "customer_name" in args
            customer_name = args["customer_name"]

            # The model should include the Unicode characters, not escaped sequences
            assert (
                "你好" in customer_name
                or "你" in customer_name
                or "好" in customer_name
            ), f"Unicode characters not found in: {customer_name}"

        # Test with additional Unicode examples - Japanese
        msgs_jp = [
            HumanMessage(
                "Create a customer named 'こんにちは株式会社' (Hello Corporation) - a "
                "Japanese company"
            )
        ]
        ai_msg_jp = llm_with_tool.invoke(msgs_jp)

        assert isinstance(ai_msg_jp, AIMessage)

        if force_tool_call:
            assert len(ai_msg_jp.tool_calls) >= 1

        if ai_msg_jp.tool_calls:
            tool_call_jp = ai_msg_jp.tool_calls[0]
            args_jp = tool_call_jp["args"]
            customer_name_jp = args_jp["customer_name"]

            # Verify Japanese Unicode characters are preserved
            assert (
                "こんにちは" in customer_name_jp
                or "株式会社" in customer_name_jp
                or "こ" in customer_name_jp
                or "ん" in customer_name_jp
            ), f"Japanese Unicode characters not found in: {customer_name_jp}"

    def test_complex_multimodal_reasoning(self, model: BaseChatModel) -> None:
        """Test complex reasoning with multiple content types.

        TODO: expand docstring

        """
        if not self.supports_multimodal_reasoning:
            pytest.skip("Model does not support multimodal reasoning.")

        content_blocks: list[types.ContentBlock] = [
            create_text_block(
                "Compare these media files and provide reasoning analysis:"
            ),
            create_image_block(
                base64=_get_test_image_base64(),
                mime_type="image/png",
            ),
        ]

        if self.supports_audio_content_blocks:
            content_blocks.append(
                create_audio_block(
                    base64=_get_test_audio_base64(),
                    mime_type="audio/wav",
                )
            )

        message = HumanMessage(content=cast("list[types.ContentBlock]", content_blocks))
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        if self.supports_reasoning_content_blocks:
            reasoning_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_reasoning_block(block)
            ]
            assert len(reasoning_blocks) > 0

    def test_citation_generation_with_sources(self, model: BaseChatModel) -> None:
        """Test that the model can generate ``Citations`` with source links.

        TODO: expand docstring

        """
        if not self.supports_structured_citations:
            pytest.skip("Model does not support structured citations.")

        message = HumanMessage(
            "Provide factual information about the distance to the moon with proper "
            "citations to scientific sources."
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for text blocks with citations
        text_blocks_with_citations = []
        for block in result.content:
            if (
                isinstance(block, dict)
                and is_text_block(block)
                and "annotations" in block
            ):
                annotations = cast("list[dict[str, Any]]", block.get("annotations", []))
                citations = [
                    ann
                    for ann in annotations
                    if isinstance(ann, dict) and ann.get("type") == "citation"
                ]
                if citations:
                    text_blocks_with_citations.append(block)
        assert len(text_blocks_with_citations) > 0

        # Validate citation structure
        for block in text_blocks_with_citations:
            annotations = cast("list[dict[str, Any]]", block.get("annotations", []))
            for annotation in annotations:
                if annotation.get("type") == "citation":
                    # TODO: evaluate these since none are *technically* required
                    # This may be a test that needs adjustment on per-integration basis
                    assert "cited_text" in annotation
                    assert "start_index" in annotation
                    assert "end_index" in annotation

    def test_web_search_integration(self, model: BaseChatModel) -> None:
        """Test web search content blocks integration.

        TODO: expand docstring

        """
        if not self.supports_web_search_blocks:
            pytest.skip("Model does not support web search blocks.")

        message = HumanMessage(
            "Search for the latest developments in quantum computing."
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for web search blocks
        search_call_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "web_search_call"
        ]
        search_result_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "web_search_result"
        ]
        # TODO: should this be one or the other or both?
        assert len(search_call_blocks) > 0 or len(search_result_blocks) > 0

    def test_code_interpreter_blocks(self, model: BaseChatModel) -> None:
        """Test code interpreter content blocks.

        TODO: expand docstring

        """
        if not self.supports_code_interpreter:
            pytest.skip("Model does not support code interpreter blocks.")

        message = HumanMessage("Calculate the factorial of 10 using Python code.")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for code interpreter blocks
        code_blocks = [
            block
            for block in result.content
            if isinstance(block, dict)
            and block.get("type")
            in [
                "code_interpreter_call",
                "code_interpreter_output",
                "code_interpreter_result",
            ]
        ]
        # TODO: should we require all three types or just an output/result?
        assert len(code_blocks) > 0

    def test_tool_calling_with_content_blocks(self, model: BaseChatModel) -> None:
        """Test tool calling with content blocks.

        TODO: expand docstring

        """
        if not self.has_tool_calling:
            pytest.skip("Model does not support tool calls.")

        @tool
        def calculate_area(length: float, width: float) -> str:
            """Calculate the area of a rectangle."""
            area = length * width
            return f"The area is {area} square units."

        model_with_tools = model.bind_tools([calculate_area])
        message = HumanMessage(
            "Calculate the area of a rectangle with length 5 and width 3."
        )

        result = model_with_tools.invoke([message])
        _validate_tool_call_message(result)

    def test_plaintext_content_blocks_from_documents(
        self, model: BaseChatModel
    ) -> None:
        """Test PlainTextContentBlock for document plaintext content.

        TODO: expand docstring

        """
        if not self.supports_plaintext_content_blocks:
            pytest.skip("Model does not support PlainTextContentBlock.")

        # Test with PlainTextContentBlock (plaintext from document)
        plaintext_block = create_plaintext_block(
            text="This is plaintext content extracted from a document.",
            file_id="doc_123",
        )

        message = HumanMessage(
            content=cast("list[types.ContentBlock]", [plaintext_block])
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        # TODO expand

    def test_content_block_streaming_integration(self, model: BaseChatModel) -> None:
        """Test streaming with content blocks.

        TODO: expand docstring

        """
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Write a detailed explanation of machine learning.",
                }
            ]
        )

        chunks = []
        for chunk in model.stream([message]):
            chunks.append(chunk)
            assert isinstance(chunk, (AIMessage, AIMessageChunk))

        assert len(chunks) > 1  # Should receive multiple chunks

        # Aggregate chunks
        final_message = chunks[0]
        for chunk in chunks[1:]:
            final_message = final_message + chunk

        assert isinstance(final_message.content, list)

    def test_error_handling_with_invalid_content_blocks(
        self, model: BaseChatModel
    ) -> None:
        """Test error handling with various invalid content block configurations.

        TODO: expand docstring

        """
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        test_cases = [
            {"type": "text"},  # Missing text field
            {"type": "image"},  # Missing url/mime_type
            {"type": "tool_call", "name": "test"},  # Missing args/id
        ]

        for invalid_block in test_cases:
            message = HumanMessage([invalid_block])  # type: ignore[list-item]

            # Should either handle gracefully or raise appropriate error
            try:
                result = model.invoke([message])
                assert isinstance(result, AIMessage)
            except (ValueError, TypeError, KeyError) as e:
                # Acceptable to raise validation errors
                assert len(str(e)) > 0

    async def test_async_content_blocks_processing(self, model: BaseChatModel) -> None:
        """Test asynchronous processing of content blocks.

        TODO: expand docstring

        """
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        message = HumanMessage("Generate a creative story about space exploration.")

        result = await model.ainvoke([message])
        assert isinstance(result, AIMessage)

    def test_content_blocks_with_callbacks(self, model: BaseChatModel) -> None:
        """Test that content blocks work correctly with callback handlers.

        TODO: expand docstring

        """
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        class ContentBlockCallbackHandler(BaseCallbackHandler):
            def __init__(self) -> None:
                self.messages_seen: list[BaseMessage] = []

            def on_chat_model_start(
                self,
                serialized: Any,  # noqa: ARG002
                messages: Any,
                **kwargs: Any,  # noqa: ARG002
            ) -> None:
                self.messages_seen.extend(messages)

        callback_handler = ContentBlockCallbackHandler()

        message = HumanMessage("Test message for callback handling.")

        result = model.invoke([message], config={"callbacks": [callback_handler]})

        assert isinstance(result, AIMessage)
        assert len(callback_handler.messages_seen) > 0
        assert any(
            hasattr(msg, "content") and isinstance(msg.content, list)
            for msg in callback_handler.messages_seen
        )

    def test_input_conversion_string(self, model: BaseChatModel) -> None:
        """Test that string input is properly converted to messages.

        TODO: expand docstring

        """
        result = model.invoke("Test string input")
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_input_conversion_empty_string(self, model: BaseChatModel) -> None:
        """Test that empty string input is handled gracefully.

        TODO: expand docstring

        """
        result = model.invoke("")
        assert isinstance(result, AIMessage)

    def test_input_conversion_message_v1_list(self, model: BaseChatModel) -> None:
        """Test that v1 message list input is handled correctly.

        TODO: expand docstring

        """
        messages = [HumanMessage("Test message")]
        result = model.invoke(messages)
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_text_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can handle the ``TextContentBlock`` format."""
        if not self.supports_text_content_blocks:
            pytest.skip("Model does not support TextContentBlock (rare!)")

        text_block = create_text_block("Hello, world!")
        message = HumanMessage(content=[text_block])

        result = model.invoke([message])
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_mixed_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can handle messages with mixed content blocks."""
        if not (
            self.supports_text_content_blocks and self.supports_image_content_blocks
        ):
            pytest.skip(
                "Model doesn't support mixed content blocks (concurrent text and image)"
            )

        content_blocks: list[types.ContentBlock] = [
            create_text_block("Describe this image:"),
            create_image_block(
                base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                mime_type="image/png",
            ),
        ]

        message = HumanMessage(content=content_blocks)
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_reasoning_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can generate ``ReasoningContentBlock``."""
        if not self.supports_reasoning_content_blocks:
            pytest.skip("Model does not support ReasoningContentBlock.")

        message = HumanMessage("Think step by step: What is 2 + 2?")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        if isinstance(result.content, list):
            reasoning_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_reasoning_block(block)
            ]
            assert len(reasoning_blocks) > 0

    def test_non_standard_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can handle ``NonStandardContentBlock``."""
        if not self.supports_non_standard_blocks:
            pytest.skip("Model does not support NonStandardContentBlock.")

        non_standard_block = create_non_standard_block(
            {
                "custom_field": "custom_value",
                "data": [1, 2, 3],
            }
        )

        message = HumanMessage(content=[non_standard_block])

        # Should not raise an error
        result = model.invoke([message])
        assert isinstance(result, AIMessage)

    def test_invalid_tool_call_handling_basic(self, model: BaseChatModel) -> None:
        """Test that the model can handle ``InvalidToolCall`` blocks gracefully."""
        if not self.supports_invalid_tool_calls:
            pytest.skip("Model does not support InvalidToolCall handling.")

        invalid_tool_call: InvalidToolCall = {
            "type": "invalid_tool_call",
            "name": "nonexistent_tool",
            "args": None,
            "id": "invalid_123",
            "error": "Tool not found",
        }

        # Create a message with invalid tool call in history
        ai_message = AIMessage(content=[invalid_tool_call])
        follow_up = HumanMessage("Please try again with a valid approach.")

        result = model.invoke([ai_message, follow_up])
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_file_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can handle ``FileContentBlock``."""
        if not self.supports_file_content_blocks:
            pytest.skip("Model does not support FileContentBlock.")

        file_block = create_file_block(
            base64="SGVsbG8sIHdvcmxkIQ==",  # "Hello, world!"
            mime_type="text/plain",
        )

        message = HumanMessage(content=[file_block])
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        assert result.content is not None
