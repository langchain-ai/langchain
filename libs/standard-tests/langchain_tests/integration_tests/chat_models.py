"""Integration tests for chat models."""

from __future__ import annotations

import base64
import json
import os
import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    tool_example_to_messages,
)
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from typing_extensions import TypedDict, override

from langchain_tests.unit_tests.chat_models import ChatModelTests
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION

if TYPE_CHECKING:
    from pytest_benchmark.fixture import (
        BenchmarkFixture,
    )
    from vcr.cassette import Cassette


def _get_joke_class(  # noqa: RET503
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> Any:
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
        return all(key in {"setup", "punchline"} for key in result)

    if schema_type == "pydantic":
        return Joke, validate_joke

    if schema_type == "typeddict":
        return JokeDict, validate_joke_dict

    if schema_type == "json_schema":
        return Joke.model_json_schema(), validate_joke_dict


class _TestCallbackHandler(BaseCallbackHandler):
    options: list[dict[str, Any] | None]

    def __init__(self) -> None:
        super().__init__()
        self.options = []

    @override
    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.options.append(options)


class _MagicFunctionSchema(BaseModel):
    input: int = Field(..., gt=-1000, lt=1000)


@tool(args_schema=_MagicFunctionSchema)
def magic_function(_input: int) -> int:
    """Apply a magic function to an input."""
    return _input + 2


@tool
def magic_function_no_args() -> int:
    """Calculate a magic function."""
    return 5


def _validate_tool_call_message(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function"
    assert tool_call["args"] == {"input": 3}
    assert tool_call["id"] is not None
    assert tool_call.get("type") == "tool_call"

    content_tool_calls = [
        block for block in message.content_blocks if block["type"] == "tool_call"
    ]
    assert len(content_tool_calls) == 1
    content_tool_call = content_tool_calls[0]
    assert content_tool_call["name"] == "magic_function"
    assert content_tool_call["args"] == {"input": 3}
    assert content_tool_call["id"] is not None


def _validate_tool_call_message_no_args(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function_no_args"
    assert tool_call["args"] == {}
    assert tool_call["id"] is not None
    assert tool_call.get("type") == "tool_call"


def _get_base64_from_url(url: str) -> str:
    user_agent = os.environ.get("LANGCHAIN_TESTS_USER_AGENT")
    if not user_agent:
        warning_message = (
            "LANGCHAIN_TESTS_USER_AGENT environment variable not set. "
            "langchain-tests pulls (CC0 License) audio data from wikimedia.org. "
            "Consider setting a user agent to identify your requests. See "
            "https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy"
        )
        warnings.warn(warning_message, stacklevel=2)
    headers = {"User-Agent": user_agent} if user_agent else {}
    httpx_response = httpx.get(url, headers=headers).content
    return base64.b64encode(httpx_response).decode("utf-8")


@tool
def unicode_customer(customer_name: str, description: str) -> str:
    """Tool for creating a customer with Unicode name.

    Args:
        customer_name: The customer's name in their native language.
        description: Description of the customer.

    Returns:
        A confirmation message about the customer creation.

    """
    return f"Created customer: {customer_name} - {description}"


class ChatModelIntegrationTests(ChatModelTests):
    '''Base class for chat model integration tests.

    Test subclasses must implement the `chat_model_class` and
    `chat_model_params` properties to specify what model to test and its
    initialization parameters.

    ```python
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
    ```

    !!! note
        API references for individual test methods include troubleshooting tips.


    Test subclasses **must** implement the following two properties:

    `chat_model_class`: The chat model class to test, e.g., `ChatParrotLink`.

    ```python
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink
    ```

    `chat_model_params`: Initialization parameters for the chat model.

    ```python
    @property
    def chat_model_params(self) -> dict:
        return {"model": "bird-brain-001", "temperature": 0}
    ```

    In addition, test subclasses can control what features are tested (such as tool
    calling or multi-modality) by selectively overriding the following properties.

    Expand to see details:

    ???+ info "`has_tool_calling`"

        Boolean property indicating whether the chat model supports tool calling.

        By default, this is determined by whether the chat model's `bind_tools` method
        is overridden. It typically does not need to be overridden on the test class.

        ```python
        @property
        def has_tool_calling(self) -> bool:
            return True
        ```

    ??? info "`has_tool_choice`"

        Boolean property indicating whether the chat model supports forcing tool
        calling via a `tool_choice` parameter.

        By default, this is determined by whether the parameter is included in the
        signature for the corresponding `bind_tools` method.

        If `True`, the minimum requirement for this feature is that
        `tool_choice='any'` will force a tool call, and `tool_choice=<tool name>`
        will force a call to a specific tool.

        ```python
        @property
        def has_tool_choice(self) -> bool:
            return False
        ```

    ??? info "`has_structured_output`"

        Boolean property indicating whether the chat model supports structured
        output.

        By default, this is determined by whether the chat model's
        `with_structured_output` method is overridden. If the base implementation is
        intended to be used, this method should be overridden.

        See docs for [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

        ```python
        @property
        def has_structured_output(self) -> bool:
            return True
        ```

    ??? info "`structured_output_kwargs`"

        Dict property specifying additional kwargs to pass to
        `with_structured_output()` when running structured output tests.

        Override this to customize how your model generates structured output.

        The most common use case is specifying the `method` parameter:

        - `'function_calling'`: Uses tool/function calling to enforce the schema.
        - `'json_mode'`: Uses the model's JSON mode.
        - `'json_schema'`: Uses native JSON schema support (e.g., OpenAI's structured
            outputs).

        ```python
        @property
        def structured_output_kwargs(self) -> dict:
            return {"method": "json_schema"}
        ```

    ??? info "`supports_json_mode`"

        Boolean property indicating whether the chat model supports
        `method='json_mode'` in `with_structured_output`.

        Defaults to `False`.

        JSON mode constrains the model to output valid JSON without enforcing
        a specific schema (unlike `'function_calling'` or `'json_schema'` methods).

        When using JSON mode, you must prompt the model to output JSON in your
        message.

        !!! example

            ```python
            structured_llm = llm.with_structured_output(MySchema, method="json_mode")
            structured_llm.invoke("... Return the result as JSON.")
            ```

        See docs for [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

        ```python
        @property
        def supports_json_mode(self) -> bool:
            return True
        ```

    ??? info "`supports_image_inputs`"

        Boolean property indicating whether the chat model supports image inputs.

        Defaults to `False`.

        If set to `True`, the chat model will be tested by inputting an
        `ImageContentBlock` with the shape:

        ```python
        {
            "type": "image",
            "base64": "<base64 image data>",
            "mime_type": "image/jpeg",  # or appropriate MIME type
        }
        ```

        In addition to OpenAI-style content blocks:

        ```python
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        }
        ```

        See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

        ```python
        @property
        def supports_image_inputs(self) -> bool:
            return True
        ```

    ??? info "`supports_image_urls`"

        Boolean property indicating whether the chat model supports image inputs from
        URLs.

        Defaults to `False`.

        If set to `True`, the chat model will be tested using content blocks of the
        form

        ```python
        {
            "type": "image",
            "url": "https://...",
        }
        ```

        See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

        ```python
        @property
        def supports_image_urls(self) -> bool:
            return True
        ```

    ??? info "`supports_image_tool_message`"

        Boolean property indicating whether the chat model supports a `ToolMessage`
        that includes image content, e.g. in the OpenAI Chat Completions format.

        Defaults to `False`.

        ```python
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
        ```

        ...as well as the LangChain `ImageContentBlock` format:

        ```python
        ToolMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/jpeg",
                },
            ],
            tool_call_id="1",
            name="random_image",
        )
        ```

        If set to `True`, the chat model will be tested with message sequences that
        include `ToolMessage` objects of this form.

        ```python
        @property
        def supports_image_tool_message(self) -> bool:
            return True
        ```

    ??? info "`supports_pdf_inputs`"

        Boolean property indicating whether the chat model supports PDF inputs.

        Defaults to `False`.

        If set to `True`, the chat model will be tested by inputting a
        `FileContentBlock` with the shape:

        ```python
        {
            "type": "file",
            "base64": "<base64 file data>",
            "mime_type": "application/pdf",
        }
        ```

        See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

        ```python
        @property
        def supports_pdf_inputs(self) -> bool:
            return True
        ```

    ??? info "`supports_pdf_tool_message`"

        Boolean property indicating whether the chat model supports a `ToolMessage`
        that includes PDF content using the LangChain `FileContentBlock` format.

        Defaults to `False`.

        ```python
        ToolMessage(
            content=[
                {
                    "type": "file",
                    "base64": pdf_data,
                    "mime_type": "application/pdf",
                },
            ],
            tool_call_id="1",
            name="random_pdf",
        )
        ```

        If set to `True`, the chat model will be tested with message sequences that
        include `ToolMessage` objects of this form.

        ```python
        @property
        def supports_pdf_tool_message(self) -> bool:
            return True
        ```

    ??? info "`supports_audio_inputs`"

        Boolean property indicating whether the chat model supports audio inputs.

        Defaults to `False`.

        If set to `True`, the chat model will be tested by inputting an
        `AudioContentBlock` with the shape:

        ```python
        {
            "type": "audio",
            "base64": "<base64 audio data>",
            "mime_type": "audio/wav",  # or appropriate MIME type
        }
        ```

        See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

        ```python
        @property
        def supports_audio_inputs(self) -> bool:
            return True
        ```

        !!! warning
            This test downloads audio data from wikimedia.org. You may need to set the
            `LANGCHAIN_TESTS_USER_AGENT` environment variable to identify these tests,
            e.g.,

            ```bash
            export LANGCHAIN_TESTS_USER_AGENT="CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
            ```

            Refer to the [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

    ??? info "`supports_video_inputs`"

        Boolean property indicating whether the chat model supports image inputs.

        Defaults to `False`.

        No current tests are written for this feature.

    ??? info "`returns_usage_metadata`"

        Boolean property indicating whether the chat model returns usage metadata
        on invoke and streaming responses.

        Defaults to `True`.

        `usage_metadata` is an optional dict attribute on `AIMessage` objects that track
        input and output tokens.

        [See more](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.ai.UsageMetadata).

        ```python
        @property
        def returns_usage_metadata(self) -> bool:
            return False
        ```

        Models supporting `usage_metadata` should also return the name of the underlying
        model in the `response_metadata` of the `AIMessage`.

    ??? info "`supports_anthropic_inputs`"

        Boolean property indicating whether the chat model supports Anthropic-style
        inputs.

        Defaults to `False`.

        These inputs might feature "tool use" and "tool result" content blocks, e.g.,

        ```python
        [
            {"type": "text", "text": "Hmm let me think about that"},
            {
                "type": "tool_use",
                "input": {"fav_color": "green"},
                "id": "foo",
                "name": "color_picker",
            },
        ]
        ```

        If set to `True`, the chat model will be tested using content blocks of this
        form.

        ```python
        @property
        def supports_anthropic_inputs(self) -> bool:
            return True
        ```

    ??? info "`supported_usage_metadata_details`"

        Property controlling what usage metadata details are emitted in both invoke
        and stream.

        Defaults to `{"invoke": [], "stream": []}`.

        `usage_metadata` is an optional dict attribute on `AIMessage` objects that track
        input and output tokens.

        [See more](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.ai.UsageMetadata).

        It includes optional keys `input_token_details` and `output_token_details`
        that can track usage details associated with special types of tokens, such as
        cached, audio, or reasoning.

        Only needs to be overridden if these details are supplied.

    ??? info "`enable_vcr_tests`"

        Property controlling whether to enable select tests that rely on
        [VCR](https://vcrpy.readthedocs.io/en/latest/) caching of HTTP calls, such
        as benchmarking tests.

        Defaults to `False`.

        To enable these tests, follow these steps:

        1. Override the `enable_vcr_tests` property to return `True`:

            ```python
            @property
            def enable_vcr_tests(self) -> bool:
                return True
            ```

        2. Configure VCR to exclude sensitive headers and other information from
            cassettes.

            !!! warning
                VCR will by default record authentication headers and other sensitive
                information in cassettes. Read below for how to configure what
                information is recorded in cassettes.

            To add configuration to VCR, add a `conftest.py` file to the `tests/`
            directory and implement the `vcr_config` fixture there.

            `langchain-tests` excludes the headers `'authorization'`,
            `'x-api-key'`, and `'api-key'` from VCR cassettes. To pick up this
            configuration, you will need to add `conftest.py` as shown below. You can
            also exclude additional headers, override the default exclusions, or apply
            other customizations to the VCR configuration. See example below:

            ```python title="tests/conftest.py"
            import pytest
            from langchain_tests.conftest import base_vcr_config

            _EXTRA_HEADERS = [
                # Specify additional headers to redact
                ("user-agent", "PLACEHOLDER"),
            ]


            def remove_response_headers(response: dict) -> dict:
                # If desired, remove or modify headers in the response.
                response["headers"] = {}
                return response


            @pytest.fixture(scope="session")
            def vcr_config() -> dict:
                """Extend the default configuration from langchain_tests."""
                config = base_vcr_config()
                config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
                config["before_record_response"] = remove_response_headers

                return config
            ```

            ??? note "Compressing cassettes"

                `langchain-tests` includes a custom VCR serializer that compresses
                cassettes using gzip. To use it, register the `yaml.gz` serializer
                to your VCR fixture and enable this serializer in the config. See
                example below:

                ```python title="tests/conftest.py"
                import pytest
                from langchain_tests.conftest import (
                    CustomPersister,
                    CustomSerializer,
                )
                from langchain_tests.conftest import base_vcr_config
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
                def vcr_config() -> dict:
                    """Extend the default configuration from langchain_tests."""
                    config = base_vcr_config()
                    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
                    config["before_record_response"] = remove_response_headers
                    # New: enable serializer and set file extension
                    config["serializer"] = "yaml.gz"
                    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

                    return config


                def pytest_recording_configure(config: dict, vcr: VCR) -> None:
                    vcr.register_persister(CustomPersister())
                    vcr.register_serializer("yaml.gz", CustomSerializer())
                ```

                You can inspect the contents of the compressed cassettes (e.g., to
                ensure no sensitive information is recorded) using

                ```bash
                gunzip -k /path/to/tests/cassettes/TestClass_test.yaml.gz
                ```

                ...or by using the serializer:

                ```python
                from langchain_tests.conftest import (
                    CustomPersister,
                    CustomSerializer,
                )

                cassette_path = "/path/to/tests/cassettes/TestClass_test.yaml.gz"
                requests, responses = CustomPersister().load_cassette(
                    path, CustomSerializer()
                )
                ```

        3. Run tests to generate VCR cassettes.

            ```bash title="Example"
            uv run python -m pytest tests/integration_tests/test_chat_models.py::TestMyModel::test_stream_time
            ```

            This will generate a VCR cassette for the test in
            `tests/integration_tests/cassettes/`.

            !!! warning
                You should inspect the generated cassette to ensure that it does not
                contain sensitive information. If it does, you can modify the
                `vcr_config` fixture to exclude headers or modify the response
                before it is recorded.

            You can then commit the cassette to your repository. Subsequent test runs
            will use the cassette instead of making HTTP calls.
    '''  # noqa: E501

    @property
    def standard_chat_model_params(self) -> dict[str, Any]:
        """Standard parameters for chat model."""
        return {}

    def test_invoke(self, model: BaseChatModel) -> None:
        """Test to verify that `model.invoke(simple_message)` works.

        This should pass for all integrations.

        ??? question "Troubleshooting"

            If this test fails, you should make sure your `_generate` method
            does not raise any exceptions, and that it returns a valid
            `langchain_core.outputs.chat_result.ChatResult` like so:

            ```python
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Output text"))]
            )
            ```

        """
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

    async def test_ainvoke(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.ainvoke(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "natively async" implementation, but rather that the model can be used
        in an async context.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
            because `ainvoke` has a default implementation that calls `invoke` in an
            async context.

            If that test passes but not this one, you should make sure your `_agenerate`
            method does not raise any exceptions, and that it returns a valid
            `langchain_core.outputs.chat_result.ChatResult` like so:

            ```python
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Output text"))]
            )
            ```
        """
        result = await model.ainvoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_stream(self, model: BaseChatModel) -> None:
        """Test to verify that `model.stream(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "streaming" implementation, but rather that the model can be used in a
        streaming context.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
            because `stream` has a default implementation that calls `invoke` and
            yields the result as a single chunk.

            If that test passes but not this one, you should make sure your `_stream`
            method does not raise any exceptions, and that it yields valid
            `langchain_core.outputs.chat_generation.ChatGenerationChunk`
            objects like so:

            ```python
            yield ChatGenerationChunk(message=AIMessageChunk(content="chunk text"))
            ```

            The final chunk must have `chunk_position='last'` to signal stream
            completion. This enables proper parsing of `tool_call_chunks` into
            `tool_calls` on the aggregated message:

            ```python
            for i, token in enumerate(tokens):
                is_last = i == len(tokens) - 1
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=token,
                        chunk_position="last" if is_last else None,
                    )
                )
            ```
        """
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len(full.content_blocks) == 1
        assert full.content_blocks[0]["type"] == "text"

        # Verify chunk_position signaling
        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', "
            f"got {last_chunk.chunk_position!r}"
        )

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    async def test_astream(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.astream(simple_message)` works.

        This should pass for all integrations. Passing this test does not indicate
        a "natively async" or "streaming" implementation, but rather that the model can
        be used in an async streaming context.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_stream`.
            and
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`.
            because `astream` has a default implementation that calls `_stream` in
            an async context if it is implemented, or `ainvoke` and yields the result
            as a single chunk if not.

            If those tests pass but not this one, you should make sure your `_astream`
            method does not raise any exceptions, and that it yields valid
            `langchain_core.outputs.chat_generation.ChatGenerationChunk`
            objects like so:

            ```python
            yield ChatGenerationChunk(message=AIMessageChunk(content="chunk text"))
            ```

            See `test_stream` troubleshooting for `chunk_position` requirements.
        """
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len(full.content_blocks) == 1
        assert full.content_blocks[0]["type"] == "text"

        # Verify chunk_position signaling
        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', "
            f"got {last_chunk.chunk_position!r}"
        )

    def test_invoke_with_model_override(self, model: BaseChatModel) -> None:
        """Test that model name can be overridden at invoke time via kwargs.

        This enables dynamic model selection without creating new instances,
        which is useful for fallback strategies, A/B testing, or cost optimization.

        Test is skipped if `supports_model_override` is `False`.

        ??? question "Troubleshooting"

            If this test fails, ensure that your `_generate` method passes
            `**kwargs` through to the API request payload in a way that allows
            the `model` parameter to be overridden.

            For example:
            ```python
            def _get_request_payload(self, ..., **kwargs) -> dict:
                return {
                    "model": self.model,
                    ...
                    **kwargs,  # kwargs should come last to allow overrides
                }
            ```
        """
        if not self.supports_model_override:
            pytest.skip("Model override not supported.")

        override_model = self.model_override_value
        if not override_model:
            pytest.skip("model_override_value not specified.")

        result = model.invoke("Hello", model=override_model)
        assert result is not None
        assert isinstance(result, AIMessage)

        # Verify the overridden model was used
        model_name = result.response_metadata.get("model_name")
        assert model_name is not None, "model_name not found in response_metadata"
        assert override_model in model_name, (
            f"Expected model '{override_model}' but got '{model_name}'"
        )

    async def test_ainvoke_with_model_override(self, model: BaseChatModel) -> None:
        """Test that model name can be overridden at ainvoke time via kwargs.

        Test is skipped if `supports_model_override` is `False`.

        ??? question "Troubleshooting"

            See troubleshooting for `test_invoke_with_model_override`.
        """
        if not self.supports_model_override:
            pytest.skip("Model override not supported.")

        override_model = self.model_override_value
        if not override_model:
            pytest.skip("model_override_value not specified.")

        result = await model.ainvoke("Hello", model=override_model)
        assert result is not None
        assert isinstance(result, AIMessage)

        # Verify the overridden model was used
        model_name = result.response_metadata.get("model_name")
        assert model_name is not None, "model_name not found in response_metadata"
        assert override_model in model_name, (
            f"Expected model '{override_model}' but got '{model_name}'"
        )

    def test_stream_with_model_override(self, model: BaseChatModel) -> None:
        """Test that model name can be overridden at stream time via kwargs.

        Test is skipped if `supports_model_override` is `False`.

        ??? question "Troubleshooting"

            See troubleshooting for `test_invoke_with_model_override`.
        """
        if not self.supports_model_override:
            pytest.skip("Model override not supported.")

        override_model = self.model_override_value
        if not override_model:
            pytest.skip("model_override_value not specified.")

        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello", model=override_model):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk

        assert full is not None

        # Verify the overridden model was used
        model_name = full.response_metadata.get("model_name")
        assert model_name is not None, "model_name not found in response_metadata"
        assert override_model in model_name, (
            f"Expected model '{override_model}' but got '{model_name}'"
        )

    async def test_astream_with_model_override(self, model: BaseChatModel) -> None:
        """Test that model name can be overridden at astream time via kwargs.

        Test is skipped if `supports_model_override` is `False`.

        ??? question "Troubleshooting"

            See troubleshooting for `test_invoke_with_model_override`.
        """
        if not self.supports_model_override:
            pytest.skip("Model override not supported.")

        override_model = self.model_override_value
        if not override_model:
            pytest.skip("model_override_value not specified.")

        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello", model=override_model):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk

        assert full is not None

        # Verify the overridden model was used
        model_name = full.response_metadata.get("model_name")
        assert model_name is not None, "model_name not found in response_metadata"
        assert override_model in model_name, (
            f"Expected model '{override_model}' but got '{model_name}'"
        )

    def test_batch(self, model: BaseChatModel) -> None:
        """Test to verify that `model.batch([messages])` works.

        This should pass for all integrations. Tests the model's ability to process
        multiple prompts in a single batch.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because `batch` has a default implementation that calls `invoke` for
            each message in the batch.

            If that test passes but not this one, you should make sure your `batch`
            method does not raise any exceptions, and that it returns a list of valid
            `AIMessage` objects.

        """
        batch_results = model.batch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.text, str)
            assert len(result.content) > 0

    async def test_abatch(self, model: BaseChatModel) -> None:
        """Test to verify that `await model.abatch([messages])` works.

        This should pass for all integrations. Tests the model's ability to process
        multiple prompts in a single batch asynchronously.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_batch`
            and
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`
            because `abatch` has a default implementation that calls `ainvoke` for
            each message in the batch.

            If those tests pass but not this one, you should make sure your `abatch`
            method does not raise any exceptions, and that it returns a list of valid
            `AIMessage` objects.

        """
        batch_results = await model.abatch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.text, str)
            assert len(result.content) > 0

    def test_conversation(self, model: BaseChatModel) -> None:
        """Test to verify that the model can handle multi-turn conversations.

        This should pass for all integrations. Tests the model's ability to process
        a sequence of alternating `HumanMessage` and `AIMessage` objects as context for
        generating the next response.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because this test also uses `model.invoke`.

            If that test passes but not this one, you should verify that:

            1. Your model correctly processes the message history
            2. The model maintains appropriate context from previous messages
            3. The response is a valid `langchain_core.messages.AIMessage`

        """
        messages = [
            HumanMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]

        result = model.invoke(messages)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        """Test to verify that the model can handle double-message conversations.

        This should pass for all integrations. Tests the model's ability to process
        a sequence of double-system, double-human, and double-ai messages as context
        for generating the next response.

        ??? question "Troubleshooting"

            First, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
            because this test also uses `model.invoke`.

            Second, debug
            `langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_conversation`
            because this test is the "basic case" without double messages.

            If that test passes those but not this one, you should verify that:

            1. Your model API can handle double messages, or the integration should
                merge messages before sending them to the API.
            2. The response is a valid `langchain_core.messages.AIMessage`

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
        assert isinstance(result.text, str)
        assert len(result.content) > 0

    def test_usage_metadata(self, model: BaseChatModel) -> None:
        """Test to verify that the model returns correct usage metadata.

        This test is optional and should be skipped if the model does not return
        usage metadata (see configuration below).

        !!! warning "Behavior changed in `langchain-tests` 0.3.17"

            Additionally check for the presence of `model_name` in the response
            metadata, which is needed for usage tracking in callback handlers.

        ??? note "Configuration"

            By default, this test is run.

            To disable this feature, set `returns_usage_metadata` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def returns_usage_metadata(self) -> bool:
                    return False
            ```

            This test can also check the format of specific kinds of usage metadata
            based on the `supported_usage_metadata_details` property.

            This property should be configured as follows with the types of tokens that
            the model supports tracking:

            ```python
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
            ```

        ??? question "Troubleshooting"

            If this test fails, first verify that your model returns
            `langchain_core.messages.ai.UsageMetadata` dicts
            attached to the returned `AIMessage` object in `_generate`:

            ```python
            return ChatResult(
                generations=[
                    ChatGeneration(
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
                                },
                            },
                        )
                    )
                ]
            )
            ```

            Check also that the response includes a `model_name` key in its
            `usage_metadata`.
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
        # Needed for langchain_core.callbacks.usage
        model_name = result.response_metadata.get("model_name")
        assert isinstance(model_name, str)
        assert model_name, "model_name is empty"

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
            assert usage_metadata.get("input_tokens", 0) >= sum(
                v for v in input_token_details.values() if isinstance(v, int)
            )
        if "audio_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_audio_output()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                output_token_details := usage_metadata.get("output_token_details")
            ) is not None
            assert isinstance(output_token_details.get("audio"), int)
            # Asserts that total output tokens are at least the sum of the token counts
            assert usage_metadata.get("output_tokens", 0) >= sum(
                v for v in output_token_details.values() if isinstance(v, int)
            )
        if "reasoning_output" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_reasoning_output()
            assert (usage_metadata := msg.usage_metadata) is not None
            assert (
                output_token_details := usage_metadata.get("output_token_details")
            ) is not None
            assert isinstance(output_token_details.get("reasoning"), int)
            # Asserts that total output tokens are at least the sum of the token counts
            assert usage_metadata.get("output_tokens", 0) >= sum(
                v for v in output_token_details.values() if isinstance(v, int)
            )
        if "cache_read_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_read_input()
            usage_metadata = msg.usage_metadata
            assert usage_metadata is not None
            input_token_details = usage_metadata.get("input_token_details")
            assert input_token_details is not None
            cache_read_tokens = input_token_details.get("cache_read")
            assert isinstance(cache_read_tokens, int)
            assert cache_read_tokens >= 0
            # Asserts that total input tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in input_token_details.values() if isinstance(v, int) and v >= 0
            )
            input_tokens = usage_metadata.get("input_tokens", 0)
            assert isinstance(input_tokens, int)
            assert input_tokens >= total_detailed_tokens
        if "cache_creation_input" in self.supported_usage_metadata_details["invoke"]:
            msg = self.invoke_with_cache_creation_input()
            usage_metadata = msg.usage_metadata
            assert usage_metadata is not None
            input_token_details = usage_metadata.get("input_token_details")
            assert input_token_details is not None
            cache_creation_tokens = input_token_details.get("cache_creation")
            assert isinstance(cache_creation_tokens, int)
            assert cache_creation_tokens >= 0
            # Asserts that total input tokens are at least the sum of the token counts
            total_detailed_tokens = sum(
                v for v in input_token_details.values() if isinstance(v, int) and v >= 0
            )
            input_tokens = usage_metadata.get("input_tokens", 0)
            assert isinstance(input_tokens, int)
            assert input_tokens >= total_detailed_tokens

    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        """Test usage metadata in streaming mode.

        Test to verify that the model returns correct usage metadata in streaming mode.

        !!! warning "Behavior changed in `langchain-tests` 0.3.17"

            Additionally check for the presence of `model_name` in the response
            metadata, which is needed for usage tracking in callback handlers.

        ??? note "Configuration"

            By default, this test is run.
            To disable this feature, set `returns_usage_metadata` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def returns_usage_metadata(self) -> bool:
                    return False
            ```

            This test can also check the format of specific kinds of usage metadata
            based on the `supported_usage_metadata_details` property.

            This property should be configured as follows with the types of tokens that
            the model supports tracking:

            ```python
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
            ```

        ??? question "Troubleshooting"

            If this test fails, first verify that your model yields
            `langchain_core.messages.ai.UsageMetadata` dicts
            attached to the returned `AIMessage` object in `_stream`
            that sum up to the total usage metadata.

            Note that `input_tokens` should only be included on one of the chunks
            (typically the first or the last chunk), and the rest should have `0` or
            `None` to avoid counting input tokens multiple times.

            `output_tokens` typically count the number of tokens in each chunk, not
            the sum. This test will pass as long as the sum of `output_tokens` across
            all chunks is not `0`.

            ```python
            yield ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="Output text",
                            usage_metadata={
                                "input_tokens": (
                                    num_input_tokens if is_first_chunk else 0
                                ),
                                "output_tokens": 11,
                                "total_tokens": (
                                    11 + num_input_tokens if is_first_chunk else 11
                                ),
                                "input_token_details": {
                                    "audio": 10,
                                    "cache_creation": 200,
                                    "cache_read": 100,
                                },
                                "output_token_details": {
                                    "audio": 10,
                                    "reasoning": 200,
                                },
                            },
                        )
                    )
                ]
            )
            ```

            Check also that the aggregated response includes a `model_name` key
            in its `usage_metadata`.

        """
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")

        full: AIMessageChunk | None = None
        for chunk in model.stream("Write me 2 haikus. Only include the haikus."):
            assert isinstance(chunk, AIMessageChunk)
            # only one chunk is allowed to set usage_metadata.input_tokens
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
            # only one chunk is allowed to set usage_metadata.model_name
            # if multiple do, they'll be concatenated incorrectly
            if full and full.usage_metadata and full.usage_metadata.get("model_name"):
                assert not chunk.usage_metadata or not chunk.usage_metadata.get(
                    "model_name"
                ), "Only one chunk should set model_name, the rest should be None"
            full = chunk if full is None else full + chunk

        assert isinstance(full, AIMessageChunk)
        assert full.usage_metadata is not None
        assert isinstance(full.usage_metadata["input_tokens"], int)
        assert isinstance(full.usage_metadata["output_tokens"], int)
        assert isinstance(full.usage_metadata["total_tokens"], int)

        # Check model_name is in response_metadata
        # Needed for langchain_core.callbacks.usage
        model_name = full.response_metadata.get("model_name")
        assert isinstance(model_name, str)
        assert model_name, "model_name is empty"

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
        """Test that model does not fail when invoked with the `stop` parameter.

        The `stop` parameter is a standard parameter for stopping generation at a
        certain token.

        [More on standard parameters](https://python.langchain.com/docs/concepts/chat_models/#standard-parameters).

        This should pass for all integrations.

        ??? question "Troubleshooting"

            If this test fails, check that the function signature for `_generate`
            (as well as `_stream` and async variants) accepts the `stop` parameter:

            ```python
            def _generate(
                self,
                messages: List[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:

            ```
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

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_tool_calling(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls.

        This test is skipped if the `has_tool_calling` property on the test class is
        set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that `bind_tools` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            This test may fail if the chat model does not support a `tool_choice`
            parameter. This parameter can be used to force a tool call. If
            `tool_choice` is not supported and the model consistently fails this
            test, you can `xfail` the test:

            ```python
            @pytest.mark.xfail(reason=("Does not support tool_choice."))
            def test_tool_calling(self, model: BaseChatModel) -> None:
                super().test_tool_calling(model)
            ```

            Otherwise, in the case that only one tool is bound, ensure that
            `tool_choice` supports the string `'any'` to force calling that tool.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        # Test invoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message(result)

        # Test stream
        full: BaseMessage | None = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test that the model generates tool calls.

        This test is skipped if the `has_tool_calling` property on the test class is
        set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that `bind_tools` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            This test may fail if the chat model does not support a `tool_choice`
            parameter. This parameter can be used to force a tool call. If
            `tool_choice` is not supported and the model consistently fails this
            test, you can `xfail` the test:

            ```python
            @pytest.mark.xfail(reason=("Does not support tool_choice."))
            async def test_tool_calling_async(self, model: BaseChatModel) -> None:
                await super().test_tool_calling_async(model)
            ```

            Otherwise, in the case that only one tool is bound, ensure that
            `tool_choice` supports the string `'any'` to force calling that tool.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        # Test ainvoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = await model_with_tools.ainvoke(query)
        _validate_tool_call_message(result)

        # Test astream
        full: BaseMessage | None = None
        async for chunk in model_with_tools.astream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        """Test bind runnables as tools.

        Test that the model generates tool calls for tools that are derived from
        LangChain runnables. This test is skipped if the `has_tool_calling` property
        on the test class is set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that `bind_tools` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            This test may fail if the chat model does not support a `tool_choice`
            parameter. This parameter can be used to force a tool call. If
            `tool_choice` is not supported, set `has_tool_choice` to `False` in
            your test class:

            ```python
            @property
            def has_tool_choice(self) -> bool:
                return False
            ```

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
            tool_choice: str | None = "any"
        else:
            tool_choice = None
        model_with_tools = model.bind_tools([tool_], tool_choice=tool_choice)
        query = "Using the tool, generate a Pirate greeting."
        result = model_with_tools.invoke(query)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        tool_call = result.tool_calls[0]
        assert tool_call["args"].get("answer_style")
        assert tool_call.get("type") == "tool_call"

    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that message histories are compatible with string tool contents.

        For instance with OpenAI format contents.
        If a model passes this test, it should be compatible
        with messages generated from providers following OpenAI format.

        This test should be skipped if the model does not support tool calling
        (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that:

            1. The model can correctly handle message histories that include
                `AIMessage` objects with `""` content.
            2. The `tool_calls` attribute on `AIMessage` objects is correctly
                handled and passed to the model in an appropriate format.
            3. The model can correctly handle `ToolMessage` objects with string
                content and arbitrary string values for `tool_call_id`.

            You can `xfail` the test if tool calling is implemented but this format
            is not supported.

            ```python
            @pytest.mark.xfail(reason=("Not implemented."))
            def test_tool_message_histories_string_content(self, *args: Any) -> None:
                super().test_tool_message_histories_string_content(*args)
            ```
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": 1, "b": 2}

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
        """Test that message histories are compatible with list tool contents.

        For instance with Anthropic format contents.

        These message histories will include `AIMessage` objects with "tool use" and
        content blocks, e.g.,

        ```python
        [
            {"type": "text", "text": "Hmm let me think about that"},
            {
                "type": "tool_use",
                "input": {"fav_color": "green"},
                "id": "foo",
                "name": "color_picker",
            },
        ]
        ```

        This test should be skipped if the model does not support tool calling
        (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that:

            1. The model can correctly handle message histories that include
                `AIMessage` objects with list content.
            2. The `tool_calls` attribute on `AIMessage` objects is correctly
                handled and passed to the model in an appropriate format.
            3. The model can correctly handle ToolMessage objects with string content
                and arbitrary string values for `tool_call_id`.

            You can `xfail` the test if tool calling is implemented but this format
            is not supported.

            ```python
            @pytest.mark.xfail(reason=("Not implemented."))
            def test_tool_message_histories_list_content(self, *args: Any) -> None:
                super().test_tool_message_histories_list_content(*args)
            ```
        """
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

    def test_tool_choice(self, model: BaseChatModel) -> None:
        """Test `tool_choice` parameter.

        Test that the model can force tool calling via the `tool_choice`
        parameter. This test is skipped if the `has_tool_choice` property on the
        test class is set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_choice` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_choice(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check whether the `test_tool_calling` test is passing.
            If it is not, refer to the troubleshooting steps in that test first.

            If `test_tool_calling` is passing, check that the underlying model
            supports forced tool calling. If it does, `bind_tools` should accept a
            `tool_choice` parameter that can be used to force a tool call.

            It should accept (1) the string `'any'` to force calling the bound tool,
            and (2) the string name of the tool to force calling that tool.

        """
        if not self.has_tool_choice or not self.has_tool_calling:
            pytest.skip("Test requires tool choice.")

        @tool
        def get_weather(location: str) -> str:  # noqa: ARG001
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

        This test is skipped if the `has_tool_calling` property on the test class
        is set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that `bind_tools` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model. It should correctly handle the case where a tool has no
            arguments.

            This test may fail if the chat model does not support a `tool_choice`
            parameter. This parameter can be used to force a tool call. It may also
            fail if a provider does not support this form of tool. In these cases,
            you can `xfail` the test:

            ```python
            @pytest.mark.xfail(reason=("Does not support tool_choice."))
            def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
                super().test_tool_calling_with_no_arguments(model)
            ```

            Otherwise, in the case that only one tool is bound, ensure that
            `tool_choice` supports the string `'any'` to force calling that tool.

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function_no_args], tool_choice=tool_choice_value
        )
        query = "What is the value of magic_function_no_args()? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message_no_args(result)

        full: BaseMessage | None = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message_no_args(full)

    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that `ToolMessage` with `status="error"` can be handled.

        These messages may take the form:

        ```python
        ToolMessage(
            "Error: Missing required argument 'b'.",
            name="my_adder_tool",
            tool_call_id="abc123",
            status="error",
        )
        ```

        If possible, the `status` field should be parsed and passed appropriately
        to the model.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the `status` field on `ToolMessage`
            objects is either ignored or passed to the model appropriately.

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

    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        """Test that the model can process few-shot examples with tool calls.

        These are represented as a sequence of messages of the following form:

        - `HumanMessage` with string content;
        - `AIMessage` with the `tool_calls` attribute populated;
        - `ToolMessage` with string content;
        - `AIMessage` with string content (an answer);
        - `HumanMessage` with string content (a follow-up question).

        This test should be skipped if the model does not support tool calling
        (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle this
            sequence of messages.

            You can `xfail` the test if tool calling is implemented but this format
            is not supported.

            ```python
            @pytest.mark.xfail(reason=("Not implemented."))
            def test_structured_few_shot_examples(self, *args: Any) -> None:
                super().test_structured_few_shot_examples(*args)
            ```
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        model_with_tools = model.bind_tools([my_adder_tool], tool_choice="any")
        function_result = json.dumps({"result": 3})

        tool_schema = my_adder_tool.args_schema
        assert isinstance(tool_schema, type)
        assert issubclass(tool_schema, BaseModel)
        few_shot_messages = tool_example_to_messages(
            "What is 1 + 2",
            [tool_schema(a=1, b=2)],
            tool_outputs=[function_result],
            ai_response=function_result,
        )

        messages = [*few_shot_messages, HumanMessage("What is 3 + 4")]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    def test_structured_output(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        """Test to verify structured output is generated both on invoke and stream.

        This test is optional and should be skipped if the model does not support
        structured output (see configuration below).

        ??? note "Configuration"

            To disable structured output tests, set `has_structured_output` to `False`
            in your test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_structured_output(self) -> bool:
                    return False
            ```

            By default, `has_structured_output` is `True` if a model overrides the
            `with_structured_output` or `bind_tools` methods.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles both JSON Schema and Pydantic V2 models.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.

        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        schema, validation_function = _get_joke_class(schema_type)
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

        chunk = None
        for chunk in chat.stream(
            "Tell me a joke about cats.", config={"callbacks": [stream_callback]}
        ):
            validation_function(chunk)
        assert chunk is not None, "Stream returned no chunks - possible API issue"

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
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        """Test to verify structured output is generated both on invoke and stream.

        This test is optional and should be skipped if the model does not support
        structured output (see configuration below).

        ??? note "Configuration"

            To disable structured output tests, set `has_structured_output` to `False`
            in your test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_structured_output(self) -> bool:
                    return False
            ```

            By default, `has_structured_output` is `True` if a model overrides the
            `with_structured_output` or `bind_tools` methods.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles both JSON Schema and Pydantic V2 models.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.

        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        schema, validation_function = _get_joke_class(schema_type)

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

        chunk = None
        async for chunk in chat.astream(
            "Tell me a joke about cats.", config={"callbacks": [astream_callback]}
        ):
            validation_function(chunk)
        assert chunk is not None, "Stream returned no chunks - possible API issue"

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
        """Test structured output using pydantic.v1.BaseModel.

        Verify we can generate structured output using `pydantic.v1.BaseModel`.

        `pydantic.v1.BaseModel` is available in the Pydantic 2 package.

        This test is optional and should be skipped if the model does not support
        structured output (see configuration below).

        ??? note "Configuration"

            To disable structured output tests, set `has_structured_output` to `False`
            in your test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_structured_output(self) -> bool:
                    return False
            ```

            By default, `has_structured_output` is `True` if a model overrides the
            `with_structured_output` or `bind_tools` methods.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles both JSON Schema and Pydantic V1 models.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.

        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        class Joke(BaseModelV1):  # Uses langchain_core.pydantic_v1.BaseModel
            """Joke to tell user."""

            setup: str = FieldV1(description="question to set up a joke")
            punchline: str = FieldV1(description="answer to resolve the joke")

        # Pydantic class
        # Note: with_structured_output return type is dict | pydantic.BaseModel (v2),
        # but this test validates pydantic.v1.BaseModel support at runtime.
        chat = model.with_structured_output(Joke, **self.structured_output_kwargs)
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, Joke)  # type: ignore[unreachable]

        chunk = None  # type: ignore[unreachable]
        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, Joke)
        assert chunk is not None, "Stream returned no chunks - possible API issue"

        # Schema
        chat = model.with_structured_output(
            Joke.schema(), **self.structured_output_kwargs
        )
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        chunk = None
        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, dict)
        assert chunk is not None, "Stream returned no chunks - possible API issue"
        assert set(chunk.keys()) == {"setup", "punchline"}

    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        """Test structured output with optional parameters.

        Test to verify we can generate structured output that includes optional
        parameters.

        This test is optional and should be skipped if the model does not support
        structured output (see configuration below).

        ??? note "Configuration"

            To disable structured output tests, set `has_structured_output` to `False`
            in your test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_structured_output(self) -> bool:
                    return False
            ```

            By default, `has_structured_output` is True if a model overrides the
            `with_structured_output` or `bind_tools` methods.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles Pydantic V2 models with optional parameters.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.

        """
        if not self.has_structured_output:
            pytest.skip("Test requires structured output.")

        # Pydantic
        class Joke(BaseModel):
            """Joke to tell user."""

            setup: str = Field(description="question to set up a joke")
            punchline: str | None = Field(
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
            punchline: Annotated[str | None, None, "answer to resolve the joke"]

        chat = model.with_structured_output(JokeDict, **self.structured_output_kwargs)
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)

    def test_json_mode(self, model: BaseChatModel) -> None:
        """Test [structured output]((https://docs.langchain.com/oss/python/langchain/structured-output)) via JSON mode.

        This test is optional and should be skipped if the model does not support
        the JSON mode feature (see configuration below).

        ??? note "Configuration"

            To disable this test, set `supports_json_mode` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_json_mode(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            See example implementation of `with_structured_output` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output

        """  # noqa: E501
        if not self.supports_json_mode:
            pytest.skip("Test requires json mode support.")

        from pydantic import BaseModel as BaseModelProper  # noqa: PLC0415
        from pydantic import Field as FieldProper  # noqa: PLC0415

        class Joke(BaseModelProper):
            """Joke to tell user."""

            setup: str = FieldProper(description="question to set up a joke")
            punchline: str = FieldProper(description="answer to resolve the joke")

        # Pydantic class
        chat = model.with_structured_output(Joke, method="json_mode")
        msg = (
            "Tell me a joke about cats. Return the result as a JSON with 'setup' and "
            "'punchline' keys. Return nothing other than JSON."
        )
        result = chat.invoke(msg)
        assert isinstance(result, Joke)

        chunk = None
        for chunk in chat.stream(msg):
            assert isinstance(chunk, Joke)
        assert chunk is not None, "Stream returned no chunks - possible API issue"

        # Schema
        chat = model.with_structured_output(
            Joke.model_json_schema(), method="json_mode"
        )
        result = chat.invoke(msg)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        chunk = None
        for chunk in chat.stream(msg):
            assert isinstance(chunk, dict)
        assert chunk is not None, "Stream returned no chunks - possible API issue"
        assert set(chunk.keys()) == {"setup", "punchline"}

    def test_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs.

        This test should be skipped (see configuration below) if the model does not
        support PDF inputs. These will take the shape of the LangChain
        `FileContentBlock`:

        ```python
        {
            "type": "image",
            "base64": "<base64 image data>",
            "mime_type": "application/pdf",
        }
        ```

        Furthermore, for backward-compatibility, we must also support OpenAI chat
        completions file content blocks:

        ```python
        (
            {
                "type": "file",
                "file": {
                    "filename": "test_file.pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_data}",
                },
            },
        )
        ```

        ??? note "Configuration"

            To disable this test, set `supports_pdf_inputs` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_pdf_inputs(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle messages
            with pdf content blocks, including base64-encoded files. Otherwise, set
            the `supports_pdf_inputs` property to `False`.

        """
        if not self.supports_pdf_inputs:
            pytest.skip("Model does not support PDF inputs.")

        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        message = HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "file",
                    "base64": pdf_data,
                    "mime_type": "application/pdf",
                },
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Chat Completions format
        message = HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "test_file.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_data}",
                    },
                },
            ]
        )
        _ = model.invoke([message])

    def test_audio_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process audio inputs.

        This test should be skipped (see configuration below) if the model does not
        support audio inputs. These will take the shape of the LangChain
        `AudioContentBlock`:

        ```python
        {
            "type": "audio",
            "base64": "<base64 audio data>",
            "mime_type": "audio/wav",  # or appropriate MIME type
        }
        ```

        Furthermore, for backward-compatibility, we must also support OpenAI chat
        completions audio content blocks:

        ```python
        {
            "type": "input_audio",
            "input_audio": {
                "data": "<base64 audio data>",
                "format": "wav",  # or appropriate format
            },
        }
        ```

        Note: this test downloads audio data from wikimedia.org. You may need to set
        the `LANGCHAIN_TESTS_USER_AGENT` environment variable to identify these
        requests, e.g.,

        ```bash
        export LANGCHAIN_TESTS_USER_AGENT="CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
        ```

        Refer to the [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

        ??? note "Configuration"

            To disable this test, set `supports_audio_inputs` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_audio_inputs(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle messages
            with audio content blocks, specifically base64-encoded files. Otherwise,
            set the `supports_audio_inputs` property to `False`.

        """  # noqa: E501
        if not self.supports_audio_inputs:
            pytest.skip("Model does not support audio inputs.")

        # https://commons.wikimedia.org/wiki/File:Northern_Flicker_202280456.wav
        # License: CC0 1.0 Universal
        url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/Northern_Flicker_202280456.wav"
        audio_data = _get_base64_from_url(url)

        message = HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Describe this audio:",
                },
                {
                    "type": "audio",
                    "mime_type": "audio/wav",
                    "base64": audio_data,
                },
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Chat Completions format
        message = HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Describe this audio:",
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ]
        )
        _ = model.invoke([message])

    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process image inputs.

        This test should be skipped (see configuration below) if the model does not
        support image inputs. These will take the shape of the LangChain
        `ImageContentBlock`:

        ```python
        {
            "type": "image",
            "base64": "<base64 image data>",
            "mime_type": "image/jpeg",  # or appropriate MIME type
        }
        ```

        For backward-compatibility, we must also support OpenAI chat completions
        image content blocks containing base64-encoded images:

        ```python
        [
            {"type": "text", "text": "describe the weather in this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
        ```

        See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

        If the property `supports_image_urls` is set to `True`, the test will also
        check that we can process content blocks of the form:

        ```python
        {
            "type": "image",
            "url": "<url>",
        }
        ```

        ??? note "Configuration"

            To disable this test, set `supports_image_inputs` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_image_inputs(self) -> bool:
                    return False

                # Can also explicitly disable testing image URLs:
                @property
                def supports_image_urls(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle messages
            with image content blocks, including base64-encoded images. Otherwise, set
            the `supports_image_inputs` property to `False`.

        """
        if not self.supports_image_inputs:
            pytest.skip("Model does not support image message.")

        image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        # OpenAI CC format, base64 data
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Give a concise description of this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        )
        _ = model.invoke([message])

        # Standard LangChain format, base64 data
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Give a concise description of this image."},
                {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/png",
                },
            ],
        )
        _ = model.invoke([message])

        # Standard format, URL
        if self.supports_image_urls:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Give a concise description of this image.",
                    },
                    {
                        "type": "image",
                        "url": image_url,
                    },
                ],
            )
            _ = model.invoke([message])

    def test_image_tool_message(self, model: BaseChatModel) -> None:
        """Test that the model can process `ToolMessage` objects with image inputs.

        This test should be skipped if the model does not support messages of the
        Chat Completions `image_url` format:

        ```python
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
        ```

        In addition, models should support the standard LangChain `ImageContentBlock`
        format:

        ```python
        ToolMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/jpeg",
                },
            ],
            tool_call_id="1",
            name="random_image",
        )
        ```

        This test can be skipped by setting the `supports_image_tool_message` property
        to `False` (see configuration below).

        ??? note "Configuration"

            To disable this test, set `supports_image_tool_message` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_image_tool_message(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle messages
            with image content blocks in `ToolMessage` objects, including base64-encoded
            images. Otherwise, set the `supports_image_tool_message` property to
            `False`.

        """
        if not self.supports_image_tool_message:
            pytest.skip("Model does not support image tool message.")

        image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        # OpenAI CC format, base64 data
        oai_format_message = ToolMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
            tool_call_id="1",
            name="random_image",
        )

        # Standard LangChain format, base64 data
        standard_format_message = ToolMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/png",
                },
            ],
            tool_call_id="1",
            name="random_image",
        )

        for tool_message in [oai_format_message, standard_format_message]:
            messages = [
                HumanMessage(
                    "get a random diagram using the tool and give it a concise "
                    "description"
                ),
                AIMessage(
                    [],
                    tool_calls=[
                        {
                            "type": "tool_call",
                            "id": "1",
                            "name": "random_image",
                            "args": {},
                        }
                    ],
                ),
                tool_message,
            ]

            def random_image() -> str:
                """Return a random image."""
                return ""

            _ = model.bind_tools([random_image]).invoke(messages)

    def test_pdf_tool_message(self, model: BaseChatModel) -> None:
        """Test that the model can process `ToolMessage` objects with PDF inputs.

        This test should be skipped if the model does not support messages of the
        LangChain `FileContentBlock` format:

        ```python
        ToolMessage(
            content=[
                {
                    "type": "file",
                    "base64": pdf_data,
                    "mime_type": "application/pdf",
                },
            ],
            tool_call_id="1",
            name="random_pdf",
        )
        ```

        This test can be skipped by setting the `supports_pdf_tool_message` property
        to `False` (see configuration below).

        ??? note "Configuration"

            To disable this test, set `supports_pdf_tool_message` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_pdf_tool_message(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that the model can correctly handle messages
            with PDF content blocks in `ToolMessage` objects, specifically
            base64-encoded PDFs. Otherwise, set the `supports_pdf_tool_message` property
            to `False`.
        """
        if not self.supports_pdf_tool_message:
            pytest.skip("Model does not support PDF tool message.")

        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        tool_message = ToolMessage(
            content_blocks=[
                {
                    "type": "file",
                    "base64": pdf_data,
                    "mime_type": "application/pdf",
                },
            ],
            tool_call_id="1",
            name="random_pdf",
        )

        messages = [
            HumanMessage(
                "Get a random PDF using the tool and relay the title verbatim."
            ),
            AIMessage(
                [],
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "1",
                        "name": "random_pdf",
                        "args": {},
                    }
                ],
            ),
            tool_message,
        ]

        def random_pdf() -> str:
            """Return a random PDF."""
            return ""

        _ = model.bind_tools([random_pdf]).invoke(messages)

    def test_anthropic_inputs(self, model: BaseChatModel) -> None:
        """Test that model can process Anthropic-style message histories.

        These message histories will include `AIMessage` objects with `tool_use`
        content blocks, e.g.,

        ```python
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
        ```

        ...as well as `HumanMessage` objects containing `tool_result` content blocks:

        ```python
        HumanMessage(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "foo",
                    "content": [
                        {
                            "type": "text",
                            "text": "green is a great pick! "
                            "that's my sister's favorite color",
                        }
                    ],
                    "is_error": False,
                },
                {"type": "text", "text": "what's my sister's favorite color"},
            ]
        )
        ```

        This test should be skipped if the model does not support messages of this
        form (or doesn't support tool calling generally). See Configuration below.

        ??? note "Configuration"

            To disable this test, set `supports_anthropic_inputs` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def supports_anthropic_inputs(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that:

            1. The model can correctly handle message histories that include message
                objects with list content.
            2. The `tool_calls` attribute on AIMessage objects is correctly handled
                and passed to the model in an appropriate format.
            3. `HumanMessage`s with "tool_result" content blocks are correctly
                handled.

            Otherwise, if Anthropic tool call and result formats are not supported,
            set the `supports_anthropic_inputs` property to `False`.

        """
        if not self.supports_anthropic_inputs:
            pytest.skip("Model does not explicitly support Anthropic inputs.")

        # Anthropic-format tool
        color_picker = {
            "name": "color_picker",
            "input_schema": {
                "type": "object",
                "properties": {
                    "fav_color": {"type": "string"},
                },
                "required": ["fav_color"],
            },
            "description": "Input your fav color and get a random fact about it.",
            "cache_control": {"type": "ephemeral"},
        }

        human_content = [
            {
                "type": "text",
                "text": "what's your favorite color in this image",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        if self.supports_image_inputs:
            image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            human_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
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
                        "input": {"fav_color": "purple"},
                        "id": "foo",
                        "name": "color_picker",
                    },
                ],
                tool_calls=[
                    {
                        "name": "color_picker",
                        "args": {"fav_color": "purple"},
                        "id": "foo",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage("That's a great pick!", tool_call_id="foo"),
        ]
        response = model.bind_tools([color_picker]).invoke(messages)
        assert isinstance(response, AIMessage)

        # Test thinking blocks
        messages = [
            HumanMessage(
                [
                    {
                        "type": "text",
                        "text": "Hello",
                    },
                ]
            ),
            AIMessage(
                [
                    {
                        "type": "thinking",
                        "thinking": (
                            "This is a simple greeting. I should respond warmly and "
                            "professionally, and perhaps ask how I can help the person "
                            "today."
                        ),
                        "signature": (
                            "ErUBCkYICBgCIkDCTQUXPc3O7nHXd302Zercaz8WrrpddpOqHITxBih5ze"
                            "FPoJkwKBvkvZ8ID1aAfJftji6+ZI5gBYDo7XmNBIkzEgzVDHKopedAn/sc"
                            "G80aDFDXVZrDOWgla7lEBiIwLq5kfFjQjvF/CyuL8J5V7dRwsJN5gQIXaM"
                            "B6xXTs6T+2Zp0VdiyiMb/hcdrHt+7aKh0z2E1UnjiOCoTlofNFHzOnKk0q"
                            "PIoPmfGgpPgGNRgC"
                        ),
                    },
                    {
                        "type": "text",
                        "text": "Hello, how are you?",
                    },
                ]
            ),
            HumanMessage(
                [
                    {
                        "type": "text",
                        "text": "Well, thanks.",
                    },
                ]
            ),
        ]
        response = model.invoke(messages)
        assert isinstance(response, AIMessage)

    def test_message_with_name(self, model: BaseChatModel) -> None:
        """Test that `HumanMessage` with values for the `name` field can be handled.

        These messages may take the form:

        ```python
        HumanMessage("hello", name="example_user")
        ```

        If possible, the `name` field should be parsed and passed appropriately
        to the model. Otherwise, it should be ignored.

        ??? question "Troubleshooting"

            If this test fails, check that the `name` field on `HumanMessage`
            objects is either ignored or passed to the model appropriately.

        """
        result = model.invoke([HumanMessage("hello", name="example_user")])
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.text, str)
        assert len(result.content) > 0

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_agent_loop(self, model: BaseChatModel) -> None:
        """Test that the model supports a simple ReAct agent loop.

        This test is skipped if the `has_tool_calling` property on the test class is
        set to `False`.

        This test is optional and should be skipped if the model does not support
        tool calling (see configuration below).

        ??? note "Configuration"

            To disable tool calling tests, set `has_tool_calling` to `False` in your
            test class:

            ```python
            class TestMyChatModelIntegration(ChatModelIntegrationTests):
                @property
                def has_tool_calling(self) -> bool:
                    return False
            ```

        ??? question "Troubleshooting"

            If this test fails, check that `bind_tools` is implemented to correctly
            translate LangChain tool objects into the appropriate schema for your
            chat model.

            Check also that all required information (e.g., tool calling identifiers)
            from `AIMessage` objects is propagated correctly to model payloads.

            This test may fail if the chat model does not consistently generate tool
            calls in response to an appropriate query. In these cases you can `xfail`
            the test:

            ```python
            @pytest.mark.xfail(reason=("Does not support tool_choice."))
            def test_agent_loop(self, model: BaseChatModel) -> None:
                super().test_agent_loop(model)
            ```

        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        @tool
        def get_weather(location: str) -> str:  # noqa: ARG001
            """Get the weather at a location."""
            return "It's sunny."

        llm_with_tools = model.bind_tools([get_weather])
        input_message = HumanMessage("What is the weather in San Francisco, CA?")
        tool_call_message = llm_with_tools.invoke([input_message])
        assert isinstance(tool_call_message, AIMessage)
        content_blocks = tool_call_message.content_blocks
        assert any(block["type"] == "tool_call" for block in content_blocks)
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

        See `enable_vcr_tests` dropdown `above <ChatModelIntegrationTests>`
        for more information.

        ??? note "Configuration"

            This test can be enabled or disabled using the `enable_vcr_tests`
            property. For example, to disable the test, set this property to `False`:

            ```python
            @property
            def enable_vcr_tests(self) -> bool:
                return False
            ```

            !!! warning
                VCR will by default record authentication headers and other sensitive
                information in cassettes. See `enable_vcr_tests` dropdown
                `above <ChatModelIntegrationTests>` for how to configure what
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
        """Invoke with audio input."""
        raise NotImplementedError

    def invoke_with_audio_output(self, *, stream: bool = False) -> AIMessage:
        """Invoke with audio output."""
        raise NotImplementedError

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """Invoke with reasoning output."""
        raise NotImplementedError

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        """Invoke with cache read input."""
        raise NotImplementedError

    def invoke_with_cache_creation_input(self, *, stream: bool = False) -> AIMessage:
        """Invoke with cache creation input."""
        raise NotImplementedError

    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: str | None = None,
        force_tool_call: bool = True,
    ) -> None:
        r"""Generic integration test for Unicode characters in tool calls.

        Args:
            model: The chat model to test
            tool_choice: Tool choice parameter to pass to `bind_tools()`
                (provider-specific)
            force_tool_call: Whether to force a tool call
                (use `tool_choice=True` if None)

        Tests that Unicode characters in tool call arguments are preserved correctly,
        not escaped as `\\uXXXX` sequences.

        """
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
