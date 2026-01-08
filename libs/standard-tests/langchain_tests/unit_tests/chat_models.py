"""Chat model unit tests."""

from __future__ import annotations

import inspect
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, SecretStr, ValidationError

from langchain_tests.base import BaseStandardTests

if TYPE_CHECKING:
    from pytest_benchmark.fixture import (
        BenchmarkFixture,
    )
    from syrupy.assertion import SnapshotAssertion


def generate_schema_pydantic() -> Any:
    """Works with either pydantic 1 or 2."""

    class PersonA(BaseModel):
        """Record attributes of a person."""

        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")

    return PersonA


TEST_PYDANTIC_MODELS = [generate_schema_pydantic()]


class ChatModelTests(BaseStandardTests):
    """Base class for chat model tests."""

    @property
    @abstractmethod
    def chat_model_class(self) -> type[BaseChatModel]:
        """The chat model class to test, e.g., `ChatParrotLink`."""
        ...

    @property
    def chat_model_params(self) -> dict[str, Any]:
        """Initialization parameters for the chat model."""
        return {}

    @property
    def standard_chat_model_params(self) -> dict[str, Any]:
        """Standard chat model parameters."""
        return {
            "temperature": 0,
            "max_tokens": 100,
            "timeout": 60,
            "stop": [],
            "max_retries": 2,
        }

    @pytest.fixture
    def model(self, request: Any) -> BaseChatModel:
        """Model fixture."""
        extra_init_params = getattr(request, "param", None) or {}
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                **extra_init_params,
            },
        )

    @pytest.fixture
    def my_adder_tool(self) -> BaseTool:
        """Adder tool fixture."""

        @tool
        def my_adder_tool(a: int, b: int) -> int:
            """Tool that adds two integers.

            Takes two integers, a and b, and returns their sum.
            """
            return a + b

        return my_adder_tool

    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool calling."""
        return self.chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @property
    def has_tool_choice(self) -> bool:
        """Whether the model supports tool calling."""
        bind_tools_params = inspect.signature(
            self.chat_model_class.bind_tools
        ).parameters
        return "tool_choice" in bind_tools_params

    @property
    def has_structured_output(self) -> bool:
        """Whether the chat model supports structured output."""
        return (
            self.chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        ) or self.has_tool_calling

    @property
    def structured_output_kwargs(self) -> dict[str, Any]:
        """Additional kwargs to pass to `with_structured_output()` in tests.

        Override this property to customize how structured output is generated
        for your model. The most common use case is specifying the `method`
        parameter, which controls the mechanism used to enforce structured output:

        - `'function_calling'`: Uses tool/function calling to enforce the schema.
        - `'json_mode'`: Uses the model's JSON mode.
        - `'json_schema'`: Uses native JSON schema support (e.g., OpenAI's
            structured outputs).

        Returns:
            A dict of kwargs passed to `with_structured_output()`.

        Example:
            ```python
            @property
            def structured_output_kwargs(self) -> dict:
                return {"method": "json_schema"}
            ```
        """
        return {}

    @property
    def supports_json_mode(self) -> bool:
        """Whether the chat model supports JSON mode."""
        return False

    @property
    def supports_image_inputs(self) -> bool:
        """Supports image inputs.

        Whether the chat model supports image inputs, defaults to
        `False`.

        """
        return False

    @property
    def supports_image_urls(self) -> bool:
        """Supports image inputs from URLs.

        Whether the chat model supports image inputs from URLs, defaults to
        `False`.

        """
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        """Whether the chat model supports PDF inputs, defaults to `False`."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Supports audio inputs.

        Whether the chat model supports audio inputs, defaults to `False`.

        """
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Supports video inputs.

        Whether the chat model supports video inputs, defaults to `False`.

        No current tests are written for this feature.
        """
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Returns usage metadata.

        Whether the chat model returns usage metadata on invoke and streaming
        responses.

        """
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the chat model supports Anthropic-style inputs."""
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """Supports image `ToolMessage` objects.

        Whether the chat model supports `ToolMessage` objects that include image
        content.
        """
        return False

    @property
    def supports_pdf_tool_message(self) -> bool:
        """Supports PDF `ToolMessage` objects.

        Whether the chat model supports `ToolMessage` objects that include PDF
        content.
        """
        return False

    @property
    def enable_vcr_tests(self) -> bool:
        """Whether to enable VCR tests for the chat model.

        !!! warning
            See `enable_vcr_tests` dropdown `above <ChatModelTests>` for more
            information.
        """
        return False

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """Supported usage metadata details.

        What usage metadata details are emitted in invoke and stream. Only needs to be
        overridden if these details are returned by the model.
        """
        return {"invoke": [], "stream": []}

    @property
    def supports_model_override(self) -> bool:
        """Whether the model supports overriding the model name at runtime.

        Defaults to `True`.

        If `True`, the model accepts a `model` kwarg in `invoke()`, `stream()`,
        etc. that overrides the model specified at initialization.

        This enables dynamic model selection without creating new instances.
        """
        return True

    @property
    def model_override_value(self) -> str | None:
        """Alternative model name to use when testing model override.

        Should return a valid model name that differs from the default model.
        Required if `supports_model_override` is `True`.
        """
        return None


class ChatModelUnitTests(ChatModelTests):
    '''Base class for chat model unit tests.

    Test subclasses must implement the `chat_model_class` and
    `chat_model_params` properties to specify what model to test and its
    initialization parameters.

    ```python
    from typing import Type

    from langchain_tests.unit_tests import ChatModelUnitTests
    from my_package.chat_models import MyChatModel


    class TestMyChatModelUnit(ChatModelUnitTests):
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

        By default, this is determined by whether the chat model overrides the
        `with_structured_output` or `bind_tools` methods. If the base
        implementations are intended to be used, this method should be overridden.

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

        JSON mode constrains the model to output valid JSON without enforcing
        a specific schema (unlike `'function_calling'` or `'json_schema'` methods).

        When using JSON mode, you must prompt the model to output JSON in your
        message.

        Example:
            ```python
            structured_llm = llm.with_structured_output(MySchema, method="json_mode")
            structured_llm.invoke("... Return the result as JSON.")
            ```

        See docs for [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

        Defaults to `False`.

        ```python
        @property
        def supports_json_mode(self) -> bool:
            return True
        ```

    ??? info "`supports_image_inputs`"

        Boolean property indicating whether the chat model supports image inputs.

        Defaults to `False`.

        If set to `True`, the chat model will be tested using the LangChain
        `ImageContentBlock` format:

        ```python
        {
            "type": "image",
            "base64": "<base64 image data>",
            "mime_type": "image/jpeg",  # or appropriate MIME type
        }
        ```

        In addition to OpenAI Chat Completions `image_url` blocks:

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
        form.

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

        (OpenAI Chat Completions format), as well as LangChain's `ImageContentBlock`
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

        (standard format).

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

        If set to `True`, the chat model will be tested using the LangChain
        `FileContentBlock` format:

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

        using LangChain's `FileContentBlock` format.

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

        If set to `True`, the chat model will be tested using the LangChain
        `AudioContentBlock` format:

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

        Models supporting `usage_metadata` should also return the name of the
        underlying model in the `response_metadata` of the `AIMessage`.

    ??? info "`supports_anthropic_inputs`"

        Boolean property indicating whether the chat model supports Anthropic-style
        inputs.

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
            return False
        ```

    ??? info "`supported_usage_metadata_details`"

        Property controlling what usage metadata details are emitted in both `invoke`
        and `stream`.

        `usage_metadata` is an optional dict attribute on `AIMessage` objects that track
        input and output tokens.

        [See more](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.ai.UsageMetadata).

        It includes optional keys `input_token_details` and `output_token_details`
        that can track usage details associated with special types of tokens, such as
        cached, audio, or reasoning.

        Only needs to be overridden if these details are supplied.

    ??? info "`supports_model_override`"

        Boolean property indicating whether the chat model supports overriding the
        model name at runtime via kwargs.

        If `True`, the model accepts a `model` kwarg in `invoke()`, `stream()`, etc.
        that overrides the model specified at initialization. This enables dynamic
        model selection without creating new chat model instances.

        Defaults to `False`.

        ```python
        @property
        def supports_model_override(self) -> bool:
            return True
        ```

    ??? info "`model_override_value`"

        Alternative model name to use when testing model override.

        Should return a valid model name that differs from the default model.
        Required if `supports_model_override` is `True`.

        ```python
        @property
        def model_override_value(self) -> str:
            return "gpt-4o-mini"  # e.g. if default is "gpt-4o"
        ```

    ??? info "`enable_vcr_tests`"

        Property controlling whether to enable select tests that rely on
        [VCR](https://vcrpy.readthedocs.io/en/latest/) caching of HTTP calls, such
        as benchmarking tests.

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

    **Testing initialization from environment variables**

    Some unit tests may require testing initialization from environment variables.
    These tests can be enabled by overriding the `init_from_env_params`
    property (see below).

    ??? info "`init_from_env_params`"

        This property is used in unit tests to test initialization from
        environment variables. It should return a tuple of three dictionaries
        that specify the environment variables, additional initialization args,
        and expected instance attributes to check.

        Defaults to empty dicts. If not overridden, the test is skipped.

        Example:
        ```python
        @property
        def init_from_env_params(self) -> Tuple[dict, dict, dict]:
            return (
                {
                    "MY_API_KEY": "api_key",
                },
                {
                    "model": "bird-brain-001",
                },
                {
                    "my_api_key": "api_key",
                },
            )
        ```
    '''  # noqa: E501,D214

    @property
    def standard_chat_model_params(self) -> dict[str, Any]:
        """Standard chat model parameters."""
        params = super().standard_chat_model_params
        params["api_key"] = "test"
        return params

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        """Init from env params.

        Environment variables, additional initialization args, and expected instance
        attributes for testing initialization from environment variables.
        """
        return {}, {}, {}

    def test_init(self) -> None:
        """Test model initialization. This should pass for all integrations.

        ??? question "Troubleshooting"

            If this test fails, ensure that:

            1. `chat_model_params` is specified and the model can be initialized
                from those params;
            2. The model accommodates
                [standard parameters](https://docs.langchain.com/oss/python/langchain/models#parameters).

        """
        model = self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
            }
        )
        assert model is not None

    def test_init_from_env(self) -> None:
        """Test initialization from environment variables.

        Relies on the `init_from_env_params` property. Test is skipped if that
        property is not set.

        ??? question "Troubleshooting"

            If this test fails, ensure that `init_from_env_params` is specified
            correctly and that model parameters are properly set from environment
            variables during initialization.

        """
        env_params, model_params, expected_attrs = self.init_from_env_params
        if not env_params:
            pytest.skip("init_from_env_params not specified.")
        else:
            with mock.patch.dict(os.environ, env_params):
                model = self.chat_model_class(**model_params)
            assert model is not None
            for k, expected in expected_attrs.items():
                actual = getattr(model, k)
                if isinstance(actual, SecretStr):
                    actual = actual.get_secret_value()
                assert actual == expected

    def test_init_streaming(
        self,
    ) -> None:
        """Test that model can be initialized with `streaming=True`.

        This is for backward-compatibility purposes.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model can be initialized with a
            boolean `streaming` parameter.

        """
        model = self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                "streaming": True,
            }
        )
        assert model is not None

    def test_bind_tool_pydantic(
        self,
        model: BaseChatModel,
        my_adder_tool: BaseTool,
    ) -> None:
        """Test bind tools with Pydantic models.

        Test that chat model correctly handles Pydantic models that are passed
        into `bind_tools`. Test is skipped if the `has_tool_calling` property
        on the test class is False.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles Pydantic V2 models.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.
        """
        if not self.has_tool_calling:
            return

        def my_adder(a: int, b: int) -> int:
            """Return the sum of two integers."""
            return a + b

        tools = [my_adder_tool, my_adder]

        for pydantic_model in TEST_PYDANTIC_MODELS:
            model_schema = (
                pydantic_model.model_json_schema()
                if hasattr(pydantic_model, "model_json_schema")
                else pydantic_model.schema()
            )
            tools.extend([pydantic_model, model_schema])

        # Doing a mypy ignore here since some of the tools are from pydantic
        # BaseModel 2 which isn't typed properly yet. This will need to be fixed
        # so type checking does not become annoying to users.
        tool_model = model.bind_tools(tools, tool_choice="any")  # type: ignore[arg-type]
        assert isinstance(tool_model, RunnableBinding)

    @pytest.mark.parametrize("schema", TEST_PYDANTIC_MODELS)
    def test_with_structured_output(
        self,
        model: BaseChatModel,
        schema: Any,
    ) -> None:
        """Test `with_structured_output` method.

        Test is skipped if the `has_structured_output` property on the test class is
        False.

        ??? question "Troubleshooting"

            If this test fails, ensure that the model's `bind_tools` method
            properly handles Pydantic V2 models.

            `langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
            that will accommodate most formats.

            See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
            of `with_structured_output`.
        """
        if not self.has_structured_output:
            return

        assert model.with_structured_output(schema) is not None
        for method in ["json_schema", "function_calling", "json_mode"]:
            strict_values = [None, False, True] if method != "json_mode" else [None]
            for strict in strict_values:
                assert model.with_structured_output(
                    schema, method=method, strict=strict
                )

    def test_standard_params(self, model: BaseChatModel) -> None:
        """Test that model properly generates standard parameters.

        These are used for tracing purposes.

        ??? question "Troubleshooting"

            If this test fails, check that the model accommodates [standard parameters](https://docs.langchain.com/oss/python/langchain/models#parameters).

            Check also that the model class is named according to convention
            (e.g., `ChatProviderName`).
        """

        class ExpectedParams(BaseModel):
            ls_provider: str
            ls_model_name: str
            ls_model_type: Literal["chat"]
            ls_temperature: float | None = None
            ls_max_tokens: int | None = None
            ls_stop: list[str] | None = None

        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationError as e:
            pytest.fail(f"Validation error: {e}")

        # Test optional params
        model = self.chat_model_class(
            max_tokens=10,
            stop=["test"],
            **self.chat_model_params,
        )
        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationError as e:
            pytest.fail(f"Validation error: {e}")

    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        """Test serialization and deserialization of the model.

        Test is skipped if the `is_lc_serializable` property on the chat model class
        is not overwritten to return `True`.

        ??? question "Troubleshooting"

            If this test fails, check that the `init_from_env_params` property is
            correctly set on the test class.
        """
        if not self.chat_model_class.is_lc_serializable():
            pytest.skip("Model is not serializable.")
        else:
            env_params, _model_params, _expected_attrs = self.init_from_env_params
            with mock.patch.dict(os.environ, env_params):
                ser = dumpd(model)
                assert ser == snapshot(name="serialized")
                assert (
                    model.dict()
                    == load(
                        dumpd(model),
                        valid_namespaces=model.get_lc_namespace()[:1],
                        allowed_objects="all",
                        secrets_from_env=True,
                    ).dict()
                )

    @pytest.mark.benchmark
    def test_init_time(self, benchmark: BenchmarkFixture) -> None:
        """Test initialization time of the chat model.

        If this test fails, check that
        we are not introducing undue overhead in the model's initialization.
        """

        def _init_in_loop() -> None:
            for _ in range(10):
                self.chat_model_class(**self.chat_model_params)

        benchmark(_init_in_loop)
