""":autodoc-options: autoproperty.

Standard unit tests for chat models supporting v1 messages.

This module provides updated test patterns for the new messages introduced in
``langchain_core.messages.content_blocks``. Notably, this includes the standardized
content blocks system.
"""

import inspect
import os
from abc import abstractmethod
from typing import Any, Literal, Optional
from unittest import mock

import pytest
from langchain_core.load import dumpd, load
from langchain_core.messages.content_blocks import (
    create_text_block,
)
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool, tool
from langchain_core.v1.chat_models import BaseChatModel
from langchain_core.v1.messages import HumanMessage
from pydantic import BaseModel, Field, SecretStr
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]
from syrupy.assertion import SnapshotAssertion

from langchain_tests.base import BaseStandardTests
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


def generate_schema_pydantic_v1_from_2() -> Any:
    """Use to generate a schema from v1 namespace in pydantic 2.

    :private:
    """
    if PYDANTIC_MAJOR_VERSION != 2:
        msg = "This function is only compatible with Pydantic v2."
        raise AssertionError(msg)

    class PersonB(BaseModelV1):
        """Record attributes of a person."""

        name: str = FieldV1(..., description="The name of the person.")
        age: int = FieldV1(..., description="The age of the person.")

    return PersonB


def generate_schema_pydantic() -> Any:
    """Works with either pydantic 1 or 2.

    :private:
    """

    class PersonA(BaseModel):
        """Record attributes of a person."""

        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")

    return PersonA


TEST_PYDANTIC_MODELS = [generate_schema_pydantic()]

if PYDANTIC_MAJOR_VERSION == 2:
    TEST_PYDANTIC_MODELS.append(generate_schema_pydantic_v1_from_2())


class ChatModelV1Tests(BaseStandardTests):
    """Test suite for v1 chat models.

    This class provides comprehensive testing for the new message system introduced in
    LangChain v1, including the standardized content block format.

    :private:
    """

    @property
    @abstractmethod
    def chat_model_class(self) -> type[BaseChatModel]:
        """The chat model class to test, e.g., ``ChatParrotLink``.

        .. important::
            Test subclasses **must** implement this property.

        """
        ...

    @property
    def chat_model_params(self) -> dict:
        """Initialization parameters for the chat model to test.

        .. important::
            Test subclasses **must** implement this property.

        """
        return {}

    @property
    def standard_chat_model_params(self) -> dict:
        """:private:"""
        return {
            "temperature": 0,
            "max_tokens": 100,
            "timeout": 60,
            "stop": [],
            "max_retries": 2,
        }

    @pytest.fixture
    def model(self) -> BaseChatModel:
        """:private:"""
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
            }
        )

    @pytest.fixture
    def my_adder_tool(self) -> BaseTool:
        """:private:"""

        @tool
        def my_adder_tool(a: int, b: int) -> int:
            """Takes two integers, a and b, and returns their sum."""
            return a + b

        return my_adder_tool

    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool calling."""
        return self.chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @property
    def tool_choice_value(self) -> Optional[str]:
        """(None or str) To use for tool choice when used in tests.

        Not required.

        """
        return None

    @property
    def has_tool_choice(self) -> bool:
        """Whether the model supports forcing tool calling via ``tool_choice``."""
        bind_tools_params = inspect.signature(
            self.chat_model_class.bind_tools
        ).parameters
        return "tool_choice" in bind_tools_params

    @property
    def has_structured_output(self) -> bool:
        """Whether the model supports structured output."""
        return (
            self.chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        ) or self.has_tool_calling

    @property
    def structured_output_kwargs(self) -> dict:
        """Additional kwargs for ``with_structured_output``.

        Example: ``{"method": "json_schema", "strict": True}``

        """
        return {}

    @property
    def supports_json_mode(self) -> bool:
        """Whether the model supports JSON mode.

        TODO: clarify what this means exactly.

        Defaults to False.

        """
        return False

    # Content Block Support Properties
    @property
    def supports_content_blocks_v1(self) -> bool:
        """Whether the model supports content blocks v1 format.

        **Defaults to True.**

        .. important::
            This should not be overridden by a ChatV1 subclass.

        You may override the following properties to enable specific content block
        support. Each defaults to False:

        - ``supports_reasoning_content_blocks``
        - ``supports_plaintext_content_blocks``
        - ``supports_file_content_blocks``
        - ``supports_image_content_blocks``
        - ``supports_audio_content_blocks``
        - ``supports_video_content_blocks``
        - ``supports_citations``
        - ``supports_web_search_blocks``
        - ``supports_enhanced_tool_calls``
        - ``supports_invalid_tool_calls``
        - ``supports_tool_call_chunks``

        """
        return True

    @property
    def supports_non_standard_blocks(self) -> bool:
        """Whether the model supports ``NonStandardContentBlock``.

        Defaults to True.

        """
        return True

    @property
    def supports_text_content_blocks(self) -> bool:
        """Whether the model supports ``TextContentBlock``.

        .. important::
            This is a minimum requirement for v1 chat models.

        """
        return self.supports_content_blocks_v1

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """Whether the model supports ``ReasoningContentBlock``.

        Defaults to False.

        """
        return False

    @property
    def supports_plaintext_content_blocks(self) -> bool:
        """Whether the model supports ``PlainTextContentBlock``.

        Defaults to False.

        """
        return False

    @property
    def supports_file_content_blocks(self) -> bool:
        """Whether the model supports ``FileContentBlock``.

        Replaces ``supports_pdf_inputs`` from v0.

        Defaults to False.

        """
        return False

    @property
    def supports_image_content_blocks(self) -> bool:
        """Whether the model supports ``ImageContentBlock``.

        Replaces ``supports_image_inputs`` from v0.

        Defaults to False.

        """
        return False

    @property
    def supports_audio_content_blocks(self) -> bool:
        """Whether the model supports ``AudioContentBlock``.

        Replaces ``supports_audio_inputs`` from v0.

        Defaults to False.

        """
        return False

    @property
    def supports_video_content_blocks(self) -> bool:
        """Whether the model supports ``VideoContentBlock``.

        Replaces ``supports_video_inputs`` from v0.

        Defaults to False.

        """
        return False

    @property
    def supports_multimodal_reasoning(self) -> bool:
        """Whether the model can reason about multimodal content."""
        return (
            self.supports_image_content_blocks
            and self.supports_reasoning_content_blocks
        )

    @property
    def supports_citations(self) -> bool:
        """Whether the model supports ``Citation`` annotations.

        Defaults to False.

        """
        return False

    @property
    def supports_structured_citations(self) -> bool:
        """Whether the model supports structured citation generation."""
        return self.supports_citations

    @property
    def supports_web_search_blocks(self) -> bool:
        """Whether the model supports ``WebSearchCall``/``WebSearchResult`` blocks.

        Defaults to False.

        """
        return False

    @property
    def supports_code_interpreter(self) -> bool:
        """Whether the model supports code interpreter blocks.

        Defaults to False.

        """
        return False

    @property
    def supports_invalid_tool_calls(self) -> bool:
        """Whether the model can handle ``InvalidToolCall`` blocks.

        Defaults to False.

        """
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Whether the model returns usage metadata on invoke and streaming.

        Defaults to True.

        """
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the model supports Anthropic-style inputs.

        Defaults to False.

        """
        return False

    @property
    def enable_vcr_tests(self) -> bool:
        """Whether to enable VCR tests for the chat model.

        .. important::
            See ``enable_vcr_tests`` dropdown :class:`above <ChatModelTests>` for more
            information.

        Defaults to False.

        """
        return False

    # TODO: check this, since there is `reasoning_output` in usage metadata details ?
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
        """What usage metadata details are emitted in ``invoke()`` and ``stream()``."""
        return {"invoke": [], "stream": []}


class ChatModelV1UnitTests(ChatModelV1Tests):
    """Base class for chat model v1 unit tests.

    These tests run in isolation without external dependencies.

    Test subclasses must implement the ``chat_model_class`` and
    ``chat_model_params`` properties to specify what model to test and its
    initialization parameters.

    Example:

    .. code-block:: python

        from typing import Type

        from langchain_tests.unit_tests import ChatModelV1UnitTests
        from my_package.chat_models import MyChatModel


        class TestMyChatModelUnit(ChatModelV1UnitTests):
            @property
            def chat_model_class(self) -> Type[MyChatModel]:
                # Return the chat model class to test here
                return MyChatModel

            @property
            def chat_model_params(self) -> dict:
                # Return initialization parameters for the v1 model.
                return {"model": "model-001", "temperature": 0}

    .. note::
          API references for individual test methods include troubleshooting tips.

    .. important::
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

    Testing initialization from environment variables
        Some unit tests may require testing initialization from environment variables.
        These tests can be enabled by overriding the ``init_from_env_params``
        property (see below):

        .. dropdown:: init_from_env_params

            This property is used in unit tests to test initialization from
            environment variables. It should return a tuple of three dictionaries
            that specify the environment variables, additional initialization args,
            and expected instance attributes to check.

            Defaults to empty dicts. If not overridden, the test is skipped.

            Example:

            .. code-block:: python

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

    """  # noqa: E501

    @property
    def standard_chat_model_params(self) -> dict:
        """:private:"""
        params = super().standard_chat_model_params
        params["api_key"] = "test"
        return params

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Environment variables, additional initialization args, and expected
        instance attributes for testing initialization from environment variables.

        Not required.

        """
        return {}, {}, {}

    # Initialization Tests
    def test_init(self) -> None:
        """Test model initialization. This should pass for all integrations.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that:

            1. ``chat_model_params`` is specified and the model can be initialized from those params;
            2. The model accommodates `standard parameters <https://python.langchain.com/docs/concepts/chat_models/#standard-parameters>`__

        """  # noqa: E501
        model = self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
            }
        )
        assert model is not None

    def test_init_from_env(self) -> None:
        """Test initialization from environment variables. Relies on the
        ``init_from_env_params`` property. Test is skipped if that property is not
        set.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that ``init_from_env_params`` is specified
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
        """Test that model can be initialized with ``streaming=True``. This is for
        backward-compatibility purposes.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model can be initialized with a
            boolean ``streaming`` parameter.

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
        """Test that chat model correctly handles Pydantic models that are passed
        into ``bind_tools``. Test is skipped if the ``has_tool_calling`` property
        on the test class is False.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles Pydantic V2 models. ``langchain_core`` implements
            a utility function that will accommodate most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``bind_tools`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.bind_tools

        """
        if not self.has_tool_calling:
            return

        def my_adder(a: int, b: int) -> int:
            """Takes two integers, a and b, and returns their sum."""
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
        """Test ``with_structured_output`` method. Test is skipped if the
        ``has_structured_output`` property on the test class is False.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that the model's ``bind_tools`` method
            properly handles Pydantic V2 models. ``langchain_core`` implements
            a utility function that will accommodate most formats: https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.convert_to_openai_tool.html

            See example implementation of ``with_structured_output`` here: https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output

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
        """Test that model properly generates standard parameters. These are used
        for tracing purposes.

        .. dropdown:: Troubleshooting

            If this test fails, check that the model accommodates `standard parameters <https://python.langchain.com/docs/concepts/chat_models/#standard-parameters>`__.

            Check also that the model class is named according to convention
            (e.g., ``ChatProviderName``).
        """

        class ExpectedParams(BaseModelV1):
            ls_provider: str
            ls_model_name: str
            ls_model_type: Literal["chat"]
            ls_temperature: Optional[float]
            ls_max_tokens: Optional[int]
            ls_stop: Optional[list[str]]

        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)  # type: ignore[arg-type]
        except ValidationErrorV1 as e:
            pytest.fail(f"Validation error: {e}")

        # Test optional params
        model = self.chat_model_class(
            max_tokens=10,  # type: ignore[call-arg]
            stop=["test"],  # type: ignore[call-arg]
            **self.chat_model_params,
        )
        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)  # type: ignore[arg-type]
        except ValidationErrorV1 as e:
            pytest.fail(f"Validation error: {e}")

    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        """Test serialization and deserialization of the model. Test is skipped if the
        ``is_lc_serializable`` property on the chat model class is not overwritten
        to return ``True``.

        .. dropdown:: Troubleshooting

            If this test fails, check that the ``init_from_env_params`` property is
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
                    model.model_dump()
                    == load(
                        dumpd(model), valid_namespaces=model.get_lc_namespace()[:1]
                    ).model_dump()
                )

    @pytest.mark.benchmark
    def test_init_time(self, benchmark: BenchmarkFixture) -> None:
        """Test initialization time of the chat model. If this test fails, check that
        we are not introducing undue overhead in the model's initialization.
        """

        def _init_in_loop() -> None:
            for _ in range(10):
                self.chat_model_class(**self.chat_model_params)

        benchmark(_init_in_loop)

    # Property Tests
    def test_llm_type_property(self, model: BaseChatModel) -> None:
        """Test that ``_llm_type`` property is implemented and returns a string."""
        llm_type = model._llm_type
        assert isinstance(llm_type, str)
        assert len(llm_type) > 0

    def test_identifying_params_property(self, model: BaseChatModel) -> None:
        """Test that ``_identifying_params`` property returns a mapping."""
        params = model._identifying_params
        assert isinstance(params, dict)  # Should be dict-like mapping

    # Serialization Tests
    def test_dump_serialization(self, model: BaseChatModel) -> None:
        """Test that ``dump()`` returns proper serialization."""
        dumped = model.dump()
        assert isinstance(dumped, dict)
        assert "_type" in dumped
        assert dumped["_type"] == model._llm_type

        # Should contain identifying parameters
        for key, value in model._identifying_params.items():
            assert key in dumped
            assert dumped[key] == value

    def test_content_block_serialization(self, model: BaseChatModel) -> None:
        """Test that messages with content blocks can be serialized/deserialized."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support v1 content blocks.")

        text_block = create_text_block("Test serialization")
        message = HumanMessage(content=[text_block])

        # Test serialization
        serialized = dumpd(message)
        assert isinstance(serialized, dict)

        # Test deserialization
        deserialized = load(serialized)
        assert isinstance(deserialized, HumanMessage)
        assert deserialized.content == message.content
        # TODO: make more robust? include more fields
