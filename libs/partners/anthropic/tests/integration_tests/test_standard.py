"""Standard LangChain interface tests."""

from pathlib import Path
from typing import Literal, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_anthropic import ChatAnthropic

REPO_ROOT_DIR = Path(__file__).parents[5]

MODEL = "claude-3-5-haiku-20241022"


class TestAnthropicStandard(ChatModelIntegrationTests):
    """Use standard chat model integration tests against the `ChatAnthropic` class."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatAnthropic

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def enable_vcr_tests(self) -> bool:
        return True

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
        return {
            "invoke": ["cache_read_input", "cache_creation_input"],
            "stream": ["cache_read_input", "cache_creation_input"],
        }

    def invoke_with_cache_creation_input(self, *, stream: bool = False) -> AIMessage:
        llm = ChatAnthropic(
            model=MODEL,  # type: ignore[call-arg]
        )
        with Path.open(REPO_ROOT_DIR / "README.md") as f:
            readme = f.read()

        input_ = f"""What's langchain? Here's the langchain README:

        {readme}
        """
        return _invoke(
            llm,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            stream,
        )

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        llm = ChatAnthropic(
            model=MODEL,  # type: ignore[call-arg]
        )
        with Path.open(REPO_ROOT_DIR / "README.md") as f:
            readme = f.read()

        input_ = f"""What's langchain? Here's the langchain README:

        {readme}
        """

        # invoke twice so first invocation is cached
        _invoke(
            llm,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            stream,
        )
        return _invoke(
            llm,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            stream,
        )


def _invoke(llm: ChatAnthropic, input_: list, stream: bool) -> AIMessage:  # noqa: FBT001
    if stream:
        full = None
        for chunk in llm.stream(input_):
            full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
        return cast("AIMessage", full)
    return cast("AIMessage", llm.invoke(input_))


class NativeStructuredOutputTests(TestAnthropicStandard):
    @property
    def chat_model_params(self) -> dict:
        return {"model": "claude-sonnet-4-5"}

    @property
    def structured_output_kwargs(self) -> dict:
        return {"method": "json_schema"}


@pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
def test_native_structured_output(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> None:
    test_instance = NativeStructuredOutputTests()
    model = test_instance.chat_model_class(**test_instance.chat_model_params)
    NativeStructuredOutputTests().test_structured_output(model, schema_type)


@pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
async def test_native_structured_output_async(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> None:
    test_instance = NativeStructuredOutputTests()
    model = test_instance.chat_model_class(**test_instance.chat_model_params)
    await NativeStructuredOutputTests().test_structured_output_async(model, schema_type)
