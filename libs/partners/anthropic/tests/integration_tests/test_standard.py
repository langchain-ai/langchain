"""Standard LangChain interface tests."""

from pathlib import Path
from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_anthropic import ChatAnthropic

REPO_ROOT_DIR = Path(__file__).parents[5]


class TestAnthropicStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatAnthropic

    @property
    def chat_model_params(self) -> dict:
        return {"model": "claude-3-5-sonnet-latest"}

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
            model="claude-3-5-sonnet-20240620",  # type: ignore[call-arg]
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},  # type: ignore[call-arg]
        )
        with open(REPO_ROOT_DIR / "README.md") as f:
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
            model="claude-3-5-sonnet-20240620",  # type: ignore[call-arg]
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},  # type: ignore[call-arg]
        )
        with open(REPO_ROOT_DIR / "README.md") as f:
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
            full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
        return cast(AIMessage, full)
    return cast(AIMessage, llm.invoke(input_))
