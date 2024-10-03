"""Standard LangChain interface tests"""

from typing import Type, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_anthropic import ChatAnthropic


class TestAnthropicStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAnthropic

    @property
    def chat_model_params(self) -> dict:
        return {"model": "claude-3-haiku-20240307"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> List[
        Literal[
            "audio_input",
            "audio_output",
            "reasoning_output",
            "cache_read_input",
            "cache_creation_input",
        ]
    ]:
        return ["cache_read_input", "cache_creation_input"]

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        ...

    def invoke_with_cache_creation_input(self, *, stream: bool = False) -> AIMessage:
        ...
