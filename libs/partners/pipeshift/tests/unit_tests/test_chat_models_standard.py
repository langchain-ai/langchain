"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_pipeshift import ChatPipeshift


class TestPipeshiftStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatPipeshift

    @property
    def chat_model_params(self) -> dict:
        return {"model": "meta-llama/Llama-2-7b-chat-hf"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "PIPESHIFT_API_KEY": "pipeshift_api_secret",
                "PIPESHIFT_API_BASE": "https://api.baseurl.com",
            },
            {},
            {
                "pipeshift_api_key": "pipeshift_api_secret",
                "pipeshift_api_base": "https://api.baseurl.com",
            },
        )
