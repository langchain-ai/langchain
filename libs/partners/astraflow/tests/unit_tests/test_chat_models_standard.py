"""Standard LangChain interface tests."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_astraflow import ChatAstraflow

MODEL_NAME = "gpt-4o"


class TestAstraflowStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatAstraflow

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "ASTRAFLOW_API_KEY": "api_key",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "astraflow_api_key": "api_key",
                "astraflow_api_base": "https://api-us-ca.umodelverse.ai/v1",
            },
        )
