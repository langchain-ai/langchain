"""Test Perplexity Chat API wrapper."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_perplexity import ChatPerplexity


class TestPerplexityStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatPerplexity

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ({"PPLX_API_KEY": "api_key"}, {}, {"pplx_api_key": "api_key"})
