"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_standard_tests.unit_tests import ChatModelUnitTests
from langchain_standard_tests.unit_tests.chat_models import Person, my_adder_tool

from langchain_openai import AzureChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": "test",
            "openai_api_version": "2021-10-01",
            "azure_endpoint": "https://test.azure.com",
        }

    def test_bind_tool_pydantic(self, model: BaseChatModel) -> None:
        """Does not currently support tool_choice='any'."""
        if not self.has_tool_calling:
            return

        tool_model = model.bind_tools([Person, Person.schema(), my_adder_tool])
        assert isinstance(tool_model, RunnableBinding)
