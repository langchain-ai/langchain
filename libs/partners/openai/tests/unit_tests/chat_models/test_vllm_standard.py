"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_standard_tests.unit_tests import ChatModelUnitTests
from langchain_standard_tests.unit_tests.chat_models import Person, my_adder_tool

from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.vllm import VLLMChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return VLLMChatOpenAI
