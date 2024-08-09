"""Standard LangChain interface tests"""
# requires either passing login information as environment variable,
# or passing into the chat_model_params

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_snowflake import ChatSnowflakeCortex

class TestChatSnowflakeCortex(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatSnowflakeCortex
    
    @property
    def chat_model_params(self) -> dict:
        return {
                "model": "mistral-large",
                "cortex_function": "complete",
        }
