"""Test chat model integration."""

from langchain_standard_tests.unit_tests import ChatModelUnitTests
from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel

from __module_name__.chat_models import Chat__ModuleName__


class TestChat__ModuleName__StandardUnitTests(ChatModelUnitTests):
    """Standard LangChain interface tests, applied to __ModuleName__."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return chat model class."""
        return Chat__ModuleName__

    @property
    def chat_model_params(self) -> dict:
        # TODO: Update with chat model parameters
        # e.g. return {"model": "claude-3-haiku-20240307"} for ChatAnthropic
        return {}
