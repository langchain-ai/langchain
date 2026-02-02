import pytest
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


class DummyChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def bind_tools(self, tools, **kwargs) -> Runnable:
        return self

    def _generate(self, messages, stop=None, **kwargs):
        raise RuntimeError("Should never be called")

    async def _agenerate(self, messages, stop=None, **kwargs):
        raise RuntimeError("Should never be called")


class MyStatus(Enum):
    TODO = "To Do"
    SUCCESSFUL = "Successful"
    FAILED = "Failed"


def test_with_structured_output_rejects_enum_schema():
    model = DummyChatModel()

    with pytest.raises(ValueError) as exc_info:
        model.with_structured_output(MyStatus)

    assert "Enum is not a valid schema" in str(exc_info.value)
