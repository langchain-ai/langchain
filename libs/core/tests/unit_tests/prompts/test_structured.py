from ast import Dict
from functools import partial
from typing import Type, Union

from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableLambda
from tests.unit_tests.fake.chat_model import FakeListChatModel


def _fake_runnable(schema: Type[BaseModel], _) -> BaseModel:
    return schema(name="yo", value=42)


class FakeStructuredChatModel(FakeListChatModel):
    """Fake ChatModel for testing purposes."""

    def with_structured_output(self, schema: Union[Dict, Type[BaseModel]]) -> Runnable:
        return RunnableLambda(partial(_fake_runnable, schema))

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


def test_structured_prompt() -> None:
    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt.from_messages_and_schema(
        [
            ("human", "I'm very structured, how about you?"),
        ],
        OutputSchema,
    )

    model = FakeStructuredChatModel(responses=[])

    chain = prompt | model

    assert chain.invoke({"hello": "there"}) == OutputSchema(name="yo", value=42)
