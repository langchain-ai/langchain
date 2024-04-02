from functools import partial
from inspect import isclass
from typing import Any, Dict, Type, Union, cast

from langchain_core.language_models import FakeListChatModel
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableLambda


def _fake_runnable(
    schema: Union[Dict, Type[BaseModel]], _: Any
) -> Union[BaseModel, Dict]:
    if isclass(schema) and issubclass(schema, BaseModel):
        return schema(name="yo", value=42)
    else:
        params = cast(Dict, schema)["parameters"]
        return {k: 1 for k, v in params.items()}


class FakeStructuredChatModel(FakeListChatModel):
    """Fake ChatModel for testing purposes."""

    def with_structured_output(
        self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any
    ) -> Runnable:
        return RunnableLambda(partial(_fake_runnable, schema))

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


def test_structured_prompt_pydantic() -> None:
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


def test_structured_prompt_dict() -> None:
    prompt = StructuredPrompt.from_messages_and_schema(
        [
            ("human", "I'm very structured, how about you?"),
        ],
        {
            "name": "yo",
            "description": "a structured output",
            "parameters": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        },
    )

    model = FakeStructuredChatModel(responses=[])

    chain = prompt | model

    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 1}

    assert loads(dumps(prompt)) == prompt

    chain = loads(dumps(prompt)) | model

    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 1}
