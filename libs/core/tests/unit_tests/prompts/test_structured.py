from functools import partial
from inspect import isclass
from typing import Any, Union, cast

import pytest
from pydantic import BaseModel

from langchain_core.language_models import FakeListChatModel
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.messages import HumanMessage
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.utils.mustache import ChevronError
from langchain_core.utils.pydantic import is_basemodel_subclass


def _fake_runnable(
    _: Any, *, schema: Union[dict, type[BaseModel]], value: Any = 42, **_kwargs: Any
) -> Union[BaseModel, dict]:
    if isclass(schema) and is_basemodel_subclass(schema):
        return schema(name="yo", value=value)
    params = cast("dict", schema)["parameters"]
    return {k: 1 if k != "value" else value for k, v in params.items()}


class FakeStructuredChatModel(FakeListChatModel):
    """Fake ChatModel for testing purposes."""

    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> Runnable:
        return RunnableLambda(partial(_fake_runnable, schema=schema, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


def test_structured_prompt_pydantic() -> None:
    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [
            ("human", "I'm very structured, how about you?"),
        ],
        OutputSchema,
    )

    model = FakeStructuredChatModel(responses=[])

    chain = prompt | model

    assert chain.invoke({"hello": "there"}) == OutputSchema(name="yo", value=42)  # type: ignore[comparison-overlap]


def test_structured_prompt_dict() -> None:
    prompt = StructuredPrompt(
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

    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 42}  # type: ignore[comparison-overlap]

    assert loads(dumps(prompt)).model_dump() == prompt.model_dump()

    chain = loads(dumps(prompt)) | model

    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 42}


def test_structured_prompt_kwargs() -> None:
    prompt = StructuredPrompt(
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
        value=7,
    )
    model = FakeStructuredChatModel(responses=[])
    chain = prompt | model
    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 7}  # type: ignore[comparison-overlap]
    assert loads(dumps(prompt)).model_dump() == prompt.model_dump()
    chain = loads(dumps(prompt)) | model
    assert chain.invoke({"hello": "there"}) == {"name": 1, "value": 7}

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [("human", "I'm very structured, how about you?")], OutputSchema, value=7
    )

    model = FakeStructuredChatModel(responses=[])

    chain = prompt | model

    assert chain.invoke({"hello": "there"}) == OutputSchema(name="yo", value=7)  # type: ignore[comparison-overlap]


def test_structured_prompt_template_format() -> None:
    prompt = StructuredPrompt(
        [("human", "hi {{person.name}}")], schema={}, template_format="mustache"
    )
    assert prompt.messages[0].prompt.template_format == "mustache"  # type: ignore[union-attr, union-attr]
    assert prompt.input_variables == ["person"]
    assert prompt.invoke({"person": {"name": "foo"}}).to_messages() == [
        HumanMessage("hi foo")
    ]


def test_structured_prompt_template_empty_vars() -> None:
    with pytest.raises(ChevronError, match="empty tag"):
        StructuredPrompt([("human", "hi {{}}")], schema={}, template_format="mustache")
