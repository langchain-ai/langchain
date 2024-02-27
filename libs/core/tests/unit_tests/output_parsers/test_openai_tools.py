import json
from typing import List

import pytest

from langchain_core.messages import AIMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import _PYDANTIC_MAJOR_VERSION, BaseModel


@pytest.fixture
def generations() -> List[Generation]:
    tool_call = {
        "id": "",
        "type": "function",
        "function": {"name": "Foo", "arguments": json.dumps({"bloop": 2})},
    }
    return [
        ChatGeneration(
            message=AIMessage("", additional_kwargs={"tool_calls": [tool_call]})
        )
    ]


@pytest.mark.skipif(
    _PYDANTIC_MAJOR_VERSION != 1,
    reason=f"test only runs on pydantic v1, current version {_PYDANTIC_MAJOR_VERSION}",
)
def test_tools_parser_context_pydantic_v1(generations: List[Generation]) -> None:
    class Foo(BaseModel):
        bloop: int

    parser = PydanticToolsParser(tools=[Foo], context={"baz": "bar"})

    assert parser.parse_result(generations) == [Foo(bloop=2)]


@pytest.mark.skipif(
    _PYDANTIC_MAJOR_VERSION != 2,
    reason=f"test only runs on pydantic v2, current version {_PYDANTIC_MAJOR_VERSION}",
)
def test_tools_parser_context_pydantic_v2(generations: List[Generation]) -> None:
    from pydantic import BaseModel as BaseModelV2
    from pydantic import ValidationInfo, model_validator

    class Foo(BaseModelV2):
        bloop: int

        @model_validator(mode="before")
        def validate_env(cls, values: dict, info: ValidationInfo) -> dict:
            context = info.context or {}
            if context.get("baz") == "bar":
                raise ValueError()
            return values

    parser = PydanticToolsParser(tools=[Foo], context={"baz": "bar"})
    with pytest.raises(ValueError):
        parser.parse_result(generations)
