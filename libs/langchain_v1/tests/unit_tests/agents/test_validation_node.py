from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as dec_tool
from langgraph.prebuilt import ValidationNode
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

pytestmark = pytest.mark.anyio


def my_function(some_val: int, some_other_val: str) -> str:
    return f"{some_val} - {some_other_val}"


class MyModel(BaseModel):
    some_val: int
    some_other_val: str


class MyModelV1(BaseModelV1):
    some_val: int
    some_other_val: str


@dec_tool
def my_tool(some_val: int, some_other_val: str) -> str:
    """Cool."""
    return f"{some_val} - {some_other_val}"


@pytest.mark.parametrize(
    "tool_schema",
    [
        my_function,
        MyModel,
        MyModelV1,
        my_tool,
    ],
)
@pytest.mark.parametrize("use_message_key", [True, False])
async def test_validation_node(tool_schema: Any, use_message_key: bool):
    validation_node = ValidationNode([tool_schema])
    tool_name = getattr(tool_schema, "name", getattr(tool_schema, "__name__", None))
    inputs = [
        AIMessage(
            "hi?",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"some_val": 1, "some_other_val": "foo"},
                    "id": "some 0",
                },
                {
                    "name": tool_name,
                    # Wrong type for some_val
                    "args": {"some_val": "bar", "some_other_val": "foo"},
                    "id": "some 1",
                },
            ],
        ),
    ]
    if use_message_key:
        inputs = {"messages": inputs}
    result = await validation_node.ainvoke(inputs)
    if use_message_key:
        result = result["messages"]

    def check_results(messages: list):
        assert len(messages) == 2
        assert all(m.type == "tool" for m in messages)
        assert not messages[0].additional_kwargs.get("is_error")
        assert messages[1].additional_kwargs.get("is_error")

    check_results(result)
    result_sync = validation_node.invoke(inputs)
    if use_message_key:
        result_sync = result_sync["messages"]
    check_results(result_sync)
