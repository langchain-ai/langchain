import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytest
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseModel, BaseTool, Field

from langchain_cohere.cohere_agent import (
    _format_to_cohere_tools,
    _format_to_cohere_tools_messages,
)

expected_test_tool_definition = {
    "description": "test_tool description",
    "name": "test_tool",
    "parameter_definitions": {
        "arg_1": {
            "description": "Arg1 description",
            "required": True,
            "type": "str",
        },
        "optional_arg_2": {
            "description": "Arg2 description",
            "required": False,
            "type": "str",
        },
        "arg_3": {
            "description": "Arg3 description",
            "required": True,
            "type": "int",
        },
    },
}


class _TestToolSchema(BaseModel):
    arg_1: str = Field(description="Arg1 description")
    optional_arg_2: Optional[str] = Field(description="Arg2 description", default="2")
    arg_3: int = Field(description="Arg3 description")


class _TestTool(BaseTool):
    name = "test_tool"
    description = "test_tool description"
    args_schema: Type[_TestToolSchema] = _TestToolSchema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class test_tool(BaseModel):
    """test_tool description"""

    arg_1: str = Field(description="Arg1 description")
    optional_arg_2: Optional[str] = Field(description="Arg2 description", default="2")
    arg_3: int = Field(description="Arg3 description")


test_tool_as_dict = {
    "title": "test_tool",
    "description": "test_tool description",
    "properties": {
        "arg_1": {"description": "Arg1 description", "type": "string"},
        "optional_arg_2": {
            "description": "Arg2 description",
            "type": "string",
            "default": "2",
        },
        "arg_3": {"description": "Arg3 description", "type": "integer"},
    },
}


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param(_TestTool(), id="tool from BaseTool"),
        pytest.param(test_tool, id="BaseModel"),
        pytest.param(test_tool_as_dict, id="JSON schema dict"),
    ],
)
def test_format_to_cohere_tools(
    tool: Union[Dict[str, Any], BaseTool, Type[BaseModel]],
) -> None:
    actual = _format_to_cohere_tools([tool])

    assert [expected_test_tool_definition] == actual


@pytest.mark.parametrize(
    "intermediate_step,expected",
    [
        pytest.param(
            (
                AgentAction(tool="tool_name", tool_input={"arg1": "value1"}, log=""),
                "result",
            ),
            {
                "call": {"name": "tool_name", "parameters": {"arg1": "value1"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as dict",
        ),
        pytest.param(
            (
                AgentAction(
                    tool="tool_name", tool_input=json.dumps({"arg1": "value1"}), log=""
                ),
                "result",
            ),
            {
                "call": {"name": "tool_name", "parameters": {"arg1": "value1"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as serialized dict",
        ),
        pytest.param(
            (AgentAction(tool="tool_name", tool_input="foo", log=""), "result"),
            {
                "call": {"name": "tool_name", "parameters": {"input": "foo"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as string",
        ),
        pytest.param(
            (AgentAction(tool="tool_name", tool_input="['foo']", log=""), "result"),
            {
                "call": {"name": "tool_name", "parameters": {"input": "['foo']"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input unrelated JSON",
        ),
    ],
)
def test_format_to_cohere_tools_messages(
    intermediate_step: Tuple[AgentAction, str], expected: List[Dict[str, Any]]
) -> None:
    actual = _format_to_cohere_tools_messages(intermediate_steps=[intermediate_step])

    assert [expected] == actual
