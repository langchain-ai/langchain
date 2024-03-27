from typing import Any, Dict, List, Optional, Sequence, Type, Union

import pytest
from langchain_core.tools import BaseModel, BaseTool, Field

from langchain_cohere.cohere_agent import format_to_cohere_tools


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


@pytest.mark.parametrize(
    "tools,expected",
    [
        pytest.param(
            [_TestTool()],
            [
                {
                    "description": "test_tool description",
                    "name": "test_tool",
                    "parameter_definitions": {
                        "arg_1": {
                            "description": "Arg1 description",
                            "required": False,
                            "type": "string",
                        },
                        "optional_arg_2": {
                            "description": "Arg2 description",
                            "required": True,
                            "type": "string",
                        },
                        "arg_3": {
                            "description": "Arg3 description",
                            "required": False,
                            "type": "integer",
                        },
                    },
                }
            ],
            id="tool from BaseTool",
        )
    ],
)
def test_format_to_cohere_tools(
    tools: Sequence[Union[Dict[str, Any], BaseTool]], expected: List[Dict[str, Any]]
) -> None:
    actual = format_to_cohere_tools(tools)

    assert expected == actual
