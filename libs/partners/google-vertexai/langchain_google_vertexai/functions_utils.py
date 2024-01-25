import json
from typing import Dict, List, Type, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import FunctionDescription
from langchain_core.utils.json_schema import dereference_refs
from vertexai.preview.generative_models import (  # type: ignore
    FunctionDeclaration,
)
from vertexai.preview.generative_models import Tool as VertexTool


def _format_pydantic_to_vertex_function(
    pydantic_model: Type[BaseModel],
) -> FunctionDescription:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)

    return {
        "name": schema["title"],
        "description": schema.get("description", ""),
        "parameters": {
            "properties": {
                k: {
                    "type": v["type"],
                    "description": v.get("description"),
                }
                for k, v in schema["properties"].items()
            },
            "required": schema["required"],
            "type": schema["type"],
        },
    }


def _format_tool_to_vertex_function(tool: BaseTool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)

        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": {
                "properties": {
                    k: {
                        "type": v["type"],
                        "description": v.get("description"),
                    }
                    for k, v in schema["properties"].items()
                },
                "required": schema["required"],
                "type": schema["type"],
            },
        }
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "properties": {
                    "__arg1": {"type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            },
        }


def _format_tools_to_vertex_tool(
    tools: List[Union[BaseTool, Type[BaseModel]]],
) -> List[VertexTool]:
    "Format tool into the Vertex Tool instance."
    function_declarations = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            func = _format_tool_to_vertex_function(tool)
        else:
            func = _format_pydantic_to_vertex_function(tool)
        function_declarations.append(FunctionDeclaration(**func))

    return [VertexTool(function_declarations=function_declarations)]


class PydanticFunctionsOutputParser(BaseOutputParser):
    """Parse an output as a pydantic object.

    This parser is used to parse the output of a ChatModel that uses
    Google Vertex function format to invoke functions.

    The parser extracts the function call invocation and matches
    them to the pydantic schema provided.

    An exception will be raised if the function call does not match
    the provided schema.

    Example:

        ... code-block:: python

            message = AIMessage(
                content="This is a test message",
                additional_kwargs={
                    "function_call": {
                        "name": "cookie",
                        "arguments": json.dumps({"name": "value", "age": 10}),
                    }
                },
            )
            chat_generation = ChatGeneration(message=message)

            class Cookie(BaseModel):
                name: str
                age: int

            class Dog(BaseModel):
                species: str

            # Full output
            parser = PydanticOutputFunctionsParser(
                pydantic_schema={"cookie": Cookie, "dog": Dog}
            )
            result = parser.parse_result([chat_generation])
    """

    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> BaseModel:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        function_call = message.additional_kwargs.get("function_call", {})
        if function_call:
            function_name = function_call["name"]
            tool_input = function_call.get("arguments", {})
            if isinstance(self.pydantic_schema, dict):
                schema = self.pydantic_schema[function_name]
            else:
                schema = self.pydantic_schema
            return schema(**json.loads(tool_input))
        else:
            raise OutputParserException(f"Could not parse function call: {message}")

    def parse(self, text: str) -> BaseModel:
        raise ValueError("Can only parse messages")
