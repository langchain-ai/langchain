from typing import List, Type, Union

from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from vertexai.preview.generative_models import FunctionDeclaration  # type: ignore
from vertexai.preview.generative_models import Tool as VertexTool


def _format_tools_to_vertex_tool(
    tools: List[Union[BaseTool, Type[BaseModel], dict]],
) -> List[VertexTool]:
    "Format tool into the Vertex Tool instance."
    function_declarations = []
    for tool in tools:
        func = convert_to_openai_function(tool)
        function_declarations.append(FunctionDeclaration(**func))
    return [VertexTool(function_declarations=function_declarations)]


# For backwards compatibility
PydanticFunctionsOutputParser = PydanticOutputFunctionsParser
