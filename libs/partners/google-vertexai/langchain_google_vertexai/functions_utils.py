import asyncio
from typing import List, Union

from langchain.agents.agent import AgentOutputParser
from langchain_community.utils.openai_functions import (
    FunctionDescription,
)
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import Tool
from langchain_core.utils.json_schema import dereference_refs
from vertexai.preview.generative_models import (
    FunctionDeclaration,
)
from vertexai.preview.generative_models import (
    Tool as VertexTool,  # type: ignore
)


def format_tool_to_vertex_function(tool: Tool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)
        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": schema,
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


def format_tools_to_vertex_tool(tools: List[Tool]) -> List[VertexTool]:
    "Format tool into the Vertex Tool instance."
    function_declarations = []
    for tool in tools:
        function_declarations.append(
            FunctionDeclaration(**format_tool_to_vertex_function(tool))
        )

    return [VertexTool(function_declarations=function_declarations)]


class VertexAIFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    function_call parameter from OpenAI to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "vertexai-functions-agent"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        function_call = message.additional_kwargs.get("function_call", {})

        if function_call:
            function_name = function_call["name"]
            tool_input = function_call.get("arguments", {})

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )

        return AgentFinish(
            return_values={"output": message.content}, log=str(message.content)
        )

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.parse_result, result
        )
