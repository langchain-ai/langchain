from typing import Any, Dict, List, Sequence, Tuple, Type, Union

from cohere.types import Tool, ToolParameterDefinitionsValue
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function


def create_cohere_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    def llm_with_tools(input_: Dict) -> Runnable:
        tool_results = (
            input_["tool_results"] if len(input_["tool_results"]) > 0 else None
        )
        tools_ = input_["tools"] if len(input_["tools"]) > 0 else None
        return RunnableLambda(lambda x: x["input"]) | llm.bind(
            tools=tools_, tool_results=tool_results
        )

    agent = (
        RunnablePassthrough.assign(
            # Intermediate steps are in tool results.
            # Edit below to change the prompt parameters.
            input=lambda x: prompt.format_messages(
                input=x["input"], agent_scratchpad=[]
            ),
            tools=lambda x: _format_to_cohere_tools(tools),
            tool_results=lambda x: _format_to_cohere_tools_messages(
                x["intermediate_steps"]
            ),
        )
        | llm_with_tools
        | _CohereToolsAgentOutputParser()
    )
    return agent


def _format_to_cohere_tools(
    tools: Sequence[Union[Dict[str, Any], BaseTool, Type[BaseModel]]],
) -> List[Dict[str, Any]]:
    return [_convert_to_cohere_tool(tool) for tool in tools]


def _format_to_cohere_tools_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> list:
    """Convert (AgentAction, tool output) tuples into tool messages."""
    if len(intermediate_steps) == 0:
        return []
    tool_results = []
    for agent_action, observation in intermediate_steps:
        tool_results.append(
            {
                "call": {
                    "name": agent_action.tool,
                    "parameters": agent_action.tool_input,
                },
                "outputs": [{"answer": observation}],
            }
        )

    return tool_results


def _convert_to_cohere_tool(
    tool: Union[Dict[str, Any], BaseTool, Type[BaseModel]],
) -> Dict[str, Any]:
    """
    Convert a BaseTool instance, JSON schema dict, or BaseModel type to a Cohere tool.
    """
    if isinstance(tool, BaseTool):
        return Tool(
            name=tool.name,
            description=tool.description,
            parameter_definitions={
                param_name: ToolParameterDefinitionsValue(
                    description=param_definition.get("description"),
                    type=param_definition.get("type"),
                    required="default" not in param_definition,
                )
                for param_name, param_definition in tool.args.items()
            },
        ).dict()
    elif isinstance(tool, dict):
        if not all(k in tool for k in ("title", "description", "properties")):
            raise ValueError(
                "Unsupported dict type. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
            )
        return Tool(
            name=tool.get("title"),
            description=tool.get("description"),
            parameter_definitions={
                param_name: ToolParameterDefinitionsValue(
                    description=param_definition.get("description"),
                    type=param_definition.get("type"),
                    required="default" not in param_definition,
                )
                for param_name, param_definition in tool.get("properties", {}).items()
            },
        ).dict()
    elif issubclass(tool, BaseModel):
        as_json_schema_function = convert_to_openai_function(tool)
        parameters = as_json_schema_function.get("parameters", {})
        properties = parameters.get("properties", {})
        return Tool(
            name=as_json_schema_function.get("name"),
            description=as_json_schema_function.get(
                # The Cohere API requires the description field.
                "description",
                as_json_schema_function.get("name"),
            ),
            parameter_definitions={
                param_name: ToolParameterDefinitionsValue(
                    description=param_definition.get("description"),
                    type=param_definition.get("type"),
                    required=param_name in parameters.get("required", []),
                )
                for param_name, param_definition in properties.items()
            },
        ).dict()
    else:
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
        )


class _CohereToolsAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError(f"Expected ChatGeneration, got {type(result)}")
        if result[0].message.additional_kwargs["tool_calls"]:
            actions = []
            for tool in result[0].message.additional_kwargs["tool_calls"]:
                function = tool.get("function", {})
                actions.append(
                    AgentAction(
                        tool=function.get("name"),
                        tool_input=function.get("arguments"),
                        log=function.get("name"),
                    )
                )
            return actions
        else:
            return AgentFinish(
                return_values={
                    "text": result[0].message.content,
                    "additional_info": result[0].message.additional_kwargs,
                },
                log="",
            )

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
