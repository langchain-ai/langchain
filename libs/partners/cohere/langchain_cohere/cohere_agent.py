import json
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

from cohere.types import (
    ChatRequestToolResultsItem,
    Tool,
    ToolCall,
    ToolParameterDefinitionsValue,
)
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
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)

from langchain_cohere.utils import (
    JSON_TO_PYTHON_TYPES,
    _remove_signature_from_tool_description,
)


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
            input=lambda x: prompt.format_messages(**x, agent_scratchpad=[]),
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
) -> List[Dict[str, Any]]:
    """Convert (AgentAction, tool output) tuples into tool messages."""
    if len(intermediate_steps) == 0:
        return []
    tool_results = []
    for agent_action, observation in intermediate_steps:
        # agent_action.tool_input can be a dict, serialised dict, or string.
        # Cohere API only accepts a dict.
        tool_call_parameters: Dict[str, Any]
        if isinstance(agent_action.tool_input, dict):
            # tool_input is a dict, use as-is.
            tool_call_parameters = agent_action.tool_input
        else:
            try:
                # tool_input is serialised dict.
                tool_call_parameters = json.loads(agent_action.tool_input)
                if not isinstance(tool_call_parameters, dict):
                    raise ValueError()
            except ValueError:
                # tool_input is a string, last ditch attempt at having something useful.
                tool_call_parameters = {"input": agent_action.tool_input}
        tool_results.append(
            ChatRequestToolResultsItem(
                call=ToolCall(
                    name=agent_action.tool,
                    parameters=tool_call_parameters,
                ),
                outputs=[{"answer": observation}],
            ).dict()
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
            description=_remove_signature_from_tool_description(
                tool.name, tool.description
            ),
            parameter_definitions={
                param_name: ToolParameterDefinitionsValue(
                    description=param_definition.get("description")
                    if "description" in param_definition
                    else "",
                    type=JSON_TO_PYTHON_TYPES.get(
                        param_definition.get("type"), param_definition.get("type")
                    ),
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
                    type=JSON_TO_PYTHON_TYPES.get(
                        param_definition.get("type"), param_definition.get("type")
                    ),
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
                    type=JSON_TO_PYTHON_TYPES.get(
                        param_definition.get("type"), param_definition.get("type")
                    ),
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
        if "tool_calls" in result[0].message.additional_kwargs:
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
