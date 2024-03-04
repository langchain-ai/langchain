from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core._api.beta_decorator import beta
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_anthropic.chat_models import ChatAnthropic

SYSTEM_PROMPT_FORMAT = """In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{formatted_tools}
</tools>"""  # noqa: E501

TOOL_FORMAT = """<tool_description>
<tool_name>{tool_name}</tool_name>
<description>{tool_description}</description>
<parameters>
{formatted_parameters}
</parameters>
</tool_description>"""

TOOL_PARAMETER_FORMAT = """<parameter>
<name>{parameter_name}</name>
<type>{parameter_type}</type>
<description>{parameter_description}</description>
</parameter>"""


def get_system_message(tools: List[Dict]) -> str:
    tools_data: List[Dict] = [
        {
            "tool_name": tool["name"],
            "tool_description": tool["description"],
            "formatted_parameters": "\n".join(
                [
                    TOOL_PARAMETER_FORMAT.format(
                        parameter_name=name,
                        parameter_type=parameter["type"],
                        parameter_description=parameter.get("description"),
                    )
                    for name, parameter in tool["parameters"]["properties"].items()
                ]
            ),
        }
        for tool in tools
    ]
    tools_formatted = "\n".join(
        [
            TOOL_FORMAT.format(
                tool_name=tool["tool_name"],
                tool_description=tool["tool_description"],
                formatted_parameters=tool["formatted_parameters"],
            )
            for tool in tools_data
        ]
    )
    return SYSTEM_PROMPT_FORMAT.format(formatted_tools=tools_formatted)


@beta()
class ChatAnthropicFunctions(ChatAnthropic):
    """Chat model for interacting with Anthropic functions."""

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
        **kwargs,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model."""
        formatted_tools = [convert_to_openai_function(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError("kwargs are not supported for with_structured_output")
        llm = self.bind_tools([schema])
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # schema is pydantic
            return llm | PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            # schema is dict
            key_name = convert_to_openai_function(schema)["name"]
            return llm | JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict:
        tools: List[Dict] = kwargs.get("tools", None)
        # experimental tools are sent in as part of system prompt, so if
        # both are set, turn system prompt into tools + system prompt (tools first)
        if tools:
            tool_system = get_system_message(tools)

            if messages[0].type == "system":
                sys_content = messages[0].content
                new_sys_content = f"{tool_system}\n\n{sys_content}"
                messages = [SystemMessage(content=new_sys_content), *messages[1:]]
                return super()._format_params(messages=messages, stop=stop, **kwargs)

        return super()._format_params(messages=messages, stop=stop, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # streaming not supported for functions
        result = self._generate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        yield result.generations[0]
