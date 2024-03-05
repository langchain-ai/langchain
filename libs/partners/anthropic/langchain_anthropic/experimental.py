import json
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
    cast,
)

from langchain_core._api.beta_decorator import beta
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
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


def _xml_to_dict(t: Any) -> Union[str, Dict[str, Any]]:
    # Base case: If the element has no children, return its text or an empty string.
    if len(t) == 0:
        return t.text or ""

    # Recursive case: The element has children. Convert them into a dictionary.
    d: Dict[str, Any] = {}
    for child in t:
        if child.tag not in d:
            d[child.tag] = _xml_to_dict(child)
        else:
            # Handle multiple children with the same tag
            if not isinstance(d[child.tag], list):
                d[child.tag] = [d[child.tag]]  # Convert existing entry into a list
            d[child.tag].append(_xml_to_dict(child))
    return d


def _xml_to_tool_calls(elem: Any) -> List[Dict[str, Any]]:
    """
    Convert an XML element and its children into a dictionary of dictionaries.
    """
    invokes = elem.findall("invoke")
    return [
        {
            "function": {
                "name": invoke.find("tool_name").text,
                "arguments": json.dumps(_xml_to_dict(invoke.find("parameters"))),
            },
            "type": "function",
        }
        for invoke in invokes
    ]


@beta()
class ChatAnthropicTools(ChatAnthropic):
    """Chat model for interacting with Anthropic functions."""

    _xmllib: Any = Field(default=None)

    @root_validator()
    def check_xml_lib(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # do this as an optional dep for temporary nature of this feature
            import defusedxml.ElementTree as DET  # type: ignore

            values["_xmllib"] = DET
        except ImportError:
            raise ImportError(
                "Could not import defusedxml python package. "
                "Please install it using `pip install defusedxml`"
            )
        return values

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
        **kwargs: Any,
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
            else:
                messages = [SystemMessage(content=tool_system), *messages]

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
        to_yield = result.generations[0]
        chunk = ChatGenerationChunk(
            message=cast(BaseMessageChunk, to_yield.message),
            generation_info=to_yield.generation_info,
        )
        if run_manager:
            run_manager.on_llm_new_token(
                cast(str, to_yield.message.content), chunk=chunk
            )
        yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # streaming not supported for functions
        result = await self._agenerate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        to_yield = result.generations[0]
        chunk = ChatGenerationChunk(
            message=cast(BaseMessageChunk, to_yield.message),
            generation_info=to_yield.generation_info,
        )
        if run_manager:
            await run_manager.on_llm_new_token(
                cast(str, to_yield.message.content), chunk=chunk
            )
        yield chunk

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        """Format the output of the model, parsing xml as a tool call."""
        text = data.content[0].text
        tools = kwargs.get("tools", None)

        additional_kwargs: Dict[str, Any] = {}

        if tools:
            # parse out the xml from the text
            try:
                # get everything between <function_calls> and </function_calls>
                start = text.find("<function_calls>")
                end = text.find("</function_calls>") + len("</function_calls>")
                xml_text = text[start:end]

                xml = self._xmllib.fromstring(xml_text)
                additional_kwargs["tool_calls"] = _xml_to_tool_calls(xml)
                text = ""
            except Exception:
                pass

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text, additional_kwargs=additional_kwargs)
                )
            ],
            llm_output=data,
        )
