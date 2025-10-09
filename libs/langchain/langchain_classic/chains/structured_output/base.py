import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel


def _create_openai_tools_runnable(
    tool: dict[str, Any] | type[BaseModel] | Callable,
    llm: Runnable,
    *,
    prompt: BasePromptTemplate | None,
    output_parser: BaseOutputParser | BaseGenerationOutputParser | None,
    enforce_tool_usage: bool,
    first_tool_only: bool,
) -> Runnable:
    oai_tool = convert_to_openai_tool(tool)
    llm_kwargs: dict[str, Any] = {"tools": [oai_tool]}
    if enforce_tool_usage:
        llm_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": oai_tool["function"]["name"]},
        }
    output_parser = output_parser or _get_openai_tool_output_parser(
        tool,
        first_tool_only=first_tool_only,
    )
    if prompt:
        return prompt | llm.bind(**llm_kwargs) | output_parser
    return llm.bind(**llm_kwargs) | output_parser


def _get_openai_tool_output_parser(
    tool: dict[str, Any] | type[BaseModel] | Callable,
    *,
    first_tool_only: bool = False,
) -> BaseOutputParser | BaseGenerationOutputParser:
    if isinstance(tool, type) and is_basemodel_subclass(tool):
        output_parser: BaseOutputParser | BaseGenerationOutputParser = (
            PydanticToolsParser(tools=[tool], first_tool_only=first_tool_only)
        )
    else:
        key_name = convert_to_openai_tool(tool)["function"]["name"]
        output_parser = JsonOutputKeyToolsParser(
            first_tool_only=first_tool_only,
            key_name=key_name,
        )
    return output_parser


def get_openai_output_parser(
    functions: Sequence[dict[str, Any] | type[BaseModel] | Callable],
) -> BaseOutputParser | BaseGenerationOutputParser:
    """Get the appropriate function output parser given the user functions.

    Args:
        functions: Sequence where element is a dictionary, a pydantic.BaseModel class,
            or a Python function. If a dictionary is passed in, it is assumed to
            already be a valid OpenAI function.

    Returns:
        A PydanticOutputFunctionsParser if functions are Pydantic classes, otherwise
            a JsonOutputFunctionsParser. If there's only one function and it is
            not a Pydantic class, then the output parser will automatically extract
            only the function arguments and not the function name.
    """
    if isinstance(functions[0], type) and is_basemodel_subclass(functions[0]):
        if len(functions) > 1:
            pydantic_schema: dict | type[BaseModel] = {
                convert_to_openai_function(fn)["name"]: fn for fn in functions
            }
        else:
            pydantic_schema = functions[0]
        output_parser: BaseOutputParser | BaseGenerationOutputParser = (
            PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
        )
    else:
        output_parser = JsonOutputFunctionsParser(args_only=len(functions) <= 1)
    return output_parser


def _create_openai_json_runnable(
    output_schema: dict[str, Any] | type[BaseModel],
    llm: Runnable,
    prompt: BasePromptTemplate | None = None,
    *,
    output_parser: BaseOutputParser | BaseGenerationOutputParser | None = None,
) -> Runnable:
    if isinstance(output_schema, type) and is_basemodel_subclass(output_schema):
        output_parser = output_parser or PydanticOutputParser(
            pydantic_object=output_schema,
        )
        schema_as_dict = convert_to_openai_function(output_schema)["parameters"]
    else:
        output_parser = output_parser or JsonOutputParser()
        schema_as_dict = output_schema

    llm = llm.bind(response_format={"type": "json_object"})
    if prompt:
        if "output_schema" in prompt.input_variables:
            prompt = prompt.partial(output_schema=json.dumps(schema_as_dict, indent=2))

        return prompt | llm | output_parser
    return llm | output_parser


def _create_openai_functions_structured_output_runnable(
    output_schema: dict[str, Any] | type[BaseModel],
    llm: Runnable,
    prompt: BasePromptTemplate | None = None,
    *,
    output_parser: BaseOutputParser | BaseGenerationOutputParser | None = None,
    **llm_kwargs: Any,
) -> Runnable:
    if isinstance(output_schema, dict):
        function: Any = {
            "name": "output_formatter",
            "description": (
                "Output formatter. Should always be used to format your response to the"
                " user."
            ),
            "parameters": output_schema,
        }
    else:

        class _OutputFormatter(BaseModel):
            """Output formatter.

            Should always be used to format your response to the user.
            """

            output: output_schema  # type: ignore[valid-type]

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter,
            attr_name="output",
        )
    return create_openai_fn_runnable(
        [function],
        llm,
        prompt=prompt,
        output_parser=output_parser,
        **llm_kwargs,
    )
