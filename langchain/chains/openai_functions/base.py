""""""
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseLLMOutputParser

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
}


def _get_python_function_name(function: Callable) -> str:
    source = inspect.getsource(function)
    return re.search(r"^def (.*)\(", source).groups()[0]  # type: ignore


def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """"""
    docstring = inspect.getdoc(function)
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":")
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions


def _get_python_function_arguments(function: Callable, arg_descriptions: dict) -> dict:
    """"""
    properties = {}
    annotations = inspect.get_annotations(function)
    for arg, arg_type in annotations.items():
        if arg == "return":
            continue
        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            properties[arg] = arg_type.schema()
        elif arg_type.__name__ in PYTHON_TO_JSON_TYPES:
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_descriptions[arg]
    return properties


def _get_python_function_required_args(function: Callable) -> List[str]:
    """"""
    spec = inspect.getfullargspec(function)
    required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]
    return required


def convert_python_function_to_openai_function(function: Callable) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict."""
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {
        "name": _get_python_function_name(function),
        "description": description,
        "parameters": {
            "type": "object",
            "properties": _get_python_function_arguments(function, arg_descriptions),
            "required": _get_python_function_required_args(function),
        },
    }


def convert_to_openai_function(
    function: Union[Dict[str, Any], BaseModel, Callable]
) -> Dict[str, Any]:
    """"""
    if isinstance(function, dict):
        return function
    elif isinstance(function, type) and issubclass(function, BaseModel):
        schema = function.schema()
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": schema,
        }
    elif callable(function):
        return convert_python_function_to_openai_function(function)

    else:
        raise ValueError(
            f"Unsupported function type {type(function)}. Functions must be passed in"
            f" as Dict, pydantic.BaseModel, or Callable."
        )


def _get_openai_output_parser(
    functions: List[Union[Dict[str, Any], BaseModel, Callable]],
    function_names: List[str],
) -> BaseLLMOutputParser:
    if isinstance(functions[0], type) and issubclass(functions[0], BaseModel):
        if len(functions) > 1:
            pydantic_schema: Union[Dict, Type[BaseModel]] = {
                name: fn for name, fn in zip(function_names, functions)
            }
        else:
            pydantic_schema = functions[0]
        output_parser: BaseLLMOutputParser = PydanticOutputFunctionsParser(
            pydantic_schema=pydantic_schema
        )
    else:
        output_parser = JsonOutputFunctionsParser(args_only=False)
    return output_parser


def create_openai_fn_chain(
    functions: List[Union[Dict[str, Any], BaseModel, Callable]],
    llm: Optional[BaseLanguageModel] = None,
    prompt: Optional[BasePromptTemplate] = None,
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any,
) -> LLMChain:
    """"""
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    openai_functions = [convert_to_openai_function(f) for f in functions]
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo-0613")
    prompt = prompt or ChatPromptTemplate.from_template("{input}")
    fn_names = [oai_fn["name"] for oai_fn in openai_functions]
    output_parser = output_parser or _get_openai_output_parser(functions, fn_names)
    llm_kwargs: Dict[str, Any] = {
        "functions": openai_functions,
    }
    if len(openai_functions) == 1:
        llm_kwargs["function_call"] = {"name": openai_functions[0]["name"]}
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        llm_kwargs=llm_kwargs,
        output_key="function",
        **kwargs,
    )
    return llm_chain
