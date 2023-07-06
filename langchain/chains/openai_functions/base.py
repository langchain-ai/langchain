""""""
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.input import get_colored_text
from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
}


def convert_python_function_to_openai_function(function: Callable) -> Dict[str, Any]:
    """Convert a simple Python function to a OpenAI function-calling API compatible dict."""
    source = inspect.getsource(function)
    name = re.search(r"^def (.*)\(", source).groups()[0]
    docstring_blocks = inspect.getdoc(function).split("\n\n")
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
    arg_descriptions = {}
    if args_block:
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":")
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + desc.strip()
    annotations = inspect.get_annotations(function)
    properties = {}
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
    spec = inspect.getfullargspec(function)

    required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def convert_to_openai_function(
    function: Union[Dict[str, Any], BaseModel, Callable]
) -> Dict[str, Any]:
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
            f"Unsupposerted function type {type(function)}. Functions must be passed in as Dict, pydantic.BaseModel, or Callable."
        )


class FunctionExecutorChain(Chain):
    functions: Dict[str, Callable]
    output_key: str = "output"
    input_key: str = "function"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the logic of this chain and return the output."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        name = inputs["function"].pop("name")
        args = inputs["function"].pop("arguments")
        _pretty_name = get_colored_text(name, "green")
        _pretty_args = get_colored_text(json.dumps(args, indent=2), "green")
        _text = f"Calling function {_pretty_name} with arguments:\n" + _pretty_args
        _run_manager.on_text(_text)
        output = self.functions[name](**args)
        return {self.output_key: output}


def create_openai_fn_chain(
    functions: List[Union[Dict[str, Any], BaseModel, Callable]],
    llm: Optional[BaseLanguageModel] = None,
    prompt: Optional[BasePromptTemplate] = None,
    output_parser: Optional[BaseOutputParser] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Chain:
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    openai_functions = [convert_to_openai_function(f) for f in functions]
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo-0613")
    prompt = prompt or ChatPromptTemplate.from_template("{input}")
    if output_parser:
        pass
    elif isinstance(functions[0], type) and issubclass(functions[0], BaseModel):
        if len(functions) > 1:
            pydantic_schema: Union[Dict, Type[BaseModel]] = {
                openai_fn["name"]: fn
                for openai_fn, fn in zip(openai_functions, functions)
            }
        else:
            pydantic_schema = functions[0]
        output_parser = PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    else:
        output_parser = JsonOutputFunctionsParser(args_only=False)

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
        verbose=verbose,
    )
    if not (
        isinstance(functions[0], type) and issubclass(functions[0], BaseModel)
    ) and callable(functions[0]):
        fn_map = {
            openai_fn["name"]: fn for openai_fn, fn in zip(openai_functions, functions)
        }
        fn_chain = FunctionExecutorChain(functions=fn_map, verbose=verbose)
        return SequentialChain(
            chains=[llm_chain, fn_chain],
            input_variables=llm_chain.input_keys,
            output_variables=["output"],
            verbose=verbose,
        )
    return llm_chain
