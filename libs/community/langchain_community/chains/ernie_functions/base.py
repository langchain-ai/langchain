"""Methods for creating chains that use Ernie function-calling APIs."""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseLLMOutputParser,
    BaseOutputParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel

from langchain_community.output_parsers.ernie_functions import (
    JsonOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_community.utils.ernie_functions import convert_pydantic_to_ernie_function

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
}


def _get_python_function_name(function: Callable) -> str:
    """Get the name of a Python function."""
    return function.__name__


def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
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
    """Get JsonSchema describing a Python functions arguments.

    Assumes all function arguments are of primitive types (int, float, str, bool) or
    are subclasses of pydantic.BaseModel.
    """
    properties = {}
    annotations = inspect.getfullargspec(function).annotations
    for arg, arg_type in annotations.items():
        if arg == "return":
            continue
        if isinstance(arg_type, type) and is_basemodel_subclass(arg_type):
            # Mypy error:
            # "type" has no attribute "schema"
            properties[arg] = arg_type.schema()  # type: ignore[attr-defined]
        elif arg_type.__name__ in PYTHON_TO_JSON_TYPES:
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_descriptions[arg]
    return properties


def _get_python_function_required_args(function: Callable) -> List[str]:
    """Get the required arguments for a Python function."""
    spec = inspect.getfullargspec(function)
    required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]

    is_class = type(function) is type
    if is_class and required[0] == "self":
        required = required[1:]
    return required


def convert_python_function_to_ernie_function(
    function: Callable,
) -> Dict[str, Any]:
    """Convert a Python function to an Ernie function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
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


def convert_to_ernie_function(
    function: Union[Dict[str, Any], Type[BaseModel], Callable],
) -> Dict[str, Any]:
    """Convert a raw function/class to an Ernie function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid Ernie
            function.

    Returns:
        A dict version of the passed in function which is compatible with the
            Ernie function-calling API.
    """
    if isinstance(function, dict):
        return function
    elif isinstance(function, type) and is_basemodel_subclass(function):
        return cast(Dict, convert_pydantic_to_ernie_function(function))
    elif callable(function):
        return convert_python_function_to_ernie_function(function)

    else:
        raise ValueError(
            f"Unsupported function type {type(function)}. Functions must be passed in"
            f" as Dict, pydantic.BaseModel, or Callable."
        )


def get_ernie_output_parser(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
    """Get the appropriate function output parser given the user functions.

    Args:
        functions: Sequence where element is a dictionary, a pydantic.BaseModel class,
            or a Python function. If a dictionary is passed in, it is assumed to
            already be a valid Ernie function.

    Returns:
        A PydanticOutputFunctionsParser if functions are Pydantic classes, otherwise
            a JsonOutputFunctionsParser. If there's only one function and it is
            not a Pydantic class, then the output parser will automatically extract
            only the function arguments and not the function name.
    """
    function_names = [convert_to_ernie_function(f)["name"] for f in functions]
    if isinstance(functions[0], type) and is_basemodel_subclass(functions[0]):
        if len(functions) > 1:
            pydantic_schema: Union[Dict, Type[BaseModel]] = {
                name: fn for name, fn in zip(function_names, functions)
            }
        else:
            pydantic_schema = functions[0]
        output_parser: Union[BaseOutputParser, BaseGenerationOutputParser] = (
            PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
        )
    else:
        output_parser = JsonOutputFunctionsParser(args_only=len(functions) <= 1)
    return output_parser


def create_ernie_fn_runnable(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
    llm: Runnable,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **kwargs: Any,
) -> Runnable:
    """Create a runnable sequence that uses Ernie functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid Ernie functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.ernie_functions import create_ernie_fn_chain
                from langchain_community.chat_models import ErnieBotChat
                from langchain_core.prompts import ChatPromptTemplate
                from pydantic import BaseModel, Field


                class RecordPerson(BaseModel):
                    \"\"\"Record some identifying information about a person.\"\"\"

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    \"\"\"Record some identifying information about a dog.\"\"\"

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = ErnieBotChat(model_name="ERNIE-Bot-4")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "Make calls to the relevant function to record the entities in the following input: {input}"),
                        ("assistant", "OK!"),
                        ("user", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_ernie_fn_runnable([RecordPerson, RecordDog], llm, prompt)
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    ernie_functions = [convert_to_ernie_function(f) for f in functions]
    llm_kwargs: Dict[str, Any] = {"functions": ernie_functions, **kwargs}
    if len(ernie_functions) == 1:
        llm_kwargs["function_call"] = {"name": ernie_functions[0]["name"]}
    output_parser = output_parser or get_ernie_output_parser(functions)
    return prompt | llm.bind(**llm_kwargs) | output_parser


def create_structured_output_runnable(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: Runnable,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **kwargs: Any,
) -> Runnable:
    """Create a runnable that uses an Ernie function to get a structured output.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON.

    Returns:
        A runnable sequence that will pass the given function to the model when run.

    Example:
        .. code-block:: python

            from typing import Optional

            from langchain.chains.ernie_functions import create_structured_output_chain
            from langchain_community.chat_models import ErnieBotChat
            from langchain_core.prompts import ChatPromptTemplate
            from pydantic import BaseModel, Field

            class Dog(BaseModel):
                \"\"\"Identifying information about a dog.\"\"\"

                name: str = Field(..., description="The dog's name")
                color: str = Field(..., description="The dog's color")
                fav_food: Optional[str] = Field(None, description="The dog's favorite food")

            llm = ErnieBotChat(model_name="ERNIE-Bot-4")
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("user", "Use the given format to extract information from the following input: {input}"),
                    ("assistant", "OK!"),
                    ("user", "Tip: Make sure to answer in the correct format"),
                ]
            )
            chain = create_structured_output_chain(Dog, llm, prompt)
            chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
            # -> Dog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
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
            """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

            output: output_schema  # type: ignore[valid-type]

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter, attr_name="output"
        )
    return create_ernie_fn_runnable(
        [function],
        llm,
        prompt,
        output_parser=output_parser,
        **kwargs,
    )


""" --- Legacy --- """


def create_ernie_fn_chain(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    *,
    output_key: str = "function",
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any,
) -> LLMChain:  # type: ignore[valid-type]
    """[Legacy] Create an LLM chain that uses Ernie functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid Ernie functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_key: The key to use when returning the output in LLMChain.__call__.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.

    Returns:
        An LLMChain that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.ernie_functions import create_ernie_fn_chain
                from langchain_community.chat_models import ErnieBotChat
                from langchain_core.prompts import ChatPromptTemplate

                from pydantic import BaseModel, Field


                class RecordPerson(BaseModel):
                    \"\"\"Record some identifying information about a person.\"\"\"

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    \"\"\"Record some identifying information about a dog.\"\"\"

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = ErnieBotChat(model_name="ERNIE-Bot-4")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "Make calls to the relevant function to record the entities in the following input: {input}"),
                        ("assistant", "OK!"),
                        ("user", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_ernie_fn_chain([RecordPerson, RecordDog], llm, prompt)
                chain.run("Harry was a chubby brown beagle who loved chicken")
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    ernie_functions = [convert_to_ernie_function(f) for f in functions]
    output_parser = output_parser or get_ernie_output_parser(functions)
    llm_kwargs: Dict[str, Any] = {
        "functions": ernie_functions,
    }
    if len(ernie_functions) == 1:
        llm_kwargs["function_call"] = {"name": ernie_functions[0]["name"]}
    llm_chain = LLMChain(  # type: ignore[misc]
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        llm_kwargs=llm_kwargs,
        output_key=output_key,
        **kwargs,
    )
    return llm_chain


def create_structured_output_chain(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    *,
    output_key: str = "function",
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any,
) -> LLMChain:  # type: ignore[valid-type]
    """[Legacy] Create an LLMChain that uses an Ernie function to get a structured output.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_key: The key to use when returning the output in LLMChain.__call__.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON.

    Returns:
        An LLMChain that will pass the given function to the model.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.ernie_functions import create_structured_output_chain
                from langchain_community.chat_models import ErnieBotChat
                from langchain_core.prompts import ChatPromptTemplate

                from pydantic import BaseModel, Field

                class Dog(BaseModel):
                    \"\"\"Identifying information about a dog.\"\"\"

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ErnieBotChat(model_name="ERNIE-Bot-4")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "Use the given format to extract information from the following input: {input}"),
                        ("assistant", "OK!"),
                        ("user", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_structured_output_chain(Dog, llm, prompt)
                chain.run("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
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
            """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

            output: output_schema  # type: ignore[valid-type]

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter, attr_name="output"
        )
    return create_ernie_fn_chain(
        [function],
        llm,
        prompt,
        output_key=output_key,
        output_parser=output_parser,
        **kwargs,
    )
