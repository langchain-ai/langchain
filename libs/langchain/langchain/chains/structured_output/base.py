import json
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, Union

from langchain_core._api import deprecated
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


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that "
        "is available on ChatModels capable of tool calling. "
        "You can read more about the method here: "
        "<https://python.langchain.com/docs/modules/model_io/chat/structured_output/>. "
        "Please follow our extraction use case documentation for more guidelines "
        "on how to do information extraction with LLMs. "
        "<https://python.langchain.com/docs/use_cases/extraction/>. "
        "If you notice other issues, please provide "
        "feedback here: "
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    removal="1.0",
    alternative=(
        """
            from pydantic import BaseModel, Field
            from langchain_anthropic import ChatAnthropic

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            # Or any other chat model that supports tools.
            # Please reference to to the documentation of structured_output
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
            structured_llm = model.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats.
                Make sure to call the Joke function.")
            """
    ),
)
def create_openai_fn_runnable(
    functions: Sequence[Union[dict[str, Any], type[BaseModel], Callable]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    enforce_single_function_usage: bool = True,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **llm_kwargs: Any,
) -> Runnable:
    """Create a runnable sequence that uses OpenAI functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid OpenAI functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        enforce_single_function_usage: only used if a single function is passed in. If
            True, then the model will be forced to use the given function. If False,
            then the model will be given the option to use the given function or not.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.
        **llm_kwargs: Additional named arguments to pass to the language model.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.structured_output import create_openai_fn_runnable
                from langchain_openai import ChatOpenAI
                from pydantic import BaseModel, Field


                class RecordPerson(BaseModel):
                    '''Record some identifying information about a person.'''

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    '''Record some identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = ChatOpenAI(model="gpt-4", temperature=0)
                structured_llm = create_openai_fn_runnable([RecordPerson, RecordDog], llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken)
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        msg = "Need to pass in at least one function. Received zero."
        raise ValueError(msg)
    openai_functions = [convert_to_openai_function(f) for f in functions]
    llm_kwargs_: dict[str, Any] = {"functions": openai_functions, **llm_kwargs}
    if len(openai_functions) == 1 and enforce_single_function_usage:
        llm_kwargs_["function_call"] = {"name": openai_functions[0]["name"]}
    output_parser = output_parser or get_openai_output_parser(functions)
    if prompt:
        return prompt | llm.bind(**llm_kwargs_) | output_parser
    return llm.bind(**llm_kwargs_) | output_parser


@deprecated(
    since="0.1.17",
    message=(
        "LangChain has introduced a method called `with_structured_output` that "
        "is available on ChatModels capable of tool calling. "
        "You can read more about the method here: "
        "<https://python.langchain.com/docs/modules/model_io/chat/structured_output/>."
        "Please follow our extraction use case documentation for more guidelines "
        "on how to do information extraction with LLMs. "
        "<https://python.langchain.com/docs/use_cases/extraction/>. "
        "If you notice other issues, please provide "
        "feedback here: "
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    removal="1.0",
    alternative=(
        """
            from pydantic import BaseModel, Field
            from langchain_anthropic import ChatAnthropic

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            # Or any other chat model that supports tools.
            # Please reference to to the documentation of structured_output
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
            structured_llm = model.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats.
                Make sure to call the Joke function.")
            """
    ),
)
def create_structured_output_runnable(
    output_schema: Union[dict[str, Any], type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    enforce_function_usage: bool = True,
    return_single: bool = True,
    mode: Literal[
        "openai-functions",
        "openai-tools",
        "openai-json",
    ] = "openai-functions",
    **kwargs: Any,
) -> Runnable:
    """Create a runnable for extracting structured outputs.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use. Assumed to support the OpenAI function-calling API
            if mode is 'openai-function'. Assumed to support OpenAI response_format
            parameter if mode is 'openai-json'.
        prompt: BasePromptTemplate to pass to the model. If mode is 'openai-json' and
            prompt has input variable 'output_schema' then the given output_schema
            will be converted to a JsonSchema and inserted in the prompt.
        output_parser: Output parser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModel is passed
            in, then the OutputParser will try to parse outputs using the pydantic
            class. Otherwise model outputs will be parsed as JSON.
        mode: How structured outputs are extracted from the model. If 'openai-functions'
            then OpenAI function calling is used with the deprecated 'functions',
            'function_call' schema. If 'openai-tools' then OpenAI function
            calling with the latest 'tools', 'tool_choice' schema is used. This is
            recommended over 'openai-functions'. If 'openai-json' then OpenAI model
            with response_format set to JSON is used.
        enforce_function_usage: Only applies when mode is 'openai-tools' or
            'openai-functions'. If True, then the model will be forced to use the given
            output schema. If False, then the model can elect whether to use the output
            schema.
        return_single: Only applies when mode is 'openai-tools'. Whether to a list of
            structured outputs or a single one. If True and model does not return any
            structured outputs then chain output is None. If False and model does not
            return any structured outputs then chain output is an empty list.
        kwargs: Additional named arguments.

    Returns:
        A runnable sequence that will return a structured output(s) matching the given
            output_schema.

    OpenAI tools example with Pydantic schema (mode='openai-tools'):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from pydantic import BaseModel, Field


                class RecordDog(BaseModel):
                    '''Record some identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are an extraction algorithm. Please extract every possible instance"),
                        ('human', '{input}')
                    ]
                )
                structured_llm = create_structured_output_runnable(
                    RecordDog,
                    llm,
                    mode="openai-tools",
                    enforce_function_usage=True,
                    return_single=True
                )
                structured_llm.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")

    OpenAI tools example with dict schema (mode="openai-tools"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI


                dog_schema = {
                    "type": "function",
                    "function": {
                        "name": "record_dog",
                        "description": "Record some identifying information about a dog.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "description": "The dog's name",
                                    "type": "string"
                                },
                                "color": {
                                    "description": "The dog's color",
                                    "type": "string"
                                },
                                "fav_food": {
                                    "description": "The dog's favorite food",
                                    "type": "string"
                                }
                            },
                            "required": ["name", "color"]
                        }
                    }
                }


                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(
                    dog_schema,
                    llm,
                    mode="openai-tools",
                    enforce_function_usage=True,
                    return_single=True
                )
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> {'name': 'Harry', 'color': 'brown', 'fav_food': 'chicken'}

    OpenAI functions example (mode="openai-functions"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from pydantic import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-functions")
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")

    OpenAI functions with prompt example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from pydantic import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-functions")
                system = '''Extract information about any dogs mentioned in the user input.'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}"),]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    OpenAI json response format example (mode="openai-json"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from pydantic import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-json")
                system = '''You are a world class assistant for extracting information in structured JSON formats. \

                Extract a valid JSON blob from the user input that matches the following JSON Schema:

                {output_schema}'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}"),]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
    """  # noqa: E501
    # for backwards compatibility
    force_function_usage = kwargs.get(
        "enforce_single_function_usage",
        enforce_function_usage,
    )

    if mode == "openai-tools":
        # Protect against typos in kwargs
        keys_in_kwargs = set(kwargs.keys())
        # Backwards compatibility keys
        unrecognized_keys = keys_in_kwargs - {"enforce_single_function_usage"}
        if unrecognized_keys:
            msg = f"Got an unexpected keyword argument(s): {unrecognized_keys}."
            raise TypeError(msg)

        return _create_openai_tools_runnable(
            output_schema,
            llm,
            prompt=prompt,
            output_parser=output_parser,
            enforce_tool_usage=force_function_usage,
            first_tool_only=return_single,
        )

    if mode == "openai-functions":
        return _create_openai_functions_structured_output_runnable(
            output_schema,
            llm,
            prompt=prompt,
            output_parser=output_parser,
            enforce_single_function_usage=force_function_usage,
            **kwargs,  # llm-specific kwargs
        )
    if mode == "openai-json":
        if force_function_usage:
            msg = (
                "enforce_single_function_usage is not supported for mode='openai-json'."
            )
            raise ValueError(msg)
        return _create_openai_json_runnable(
            output_schema,
            llm,
            prompt=prompt,
            output_parser=output_parser,
            **kwargs,
        )
    msg = (
        f"Invalid mode {mode}. Expected one of 'openai-tools', 'openai-functions', "
        f"'openai-json'."
    )
    raise ValueError(msg)


def _create_openai_tools_runnable(
    tool: Union[dict[str, Any], type[BaseModel], Callable],
    llm: Runnable,
    *,
    prompt: Optional[BasePromptTemplate],
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]],
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
    tool: Union[dict[str, Any], type[BaseModel], Callable],
    *,
    first_tool_only: bool = False,
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
    if isinstance(tool, type) and is_basemodel_subclass(tool):
        output_parser: Union[BaseOutputParser, BaseGenerationOutputParser] = (
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
    functions: Sequence[Union[dict[str, Any], type[BaseModel], Callable]],
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
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
            pydantic_schema: Union[dict, type[BaseModel]] = {
                convert_to_openai_function(fn)["name"]: fn for fn in functions
            }
        else:
            pydantic_schema = functions[0]
        output_parser: Union[BaseOutputParser, BaseGenerationOutputParser] = (
            PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
        )
    else:
        output_parser = JsonOutputFunctionsParser(args_only=len(functions) <= 1)
    return output_parser


def _create_openai_json_runnable(
    output_schema: Union[dict[str, Any], type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
) -> Runnable:
    """"""
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
    output_schema: Union[dict[str, Any], type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
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
            """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

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
