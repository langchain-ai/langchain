"""Methods for creating chains that use OpenAI function-calling APIs."""
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseLLMOutputParser,
    BaseOutputParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
    PYTHON_TO_JSON_TYPES,
    convert_to_openai_function,
)

from langchain.chains import LLMChain
from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)


def get_openai_output_parser(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
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
    function_names = [convert_to_openai_function(f)["name"] for f in functions]
    if isinstance(functions[0], type) and issubclass(functions[0], BaseModel):
        if len(functions) > 1:
            pydantic_schema: Union[Dict, Type[BaseModel]] = {
                name: fn for name, fn in zip(function_names, functions)
            }
        else:
            pydantic_schema = functions[0]
        output_parser: Union[
            BaseOutputParser, BaseGenerationOutputParser
        ] = PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    else:
        output_parser = JsonOutputFunctionsParser(args_only=len(functions) <= 1)
    return output_parser


def create_openai_fn_runnable(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
    llm: Runnable,
    prompt: BasePromptTemplate,
    *,
    enforce_single_function_usage: bool = True,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **kwargs: Any,
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

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.openai_functions import create_openai_fn_runnable
                from langchain_community.chat_models import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field


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


                llm = ChatOpenAI(model="gpt-4", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a world class algorithm for recording entities."),
                        ("human", "Make calls to the relevant function to record the entities in the following input: {input}"),
                        ("human", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_openai_fn_runnable([RecordPerson, RecordDog], llm, prompt)
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    openai_functions = [convert_to_openai_function(f) for f in functions]
    llm_kwargs: Dict[str, Any] = {"functions": openai_functions, **kwargs}
    if len(openai_functions) == 1 and enforce_single_function_usage:
        llm_kwargs["function_call"] = {"name": openai_functions[0]["name"]}
    output_parser = output_parser or get_openai_output_parser(functions)
    return prompt | llm.bind(**llm_kwargs) | output_parser


def create_structured_output_runnable(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: Runnable,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **kwargs: Any,
) -> Runnable:
    """Create a runnable that uses an OpenAI function to get a structured output.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
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

                from langchain.chains.openai_functions import create_structured_output_runnable
                from langchain_community.chat_models import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    \"\"\"Identifying information about a dog.\"\"\"

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a world class algorithm for extracting information in structured formats."),
                        ("human", "Use the given format to extract information from the following input: {input}"),
                        ("human", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_structured_output_runnable(Dog, llm, prompt)
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

            output: output_schema  # type: ignore

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter, attr_name="output"
        )
    return create_openai_fn_runnable(
        [function],
        llm,
        prompt,
        output_parser=output_parser,
        **kwargs,
    )


""" --- Legacy --- """


@deprecated(since="0.1.1", removal="0.2.0", alternative="create_openai_fn_runnable")
def create_openai_fn_chain(
    functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    *,
    enforce_single_function_usage: bool = True,
    output_key: str = "function",
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any,
) -> LLMChain:
    """[Legacy] Create an LLM chain that uses OpenAI functions.

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

                from langchain.chains.openai_functions import create_openai_fn_chain
                from langchain_community.chat_models import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate

                from langchain_core.pydantic_v1 import BaseModel, Field


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


                llm = ChatOpenAI(model="gpt-4", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a world class algorithm for recording entities."),
                        ("human", "Make calls to the relevant function to record the entities in the following input: {input}"),
                        ("human", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_openai_fn_chain([RecordPerson, RecordDog], llm, prompt)
                chain.run("Harry was a chubby brown beagle who loved chicken")
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    openai_functions = [convert_to_openai_function(f) for f in functions]
    output_parser = output_parser or get_openai_output_parser(functions)
    llm_kwargs: Dict[str, Any] = {
        "functions": openai_functions,
    }
    if len(openai_functions) == 1 and enforce_single_function_usage:
        llm_kwargs["function_call"] = {"name": openai_functions[0]["name"]}
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        llm_kwargs=llm_kwargs,
        output_key=output_key,
        **kwargs,
    )
    return llm_chain


@deprecated(
    since="0.1.1", removal="0.2.0", alternative="create_structured_output_runnable"
)
def create_structured_output_chain(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    *,
    output_key: str = "function",
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any,
) -> LLMChain:
    """[Legacy] Create an LLMChain that uses an OpenAI function to get a structured output.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
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

                from langchain.chains.openai_functions import create_structured_output_chain
                from langchain_community.chat_models import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate

                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    \"\"\"Identifying information about a dog.\"\"\"

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a world class algorithm for extracting information in structured formats."),
                        ("human", "Use the given format to extract information from the following input: {input}"),
                        ("human", "Tip: Make sure to answer in the correct format"),
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

            output: output_schema  # type: ignore

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter, attr_name="output"
        )
    return create_openai_fn_chain(
        [function],
        llm,
        prompt,
        output_key=output_key,
        output_parser=output_parser,
        **kwargs,
    )


__all__ = [
    "create_openai_fn_chain",
    "create_openai_fn_runnable",
    "create_structured_output_chain",
    "create_structured_output_runnable",
    "get_openai_output_parser",
    "PYTHON_TO_JSON_TYPES",
    "convert_to_openai_function",
]
