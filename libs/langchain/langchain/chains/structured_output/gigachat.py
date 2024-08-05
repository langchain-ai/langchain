from typing import Any, Dict, Optional, Sequence, Type, Union

from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseOutputParser,
)
from langchain_core.output_parsers.gigachat_functions import (
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
    convert_to_gigachat_function,
)


def create_gigachat_fn_runnable(
    functions: Sequence[Type[BaseModel]],
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
                from langchain_community.chat_models import GigaChat
                from langchain_core.pydantic_v1 import BaseModel, Field


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


                llm = GigaChat(model="GigaChat-Pro")
                structured_llm = create_gigachat_fn_runnable([RecordPerson, RecordDog], llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken)
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    g_functions = [convert_to_gigachat_function(f) for f in functions]
    llm_kwargs_: Dict[str, Any] = {"functions": g_functions, **llm_kwargs}
    if len(g_functions) == 1 and enforce_single_function_usage:
        llm_kwargs_["function_call"] = {"name": g_functions[0]["name"]}
    output_parser = output_parser or get_gigachat_output_parser(functions)
    if prompt:
        return prompt | llm.bind(**llm_kwargs_) | output_parser
    else:
        return llm.bind(**llm_kwargs_) | output_parser


def create_structured_output_runnable(
    output_schema: Union[Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    enforce_function_usage: bool = True,
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
        enforce_function_usage: If True, then the model will be forced to use the given
            output schema. If False, then the model can elect whether to use the output
            schema.
        **kwargs: Additional named arguments.

    Returns:
        A runnable sequence that will return a structured output(s) matching the given
            output_schema.

    OpenAI functions example (mode="openai-functions"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_community.chat_models import GigaChat
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = GigaChat(model="GigaChat-Pro", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")

    Gigachat functions with prompt example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_community.chat_models import GigaChat
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = GigaChat(model="GigaChat-Pro", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm)
                system = '''Extract information about any dogs mentioned in the user input.'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}"),]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    # for backwards compatibility
    force_function_usage = kwargs.get(
        "enforce_single_function_usage", enforce_function_usage
    )

    return _create_gigachat_functions_structured_output_runnable(
        output_schema,
        llm,
        prompt=prompt,
        output_parser=output_parser,
        enforce_single_function_usage=force_function_usage,
        **kwargs,  # llm-specific kwargs
    )


def get_gigachat_output_parser(
    functions: Sequence[Type[BaseModel]],
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
    """Get the appropriate function output parser given the user functions.

    Args:
        functions: Sequence where element is a dictionary, a pydantic.BaseModel class,
            or a Python function. If a dictionary is passed in, it is assumed to
            already be a valid GigaChat function.

    Returns:
        A PydanticOutputFunctionsParser if functions are Pydantic classes, otherwise
            a JsonOutputFunctionsParser. If there's only one function and it is
            not a Pydantic class, then the output parser will automatically extract
            only the function arguments and not the function name.
    """
    if len(functions) > 1:
        pydantic_schema: Union[Dict, Type[BaseModel]] = {
            convert_to_gigachat_function(fn)["name"]: fn for fn in functions
        }
    else:
        pydantic_schema = functions[0]
    output_parser: Union[BaseOutputParser, BaseGenerationOutputParser] = (
        PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    )
    return output_parser


def _create_gigachat_functions_structured_output_runnable(
    output_schema: Union[Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **llm_kwargs: Any,
) -> Runnable:
    class _OutputFormatter(BaseModel):
        """Output formatter. Всегда используй чтобы выдать ответ"""  # noqa: E501

        output: output_schema  # type: ignore

    function = _OutputFormatter
    output_parser = output_parser or PydanticAttrOutputFunctionsParser(
        pydantic_schema=_OutputFormatter, attr_name="output"
    )
    return create_gigachat_fn_runnable(
        [function],
        llm,
        prompt=prompt,
        output_parser=output_parser,
        **llm_kwargs,
    )
