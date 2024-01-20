from typing import (
    Dict,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseOutputParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable

from langchain_google_vertexai.functions_utils import PydanticFunctionsOutputParser


def get_output_parser(
    functions: Sequence[Type[BaseModel]],
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
    """Get the appropriate function output parser given the user functions.

    Args:
        functions: Sequence where element is a dictionary, a pydantic.BaseModel class,
            or a Python function. If a dictionary is passed in, it is assumed to
            already be a valid OpenAI function.

    Returns:
        A PydanticFunctionsOutputParser
    """
    function_names = [f.__name__ for f in functions]
    if len(functions) > 1:
        pydantic_schema: Union[Dict, Type[BaseModel]] = {
            name: fn for name, fn in zip(function_names, functions)
        }
    else:
        pydantic_schema = functions[0]
    output_parser: Union[
        BaseOutputParser, BaseGenerationOutputParser
    ] = PydanticFunctionsOutputParser(pydantic_schema=pydantic_schema)
    return output_parser


def create_structured_runnable(
    function: Union[Type[BaseModel], Sequence[Type[BaseModel]]],
    llm: Runnable,
    *,
    prompt: Optional[BasePromptTemplate] = None,
) -> Runnable:
    """Create a runnable sequence that uses OpenAI functions.

    Args:
        function: Either a single pydantic.BaseModel class or a sequence
            of pydantic.BaseModels classes.
            For best results, pydantic.BaseModels
            should have descriptions of the parameters.
        llm: Language model to use,
            assumed to support the Google Vertex function-calling API.
        prompt: BasePromptTemplate to pass to the model.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain_google_vertexai import ChatVertexAI, create_structured_runnable
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


                llm = ChatVertexAI(model_name="gemini-pro")
                prompt = ChatPromptTemplate.from_template(\"\"\"
                You are a world class algorithm for recording entities.
                Make calls to the relevant function to record the entities in the following input: {input}
                Tip: Make sure to answer in the correct format\"\"\"
                                         )
                chain = create_structured_runnable([RecordPerson, RecordDog], llm, prompt=prompt)
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not function:
        raise ValueError("Need to pass in at least one function. Received zero.")
    functions = function if isinstance(function, Sequence) else [function]
    output_parser = get_output_parser(functions)
    llm_with_functions = llm.bind(functions=functions)
    if prompt is None:
        initial_chain = llm_with_functions
    else:
        initial_chain = prompt | llm_with_functions
    return initial_chain | output_parser
