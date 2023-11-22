from datetime import date
from typing import List, Optional, TypedDict, Union

from langchain.chains.cube.prompt import PROMPT
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.runnable import Runnable, RunnableParallel
from langchain.utilities.cube import CubeAPIWrapper, Query


def _strip(text: str) -> str:
    return text.strip()


class CubeQueryInput(TypedDict):
    """Input for a Cube Query Chain."""

    question: str


class CubeQueryInputWithModels(TypedDict):
    """Input for a Cube Query Chain."""

    question: str
    model_names_to_use: List[str]


def create_cube_query_chain(
    llm: BaseLanguageModel,
    cube: CubeAPIWrapper,
    *,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[CubeQueryInput, CubeQueryInputWithModels], Query]:
    """Chain for question-answering against a Cube Query.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security or https://cube.dev/security
         for more information.
    """
    if prompt is not None:
        prompt_to_use = prompt
    else:
        prompt_to_use = PROMPT

    parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Query)

    inputs = {
        "input": lambda x: x["question"] + "\nCubeQuery: ",
        "top_k": lambda _: k,
        "model_meta_information": lambda x: cube.get_model_meta_information(
            model_names=x.get("model_names_to_use")
        ),
        "format_instructions": lambda _: parser.get_format_instructions(),
        "current_date": lambda _: date.today().isoformat(),
    }

    return (
        RunnableParallel(inputs)
        | prompt_to_use
        | llm.bind(stop=["\nCubeResult:"])
        | PydanticOutputParser(pydantic_object=Query)
    )
