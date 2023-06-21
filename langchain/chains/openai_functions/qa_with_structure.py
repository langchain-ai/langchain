from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel, Field

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
from langchain.output_parsers.openai_functions import (
    OutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseLLMOutputParser, HumanMessage, SystemMessage


class AnswerWithSources(BaseModel):
    """An answer to the question being asked, with sources."""

    answer: str = Field(..., description="Answer to the question that was asked")
    sources: List[str] = Field(
        ..., description="List of sources used to answer the question"
    )


def create_qa_with_structure_chain(
    llm: BaseLanguageModel,
    schema: Union[dict, Type[BaseModel]],
    output_parser: str = "base",
    prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]] = None,
) -> LLMChain:
    if output_parser == "pydantic":
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(
                "Must provide a pydantic class for schema when output_parser is "
                "'pydantic'."
            )
        _output_parser: BaseLLMOutputParser = PydanticOutputFunctionsParser(
            pydantic_schema=schema
        )
    elif output_parser == "base":
        _output_parser = OutputFunctionsParser()
    else:
        raise ValueError(
            f"Got unexpected output_parser: {output_parser}. "
            f"Should be one of `pydantic` or `base`."
        )
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.schema()
    else:
        schema_dict = schema
    function = {
        "name": schema_dict["title"],
        "description": schema_dict["description"],
        "parameters": schema_dict,
    }
    llm_kwargs = get_llm_kwargs(function)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions in a specific format."
            )
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = prompt or ChatPromptTemplate(messages=messages)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=_output_parser,
    )
    return chain


def create_qa_with_sources_chain(llm: BaseLanguageModel, **kwargs: Any) -> LLMChain:
    return create_qa_with_structure_chain(llm, AnswerWithSources, **kwargs)
