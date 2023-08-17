from typing import Any, List, Optional, Type, Union

from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
from langchain.output_parsers.openai_functions import (
    OutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseLLMOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import HumanMessage, SystemMessage


class AnswerWithSources(BaseModel):
    """An answer to the question, with sources."""

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
    """Create a question answering chain that returns an answer with sources
     based on schema.

    Args:
        llm: Language model to use for the chain.
        schema: Pydantic schema to use for the output.
        output_parser: Output parser to use. Should be one of `pydantic` or `base`.
            Default to `base`.
        prompt: Optional prompt to use for the chain.

    Returns:

    """
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
    """Create a question answering chain that returns an answer with sources.

    Args:
        llm: Language model to use for the chain.
        **kwargs: Keyword arguments to pass to `create_qa_with_structure_chain`.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    return create_qa_with_structure_chain(llm, AnswerWithSources, **kwargs)
