from typing import Iterator, List

from pydantic import BaseModel, Field

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
from langchain.output_parsers.openai_functions import (
    PydanticOutputFunctionsParser,
)
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage


class FactWithEvidence(BaseModel):
    """Class representing single statement.

    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart
    such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(..., description="Body of the sentence, as part of a response")
    substring_quote: List[str] = Field(
        ...,
        description=(
            "Each source should be a direct quote from the context, "
            "as a substring of the original content"
        ),
    )

    def _get_span(self, quote: str, context: str, errs: int = 100) -> Iterator[str]:
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context: str) -> Iterator[str]:
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)


class QuestionAnswer(BaseModel):
    """A question and its answer as a list of facts each one should have a source.
    each sentence contains a body and a list of sources."""

    question: str = Field(..., description="Question that was asked")
    answer: List[FactWithEvidence] = Field(
        ...,
        description=(
            "Body of the answer, each fact should be "
            "its separate object with a body and a list of sources"
        ),
    )


def create_citation_fuzzy_match_chain(llm: BaseLanguageModel) -> LLMChain:
    """Create a citation fuzzy match chain.

    Args:
        llm: Language model to use for the chain.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=QuestionAnswer)
    schema = QuestionAnswer.schema()
    function = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": schema,
    }
    llm_kwargs = get_llm_kwargs(function)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions with correct and exact citations."
            )
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(
            content=(
                "Tips: Make sure to cite your sources, "
                "and use the exact words from the context."
            )
        ),
    ]
    prompt = ChatPromptTemplate(messages=messages)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
    )
    return chain
