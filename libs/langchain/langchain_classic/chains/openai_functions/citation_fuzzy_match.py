from collections.abc import Iterator

from langchain_core._api import deprecated
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.openai_functions.utils import get_llm_kwargs


class FactWithEvidence(BaseModel):
    """Class representing a single statement.

    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart
    such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(..., description="Body of the sentence, as part of a response")
    substring_quote: list[str] = Field(
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
        """Get spans of the substring quote in the context.

        Args:
            context: The context in which to find the spans of the substring quote.

        Returns:
            An iterator over the spans of the substring quote in the context.
        """
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)


class QuestionAnswer(BaseModel):
    """A question and its answer as a list of facts.

    Each fact should have a source.
    Each sentence contains a body and a list of sources.
    """

    question: str = Field(..., description="Question that was asked")
    answer: list[FactWithEvidence] = Field(
        ...,
        description=(
            "Body of the answer, each fact should be "
            "its separate object with a body and a list of sources"
        ),
    )


def create_citation_fuzzy_match_runnable(llm: BaseChatModel) -> Runnable:
    """Create a citation fuzzy match Runnable.

    Example usage:

        ```python
        from langchain_classic.chains import create_citation_fuzzy_match_runnable
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini")

        context = "Alice has blue eyes. Bob has brown eyes. Charlie has green eyes."
        question = "What color are Bob's eyes?"

        chain = create_citation_fuzzy_match_runnable(model)
        chain.invoke({"question": question, "context": context})
        ```

    Args:
        llm: Language model to use for the chain. Must implement bind_tools.

    Returns:
        Runnable that can be used to answer questions with citations.

    """
    if type(llm).bind_tools is BaseChatModel.bind_tools:
        msg = "Language model must implement bind_tools to use this function."
        raise ValueError(msg)
    prompt = ChatPromptTemplate(
        [
            SystemMessage(
                "You are a world class algorithm to answer "
                "questions with correct and exact citations.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Answer question using the following context."
                "\n\n{context}"
                "\n\nQuestion: {question}"
                "\n\nTips: Make sure to cite your sources, "
                "and use the exact words from the context.",
            ),
        ],
    )
    return prompt | llm.with_structured_output(QuestionAnswer)


@deprecated(
    since="0.2.13",
    removal="1.0",
    alternative="create_citation_fuzzy_match_runnable",
)
def create_citation_fuzzy_match_chain(llm: BaseLanguageModel) -> LLMChain:
    """Create a citation fuzzy match chain.

    Args:
        llm: Language model to use for the chain.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=QuestionAnswer)
    schema = QuestionAnswer.model_json_schema()
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
            ),
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(
            content=(
                "Tips: Make sure to cite your sources, "
                "and use the exact words from the context."
            ),
        ),
    ]
    prompt = ChatPromptTemplate(messages=messages)  # type: ignore[arg-type]

    return LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
    )
