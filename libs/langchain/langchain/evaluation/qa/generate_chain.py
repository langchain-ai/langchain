"""LLM Chain for generating examples for question answering."""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.pydantic_v1 import Field

from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.generate_prompt import PROMPT
from langchain.output_parsers.regex import RegexParser

_QA_OUTPUT_PARSER = RegexParser(
    regex=r"QUESTION: (.*?)\n+ANSWER: (.*)", output_keys=["query", "answer"]
)


@deprecated(
    since="0.2.13",
    message=(
        "This class is deprecated and will be removed in langchain 1.0. "
        "See API reference for replacement: "
        "https://api.python.langchain.com/en/latest/evaluation/langchain.evaluation.qa.generate_chain.QAGenerateChain.html"  # noqa: E501
    ),
    removal="1.0",
)
class QAGenerateChain(LLMChain):
    """LLM Chain for generating examples for question answering.

    Note: this class is deprecated. See below for a replacement implementation
        that leverages LLM tool calling features.

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from typing_extensions import TypedDict

            template = \"\"\"You are a teacher coming up with questions to ask on a quiz.
            Given the following document, please generate a question and answer based on that document.

            These questions should be detailed and be based explicitly on information in the document.
            \"\"\"

            prompt = ChatPromptTemplate.from_template(template)

            class QuestionAndAnswer(TypedDict):
                \"\"\"Question and answer based on document.\"\"\"
                question: str
                answer: str

            llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(QuestionAndAnswer)
            llm.invoke("...")

    """  # noqa: E501

    output_parser: BaseLLMOutputParser = Field(default=_QA_OUTPUT_PARSER)
    output_key: str = "qa_pairs"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> QAGenerateChain:
        """Load QA Generate Chain from LLM."""
        return cls(llm=llm, prompt=PROMPT, **kwargs)
