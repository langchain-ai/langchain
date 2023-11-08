# flake8: noqa
from typing import Type

from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from .references import References

_rank_parser = RegexParser(
    regex=r"(.*)\n?Score: (\d*)",
    output_keys=["answer", "score"],
)


class ReferenceOutputParser(BaseOutputParser[References]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[References] = References
    """The pydantic model to parse."""

    def parse(self, text: str) -> References:
        return References(response=text)

    def get_format_instructions(self) -> str:
        return ""

    @property
    def _type(self) -> str:
        return "reference"


rerank_reference_parser = ReferenceOutputParser()

prompt_template = """
Given the following extracts from several documents, a question and not prior knowledge. 

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Process step by step:
- extract the references ("IDS")
- answers the question
- calculates a score of how fully it answered the user's question
- creates a final answer

The ids must be only in the form '_idx_<number>'.
This should be in the following format:
Question: [question here]
Helpful Answer: [json answer here]
Score: [always to the next line, score between 0 and 100]

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=_rank_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nIds: {_idx}\n",
    input_variables=["page_content", "_idx"],
)
