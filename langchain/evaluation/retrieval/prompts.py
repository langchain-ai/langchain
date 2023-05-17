from typing import List

from langchain import PromptTemplate
from langchain.output_parsers.regex import ListRegexParser


class GradeOutputParser(ListRegexParser[int]):
    regex = r"\D*(\d+)"
    _cast = int


# Largely taken from
# https://github.com/langchain-ai/auto-evaluator/blob/main/api/text_utils.py
WITH_ANSWER_TEMPLATE = """\
Given the question:
{question}

Here are some documents retrieved in response to the question:
{documents}

And here is the answer to the question:
{answer}

Criteria: 
  relevance: Are the retrieved documents relevant to the question and do they support the answer?"

Your response should be as follows:

GRADE: (Comma separated score for each document. Scores are from 1 - 5, where 1 \
indicates the document was not relevant at all and 5 indicates it answered the question exactly.)
"""  # noqa: E501

GRADE_DOCS_WITH_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "documents", "answer"],
    template=WITH_ANSWER_TEMPLATE,
    output_parser=GradeOutputParser(),
)


TEMPLATE = """\
>> INSTRUCTIONS:
Given a question and a list of documents, score how relevant each document is to the question. \
Return integer scores between 1-5, where 1 means the document is completely irrelevant to the question \
and 5 means the document answers the question exactly.

>> FORMATTING INSTRUCTIONS:
Return a comma separated list of scores, with one score for each document. Do not label
the scores or add any other text. Do not return a score outside the allowed range.

>> QUESTION: 
{question}
>> CANDIDATE DOCUMENTS:

{documents}

>> RELEVANCE SCORES:
"""

GRADE_DOCS_PROMPT = PromptTemplate(
    input_variables=["question", "documents"],
    template=TEMPLATE,
    output_parser=GradeOutputParser(),
)
