from typing import List

from langchain import PromptTemplate
from langchain.output_parsers.regex import ListRegexParser


class GradeOutputParser(ListRegexParser[int]):
    regex = r"\D*(\d+)"
    _cast = int


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


SINGLE_DOC_TEMPLATE = """\
>> INSTRUCTIONS:
Given a question and a document, score how relevant the document is to the question. \
Return an integer score between 1-5, where 1 means all of the document is completely irrelevant to the question \
and 5 means that some part of the document answers the question exactly. 

*Remember*, a document is considered to be relevant if *ANY* part of the document is relevant. \

>> FORMATTING INSTRUCTIONS:
Return a single integer score. Do not label
the score or add any other text. Do not return a score outside the allowed range.

>> QUESTION: 
{question}
>> CANDIDATE DOCUMENT:

{document}

>> RELEVANCE SCORES:
"""

GRADE_SINGLE_DOC_PROMPT = PromptTemplate(
    input_variables=["question", "document"],
    template=SINGLE_DOC_TEMPLATE,
    output_parser=GradeOutputParser(),
)
