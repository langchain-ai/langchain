# flake8: noqa

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser
from .references import references_parser, References, empty_value

_map_verbatim_parser = LineListOutputParser()

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\n" "Ids: {_idx}",
    input_variables=["page_content", "_idx"],
)

_question_prompt_template = """
Use the following portion of a long document to see if any of the text is relevant to answer the question.
---
{context}
---

Question: {question}

Extract all verbatims from texts relevant to answering the question in separate strings else output an empty array.
{format_instructions}

"""

QUESTION_PROMPT = PromptTemplate(
    template=_question_prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": _map_verbatim_parser.get_format_instructions()
    },
    output_parser=_map_verbatim_parser,
)

_combine_prompt_template = """Given the following extracts from several documents, 
a question and not prior knowledge. 

QUESTION: {question}
=========
{summaries}
=========
If you are not confident with your answer, say '{empty_value}'. 
{format_instructions}
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=_combine_prompt_template,
    input_variables=["summaries", "question"],
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "empty_value": empty_value,
        # "response_example_1": str(_response_example_1),
        # "response_example_2": str(_response_example_2),
    },
    output_parser=references_parser,
)
