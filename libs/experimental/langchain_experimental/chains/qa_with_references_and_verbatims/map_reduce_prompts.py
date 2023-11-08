# flake8: noqa

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser

from .verbatims import VerbatimsFromDoc, verbatims_parser, Verbatims, empty_value

EXAMPLE_PROMPT = PromptTemplate(
    template="Ids: {_idx}\n" "Content: {page_content}\n",
    input_variables=["page_content", "_idx"],
)

_map_verbatim_parser = LineListOutputParser()

_question_prompt_template = """
Use the following portion of a long document to see if any of the text is relevant to answer the question.
---
{context}
---

Question: {question}

Extract all verbatims from texts relevant to answering the question in separate strings else output an empty array.
If you are not confident with your answer, say '{empty_value}'. 
{format_instructions}

"""

QUESTION_PROMPT = PromptTemplate(
    template=_question_prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": _map_verbatim_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=_map_verbatim_parser,
)

_combine_prompt_template = """Given the following extracts from several documents, 
a question and not prior knowledge. 

Process step by step:
- extract all verbatims
- extract all associated ids
- create a final response with these verbatims
- If you are not confident with your answer, say '{empty_value}'. 
- produces the json result

QUESTION: {question}
=========
{summaries}
=========
If you are not confident with your answer, say 'I don't know'. 
{format_instructions}
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=_combine_prompt_template,
    input_variables=["summaries", "question"],
    partial_variables={
        "format_instructions": verbatims_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=verbatims_parser,
)
