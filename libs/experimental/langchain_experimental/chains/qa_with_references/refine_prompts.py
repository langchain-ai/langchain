# flake8: noqa
from langchain.prompts import PromptTemplate

from .references import references_parser, empty_value

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\n" "ids: {_idx}\n",
    input_variables=["page_content", "_idx"],
)

_initial_qa_prompt_template = """
Context information is below. 
---------------------
{context_str}
---------------------
Process step by step:
- ignore prior knowledge
- extract references ("IDS")
- create a final answer
- produce the json result

Given the context information the question: {question}

If you don't know the answer, just say '{empty_value}', without references IDS. 
Don't try to make up an answer.

The ids must be only in the form '_idx_<number>'.
{format_instructions}
"""

INITIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "question"],
    template=_initial_qa_prompt_template,
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=references_parser,
)

_refine_prompt_template = """
Given the context information and not prior knowledge answer, the question: {question}

We have provided an existing JSON answer with the list of documents with associated ids: 
{existing_answer}

We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_str}
------------
Given the new context, refine the original answer to better answer the question. 
If you don't know how to refine the original answer, does not modify the answer.

Process step by step:
- use only the context to answer to refine the original answer to better answer the question. ONLY if you do update it,
append the new IDS from the existing answser IDS as well. 
- produce the result

ALWAYS return a "IDS" part in your answer. 
If the context isn't useful, return the original answer and the original IDS.

If you don't know the answer, just say '{empty_value}'. Don't try to make up an answer.

{format_instructions}
"""

REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=_refine_prompt_template,
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=references_parser,
)
