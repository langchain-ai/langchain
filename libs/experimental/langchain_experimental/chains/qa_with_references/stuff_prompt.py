# flake8: noqa


from langchain.prompts import PromptTemplate
from .references import references_parser, References, empty_value

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\n" "Ids: {_idx}",
    input_variables=["page_content", "_idx"],
)

_template = """Given the following extracts from several documents, a question and not prior knowledge. 
Process step by step:
- for each documents extract the references ("IDS")
- creates a final answer
- produces the json result

If you don't know the answer, just say '{empty_value}'. Don't try to make up an answer.
ALWAYS return a "IDS" part in your answer in another line.

QUESTION: {question}
=========
{summaries}
=========
The ids must be only in the form '_idx_<number>'.
{format_instructions}
FINAL ANSWER: 
"""

PROMPT = PromptTemplate(
    template=_template,
    input_variables=["summaries", "question"],
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=references_parser,
)
