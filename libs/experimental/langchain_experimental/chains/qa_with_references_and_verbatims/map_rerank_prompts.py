# flake8: noqa
from langchain.output_parsers import PydanticOutputParser, RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

from .verbatims import VerbatimsFromDoc, Verbatims, verbatims_parser, empty_value

_rank_parser = RegexParser(
    regex=r"(.*)\n?Score: (\d*)",
    output_keys=["answer", "score"],
)

prompt_template = """
Given the following extracts from several documents, a question and not prior knowledge. 

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

The ids must be only in the form '_idx_<number>'.

Process step by step:
- extract the references ("IDS")
- extract all the verbatims from the texts only if they are relevant to answering the question, in a list of strings 
- answers the question
- If you are not confident with your answer, say '{empty_value}'. 
- calculates a score of how fully it answered the user's question
- creates a final answer

The ids must be only in the form '_idx_<number>'.
{format_instructions}

This should be in the following format:
Question: [question here]
Helpful Answer: [json answer here]
Score: [to the next line, score between 0 and 100]

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": verbatims_parser.get_format_instructions(),
        "empty_value": empty_value,
    },
    output_parser=_rank_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nIds: {_idx}",
    input_variables=["page_content", "_idx"],
)
