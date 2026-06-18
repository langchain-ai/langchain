# Credit to https://github.com/openai/evals/tree/main

from langchain_core.prompts import PromptTemplate

_VERDICT_INSTRUCTION = (
    "Does the submission meet the Criteria? First, write out in a step by step manner"
    " your reasoning about each criterion to be sure that your conclusion is correct."
    " Avoid simply stating the correct answers at the outset."
    ' Then print only the single character "Y" or "N" (without quotes or'
    " punctuation) on its own line corresponding to the correct answer of whether the"
    " submission meets all criteria. Y means the submission DOES meet the criteria."
    " N means the submission DOES NOT meet the criteria."
    " Make sure your verdict is consistent with your reasoning."
    " At the end, repeat just the letter again by itself on a new line."
)

template = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: {input}
***
[Submission]: {output}
***
[Criteria]: {criteria}
***
[END DATA]
"""  # noqa: E501
template += _VERDICT_INSTRUCTION

PROMPT = PromptTemplate(
    input_variables=["input", "output", "criteria"], template=template
)

template_with_refs = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: {input}
***
[Submission]: {output}
***
[Criteria]: {criteria}
***
[Reference]: {reference}
***
[END DATA]
"""  # noqa: E501
template_with_refs += _VERDICT_INSTRUCTION

PROMPT_WITH_REFERENCES = PromptTemplate(
    input_variables=["input", "output", "criteria", "reference"],
    template=template_with_refs,
)
