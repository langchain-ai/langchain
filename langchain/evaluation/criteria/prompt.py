# flake8: noqa
# Credit to https://github.com/openai/evals/tree/main

from langchain.prompts import PromptTemplate

template = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Task]: {input}
***
[Submission]: {output}
***
[Criteria]: {criteria}
***
[END DATA]
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the letter again by itself on a new line."""

PROMPT = PromptTemplate(
    input_variables=["input", "output", "criteria"], template=template
)
