"""Prompts for comparing the outputs of two models for a given question.

This prompt is used to compare two responses and evaluate which one best follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""
# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Act as a fair judge and rate the two responses to the question below.\
 Choose the response that best followed the instructions and answered the question.\
 Your assessment should weigh helpfulness, relevance, accuracy, depth, creativity, and detail.\
 Start by comparing both responses and give a brief rationale.\
 Avoid bias from the order of presentation or response length.
After giving your rationale, make your final decision using this format:\
 "[[A]]" if assistant A is better, "[[B]]" if assistant B is better,\
 and "[[C]]" for a tie. Finally, repeat the decision again on its own on a new line.

[QUESTION]
{input}
[/QUESTION]

[RESPONSE A]
{output_a}
[/RESPONSE A]

[RESPONSE B]
{output_b}
[/RESPONSE B]"""
PROMPT = PromptTemplate(
    input_variables=["input", "output_a", "output_b"], template=template
)

ref_template = """Act as a fair judge and rate the two responses to the question below.\
 Choose the response that best followed the instructions and answered the question.\
 Your assessment should weigh helpfulness, relevance, accuracy, depth, creativity, and detail.\
 Start by comparing both responses and give a brief rationale.\
 Avoid bias from the order of presentation or response length.\
 Weigh accuracy based on the following ground truth reference\
 answer to the question:

[REFERENCE]
{reference}
[/REFERENCE]

After giving your rationale, make your final decision using this format:\
 "[[A]]" if assistant A is better, "[[B]]" if assistant B is better,\
 and "[[C]]" for a tie. Finally, repeat the decision again on its own on a new line.

[QUESTION]
{input}
[/QUESTION]

[RESPONSE A]
{output_a}
[/RESPONSE A]

[RESPONSE B]
{output_b}
[/RESPONSE B]"""

PROMPT_WITH_REFERENCE = PromptTemplate(
    input_variables=["input", "output_a", "output_b", "reference"],
    template=ref_template,
)


sim_template = """You are tasked with evaluating whether the two responses to the question below\
 are equivalent in meaning. Start by comparing both responses and give a brief rationale.\
 If the task or question are provided, use them to help determine equivalence.\

[BEGIN DATA]
***
[Question]: {input}
***
[Response 1]: {output_a}
***
[Response 2]: {output_b}
***
[END DATA]

Are the meanings of Response A and Response B the same? Choices are [[A]]: Equivalent, [[B]]: Not Equivalent, [[C]]: Impossible to tell. First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the judgement [[A]] or [[B]] on its own line corresponding to the correct answer. At the end, repeat just the letter again by itself on a new line."""

EQUIVALENCE_PROMPT = PromptTemplate(
    input_variables=["input", "output_a", "output_b"], template=sim_template
)
