"""Prompt for the comparison evaluations"""
# flake8: noqa
from langchain.prompts import PromptTemplate

# From Zheng, et. al., https://arxiv.org/abs/2306.05685

template = """Act as a fair judge and rate the two responses to the question below.\
Choose the response that best followed the instructions and answered the question.\
Your assessment should weigh helpfulness, relevance, accuracy, depth, creativity, and detail.\
Start by comparing both responses and give a brief rationale.\
Avoid bias from the order of presentation or response length.\
{reference}\
After giving your rationale, make your final decision using this format:\
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better,\
and "[[C]]" for a tie. Finally, repeat the verdict on its own on a new line.

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
    input_variables=["input", "output_a", "output_b", "reference"], template=template
)
