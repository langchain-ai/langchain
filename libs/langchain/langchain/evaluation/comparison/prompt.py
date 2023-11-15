"""Prompts for comparing the outputs of two models for a given question.

This prompt is used to compare two responses and evaluate which one best follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""
# flake8: noqa
from langchain.prompts.chat import ChatPromptTemplate

SYSTEM_MESSAGE = 'Please act as an impartial judge and evaluate the quality \
of the responses provided by two AI assistants to the user question displayed below. \
You should choose the assistant that follows the user\'s instructions \
and answers \the user\'s question better. \
Your evaluation should consider factors such as the \
helpfulness, relevance, accuracy, depth, creativity, \
and level of detail of their responses. \
Begin your evaluation by comparing the two responses and provide a short explanation. \
Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. \
Do not allow the length of the responses to influence your evaluation. \
Do not favor certain names of the assistants. Be as objective as possible. \
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.'

CRITERIA_INSTRUCTIONS = (
    "For this evaluation, you should primarily consider the following criteria:\n"
)

COMPARISON_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "{criteria}[User Question]\n{input}\n\n\
[The Start of Assistant A's Answer]\n{prediction}\n\
[The End of Assistant A's Answer]\
\n\n[The Start of Assistant B's Answer]\n{prediction_b}\n\
[The End of Assistant B's Answer]",
        ),
    ]
)

COMPARISON_TEMPLATE_WITH_REFERENCE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "{criteria}\n\nTo help you evaluate the responses, \
here is a reference answer to the user's question:\n\
{reference}\
[User Question]\n{input}\n\n\
[The Start of Assistant A's Answer]\n{prediction}\n\
[The End of Assistant A's Answer]\
\n\n[The Start of Assistant B's Answer]\n{prediction_b}\n\
[The End of Assistant B's Answer]",
        ),
    ]
)
