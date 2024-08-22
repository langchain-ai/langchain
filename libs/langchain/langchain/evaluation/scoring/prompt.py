"""Prompts for scoring the outputs of a models for a given question.

This prompt is used to score the responses and evaluate how it follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""

# flake8: noqa
from langchain_core.prompts.chat import ChatPromptTemplate

SYSTEM_MESSAGE = "You are a helpful assistant."

CRITERIA_INSTRUCTIONS = (
    "For this evaluation, you should primarily consider the following criteria:\n"
)

DEFAULT_CRITERIA = " Your evaluation \
should consider factors such as the helpfulness, relevance, accuracy, \
depth, creativity, and level of detail of the response."

SCORING_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            '[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}Begin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer]',
        ),
    ]
)

SCORING_TEMPLATE_WITH_REFERENCE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}"
            '[Ground truth]\n{reference}\nBegin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer]',
        ),
    ]
)
