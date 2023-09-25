"""Prompts for using the "LLM-as-a-judge" comparison method.

From: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl"""  # noqa: E501
from langchain.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = 'Please act as an impartial judge and evaluate the quality \
of the responses provided by two AI assistants to the user question displayed below. \
You should choose the assistant that follows the user\'s instructions\
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

COMPARISON_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[User Question]\n{input}\n\n\
            [The Start of Assistant A's Answer]\n{prediction}\n\
            [The End of Assistant A's Answer]\
            \n\n[The Start of Assistant B's Answer]\n{prediction_b}\n\
            [The End of Assistant B's Answer]",
        ),
    ]
)
