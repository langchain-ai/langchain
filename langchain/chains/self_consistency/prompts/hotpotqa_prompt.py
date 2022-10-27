"""Prompt for HotPotQA."""
# From https://arxiv.org/pdf/2203.11171.pdf

from langchain.prompt import Prompt

_PROMPT_TEMPLATE = """Q: Which magazine was started first Arthur’s Magazine or First for Women?
A: Arthur’s Magazine started in 1844. First for Women started in 1989. So Arthur’s Magazine was started first.
The answer is Arthur’s Magazine.
Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: The Oberoi family is part of the hotel company called The Oberoi Group. The Oberoi Group has its head
office in Delhi. The answer is Delhi.
Q: What nationality was James Henry Miller’s wife?
A: James Henry Miller’s wife is June Miller. June Miller is an American. The answer is American.
Q: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?
A: "House of Anubis" is based on the Dutch–Belgian television series Het Huis Anubis. Het Huis Anubis is first
aired in September 2006. The answer is 2006.
Q: {question} Reason step-by-step.
A:"""

HOTPOTQA_PROMPT = Prompt(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)
