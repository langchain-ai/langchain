# flake8: noqa
# from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

from langchain.prompts.prompt import PromptTemplate

template = """You are GPT-3. 
For questions that require algorithmic or mathematical calculations, model the problem step by step and reply with the following format:

Question: ${{Question}}
Follow up with PAL: ${{Question}}
PAL return: ${{Answer}}

Otherwise, use this format:

Question: ${{Question}}
Answer: ${{Answer}}

Examples:

Question: If in 2022 I'm 37 years old, how old I was in 1985?
Follow up with PAL: If in 2022 I'm 37 years old, how old I was in 1985?
PAL return: 0
So the final answer is: 0

Question: who is the current president of Peru?
Answer: Martín Vizcarra
So the final answer is: Martín Vizcarra

Question: ${input}
"""

PROMPT = PromptTemplate(input_variables=["input"], template=template)
