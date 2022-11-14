# flake8: noqa
from langchain.prompts.prompt import Prompt
_TEMPLATE = """I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:

{examples}

The instruction was to"""

PROMPT = Prompt(input_variables=["examples"], template=_TEMPLATE)
