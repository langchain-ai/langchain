# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

NAIVE_FUNCTIONS_FIX = """Instructions:
--------------
{instructions}
--------------
Generations:
--------------
{generations}
--------------

Above, the Generations list did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Please only respond with a generation that calls a function and satisfies the constraints laid out in the Instructions:"""


NAIVE_FUNCTIONS_FIX_PROMPT = PromptTemplate.from_template(NAIVE_FUNCTIONS_FIX)


NAIVE_FUNCTIONS_FIX_INSTRUCTIONS = """Return the correct format as an OpenAI functions generation in the additional kwargs.
Do not add extra words to the generated message.
Just interpret the original generation as a function call."""
