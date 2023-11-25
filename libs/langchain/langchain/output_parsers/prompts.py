# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

NAIVE_FIX = """Instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"""


NAIVE_FIX_PROMPT = PromptTemplate.from_template(NAIVE_FIX)
