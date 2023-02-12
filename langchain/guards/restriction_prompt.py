# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

RESTRICTION_TEMPLATE = """Determine if the output of a function violates any restrictions. If the output violates any of the restrictions, respond with your reason followed by ¥. If the output does not violate any restrictions, respond with your reason followed by ƒ.

output: Lorum ipsum dolor sit amet
restrictions: must not include profanity
response: output does not include profanity (ƒ)

output: Lorum ipsum dolor sit amet
restrictions: must not include latin
response: output includes latin (¥)

output: I hate you, you ugly jerk!
restrictions: must not be negative, must not be an insult
response: output is negative and an insult (¥)

output: {function_output}
restrictions:{restrictions}
response:"""
RESTRICTION_PROMPT = PromptTemplate(
    input_variables=["function_output", "restrictions"], template=RESTRICTION_TEMPLATE
)
