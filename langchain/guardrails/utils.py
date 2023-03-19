from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate


# TODO: perhaps prompt str -> PromptValue
def dumb_davinci_retry(prompt: str, completion: str) -> str:
    """Big model go brrrr.
    """
    davinci = OpenAI(model_name='text-davinci-003', temperature=0.5)
    retry_prompt = PromptTemplate(
        template="Prompt:\n{prompt}\nCompletion:\n{completion}\n\nAbove, the Completion did not satisfy the constraints given in the Prompt. Please try again:",
        input_variables=["prompt", "completion"]
    )
    retry_prompt_str = retry_prompt.format_prompt(prompt=prompt, completion=completion).to_string()
    return davinci(retry_prompt_str)

