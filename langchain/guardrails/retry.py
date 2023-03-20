from typing import Optional

from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.schema import PromptValue


NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_COMPLETION_RETRY_WITH_ERROR = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Details: {error}
Please try again:"""


def naive_retry(llm: BaseLanguageModel, prompt: PromptValue, completion: str, error_msg: Optional[str] = None) -> str:
    """Ask an LLM to re-complete a prompt, given a prompt and unsatisfactory completion."""
    template_vars = {
        "prompt": prompt.to_string(),
        "completion": completion
    }
    if error_msg:
        prompt_template = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY_WITH_ERROR)
        template_vars["error"] = error_msg
    else:
        prompt_template = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain.predict(**{template_vars})
