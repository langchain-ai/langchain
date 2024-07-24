from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain.chains.llm import LLMChain

TEST_GEN_TEMPLATE_SUFFIX = "Add another example."


def generate_example(
    examples: List[dict], llm: BaseLanguageModel, prompt_template: PromptTemplate
) -> str:
    """Return another example given a list of examples for a prompt."""
    prompt = FewShotPromptTemplate(
        examples=examples,
        suffix=TEST_GEN_TEMPLATE_SUFFIX,
        input_variables=[],
        example_prompt=prompt_template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict()
