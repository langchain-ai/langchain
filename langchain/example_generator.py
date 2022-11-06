"""Utility functions for working with prompts."""
from typing import List

from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.dynamic import DynamicPrompt

TEST_GEN_TEMPLATE_SUFFIX = "Add another example."


def generate_example(examples: List[str], llm: LLM) -> str:
    """Return another example given a list of examples for a prompt."""
    prompt = DynamicPrompt(examples=examples, suffix=TEST_GEN_TEMPLATE_SUFFIX)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict()


def generate_example_from_dynamic_prompt(prompt: DynamicPrompt, llm: LLM) -> str:
    """Return another example given a DynamicPrompt object."""
    return generate_example(prompt.examples, llm)
