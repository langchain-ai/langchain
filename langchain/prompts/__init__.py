"""Prompt template classes."""
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.loading import load_from_hub, load_prompt
from langchain.prompts.prompt import Prompt, PromptTemplate

__all__ = [
    "BasePromptTemplate",
    "load_prompt",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "Prompt",
    "load_from_hub",
]
