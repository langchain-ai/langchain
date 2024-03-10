from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from langchain_experimental.synthetic_data.prompts import SENTENCE_PROMPT


def create_data_generation_chain(
    llm: BaseLanguageModel,
    prompt: Optional[PromptTemplate] = None,
) -> Chain:
    """Create a chain that generates synthetic sentences with
     provided fields.

    Args:
        llm: The language model to use.
        prompt: Prompt to feed the language model with.
        If not provided, the default one will be used.
    """
    prompt = prompt or SENTENCE_PROMPT
    return LLMChain(
        llm=llm,
        prompt=prompt,
    )


class DatasetGenerator:
    """Generate synthetic dataset with a given language model."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        sentence_preferences: Optional[Dict[str, Any]] = None,
    ):
        self.generator = create_data_generation_chain(llm)
        self.sentence_preferences = sentence_preferences or {}

    def __call__(self, fields_collection: List[List[Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for fields in fields_collection:
            results.append(
                self.generator(
                    {"fields": fields, "preferences": self.sentence_preferences}
                )
            )
        return results
