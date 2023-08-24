"""Load data generation chain."""

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.synthetic_data.prompts import SENTENCE_PROMPT
from langchain.prompts import PromptTemplate


def create_data_generation_chain(
    llm: BaseLanguageModel,
    prompt: Optional[PromptTemplate] = None,
) -> Chain:
    """Creates a chain that generates synthetic sentences with
     provided fields.
     
    Args:
        llm: The language model to use.
        verbose: Whether to print the prompt or not.
    """
    prompt = prompt or SENTENCE_PROMPT
    return LLMChain(
        llm=llm,
        prompt=prompt,
    )


class DatasetGenerator():
    """Generates synthetic dataset with a given language model."""
    
    def __init__(self, llm: BaseLanguageModel, verbose: Optional[bool] = None):
        self.generator = create_data_generation_chain(llm, verbose)
        
    def __call__(self,  fields_collection: List[List[Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for fields in fields_collection:
            results.append(self.generator(fields))
        return results
