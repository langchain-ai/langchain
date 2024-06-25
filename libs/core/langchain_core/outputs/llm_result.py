from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

from langchain_core.outputs.generation import Generation
from langchain_core.outputs.run_info import RunInfo
from langchain_core.pydantic_v1 import BaseModel


class LLMResult(BaseModel):
    """A container for results of an LLM call.

    Both chat models and LLMs generate an LLMResult object. This object contains
    the generated outputs and any additional information that the model provider
    wants to return.
    """

    generations: List[List[Generation]]
    """Generated outputs.
    
    The first dimension of the list represents completions for different input
    prompts.
    
    The second dimension of the list represents different candidate generations
    for a given prompt.
    
    When returned from an LLM the type is List[List[Generation]].
    When returned from a chat model the type is List[List[ChatGeneration]].
    
    ChatGeneration is a subclass of Generation that has a field for a structured
    chat message.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output.
    
    This dictionary is a free-form dictionary that can contain any information that the
    provider wants to return. It is not standardized and is provider-specific.
    
    Users should generally avoid relying on this field and instead rely on
    accessing relevant information from standardized fields present in
    AIMessage.
    """
    run: Optional[List[RunInfo]] = None
    """List of metadata info for model call for each input."""

    def flatten(self) -> List[LLMResult]:
        """Flatten generations into a single list.

        Unpack List[List[Generation]] -> List[LLMResult] where each returned LLMResult
            contains only a single Generation. If token usage information is available,
            it is kept only for the LLMResult corresponding to the top-choice
            Generation, to avoid over-counting of token usage downstream.

        Returns:
            List of LLMResults where each returned LLMResult contains a single
                Generation.
        """
        llm_results = []
        for i, gen_list in enumerate(self.generations):
            # Avoid double counting tokens in OpenAICallback
            if i == 0:
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=self.llm_output,
                    )
                )
            else:
                if self.llm_output is not None:
                    llm_output = deepcopy(self.llm_output)
                    llm_output["token_usage"] = dict()
                else:
                    llm_output = None
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=llm_output,
                    )
                )
        return llm_results

    def __eq__(self, other: object) -> bool:
        """Check for LLMResult equality by ignoring any metadata related to runs."""
        if not isinstance(other, LLMResult):
            return NotImplemented
        return (
            self.generations == other.generations
            and self.llm_output == other.llm_output
        )
