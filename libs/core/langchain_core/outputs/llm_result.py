from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

from langchain_core.outputs.generation import Generation
from langchain_core.outputs.run_info import RunInfo
from langchain_core.pydantic_v1 import BaseModel


class LLMResult(BaseModel):
    """Class that contains all results for a batched LLM call."""

    generations: List[List[Generation]]
    """List of generated outputs. This is a List[List[]] because
    each input could have multiple candidate generations."""
    llm_output: Optional[dict] = None
    """Arbitrary LLM provider-specific output."""
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
