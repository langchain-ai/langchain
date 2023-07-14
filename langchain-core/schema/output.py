from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, root_validator

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage


class Generation(Serializable):
    """A single text generation output."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw response from the provider. May include things like the 
        reason for finishing or token log probabilities.
    """
    # TODO: add log probs as separate attribute

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is LangChain serializable."""
        return True


class ChatGeneration(Generation):
    """A single chat generation output."""

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
    message: BaseMessage
    """The message output by the chat model."""

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        values["text"] = values["message"].content
        return values


class RunInfo(BaseModel):
    """Class that contains metadata for a single execution of a Chain or model."""

    run_id: UUID
    """A unique identifier for the model or chain run."""


class ChatResult(BaseModel):
    """Class that contains all results for a single chat model call."""

    generations: List[ChatGeneration]
    """List of the chat generations. This is a List because an input can have multiple 
        candidate generations.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


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
