from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.messages import BaseMessage, BaseMessageChunk


class Generation(Serializable):
    """A single text generation output."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw response from the provider. May include things like the 
        reason for finishing or token log probabilities.
    """
    type: Literal["Generation"] = "Generation"
    """Type is used exclusively for serialization purposes."""
    # TODO: add log probs as separate attribute

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True


class GenerationChunk(Generation):
    """A Generation chunk, which can be concatenated with other Generation chunks."""

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        if isinstance(other, GenerationChunk):
            generation_info = (
                {**(self.generation_info or {}), **(other.generation_info or {})}
                if self.generation_info is not None or other.generation_info is not None
                else None
            )
            return GenerationChunk(
                text=self.text + other.text,
                generation_info=generation_info,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )


class ChatGeneration(Generation):
    """A single chat generation output."""

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
    message: BaseMessage
    """The message output by the chat model."""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        try:
            values["text"] = values["message"].content
        except (KeyError, AttributeError) as e:
            raise ValueError("Error while initializing ChatGeneration") from e
        return values


class ChatGenerationChunk(ChatGeneration):
    """A ChatGeneration chunk, which can be concatenated with other
      ChatGeneration chunks.

    Attributes:
        message: The message chunk output by the chat model.
    """

    message: BaseMessageChunk
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment] # noqa: E501
    """Type is used exclusively for serialization purposes."""

    def __add__(self, other: ChatGenerationChunk) -> ChatGenerationChunk:
        if isinstance(other, ChatGenerationChunk):
            generation_info = (
                {**(self.generation_info or {}), **(other.generation_info or {})}
                if self.generation_info is not None or other.generation_info is not None
                else None
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )


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
