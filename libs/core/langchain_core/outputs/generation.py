from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langchain_core.load import Serializable


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
