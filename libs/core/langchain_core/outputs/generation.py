"""Generation output schema."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.load import Serializable
from langchain_core.utils._merge import merge_dicts


class Generation(Serializable):
    """A single text generation output.

    Generation represents the response from an
    `"old-fashioned" LLM <https://python.langchain.com/docs/concepts/text_llms/>__` that
    generates regular text (not chat messages).

    This model is used internally by chat model and will eventually
    be mapped to a more general `LLMResult` object, and then projected into
    an `AIMessage` object.

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks). Please refer the `AIMessage` and `LLMResult` schema documentation
    for more information.
    """

    text: str
    """Generated text output."""

    generation_info: dict[str, Any] | None = None
    """Raw response from the provider.

    May include things like the reason for finishing or token log probabilities.
    """
    type: Literal["Generation"] = "Generation"
    """Type is used exclusively for serialization purposes.
    Set to "Generation" for this class."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "output"]`
        """
        return ["langchain", "schema", "output"]


class GenerationChunk(Generation):
    """Generation chunk, which can be concatenated with other Generation chunks."""

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        """Concatenate two `GenerationChunk`s.

        Args:
            other: Another `GenerationChunk` to concatenate with.

        Raises:
            TypeError: If other is not a `GenerationChunk`.

        Returns:
            A new `GenerationChunk` concatenated from self and other.
        """
        if isinstance(other, GenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return GenerationChunk(
                text=self.text + other.text,
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)
