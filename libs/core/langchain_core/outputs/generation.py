"""Generation output schema."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import computed_field

from langchain_core.load import Serializable
from langchain_core.utils._merge import merge_dicts


class Generation(Serializable):
    """A single text generation output.

    Generation represents the response from an "old-fashioned" LLM that
    generates regular text (not chat messages).

    This model is used internally by chat model and will eventually
    be mapped to a more general `LLMResult` object, and then projected into
    an `AIMessage` object.

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks). Please refer the `AIMessage` and `LLMResult` schema documentation
    for more information.
    """

    def __init__(
        self,
        text: str = "",
        generation_info: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize a Generation."""
        super().__init__(generation_info=generation_info, **kwargs)
        self._text = text

    # workaround for ChatGeneration so that we can use a computed field to populate
    # the text field from the message content (parent class needs to have a property)
    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """The text contents of the output."""
        return self._text

    generation_info: Optional[dict[str, Any]] = None
    """Raw response from the provider.

    May include things like the reason for finishing or token log probabilities.
    """

    type: Literal["Generation"] = "Generation"
    """Type is used exclusively for serialization purposes.
    Set to "Generation" for this class."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Default namespace is ["langchain", "schema", "output"].
        """
        return ["langchain", "schema", "output"]


class GenerationChunk(Generation):
    """Generation chunk, which can be concatenated with other Generation chunks."""

    def __init__(
        self,
        text: str = "",
        generation_info: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize a GenerationChunk."""
        super().__init__(text=text, generation_info=generation_info, **kwargs)
        self._text = text

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        """Concatenate two GenerationChunks."""
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
