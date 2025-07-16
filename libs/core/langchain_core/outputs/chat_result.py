"""Chat result schema."""

from typing import Optional

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration


class ChatResult(BaseModel):
    """Use to represent the result of a chat model call with a single prompt.

    This container is used internally by some implementations of chat model,
    it will eventually be mapped to a more general `LLMResult` object, and
    then projected into an `AIMessage` object.

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks). Please refer the `AIMessage` and `LLMResult` schema documentation
    for more information.
    """

    generations: list[ChatGeneration]
    """List of the chat generations.

    Generations is a list to allow for multiple candidate generations for a single
    input prompt.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output.

    This dictionary is a free-form dictionary that can contain any information that the
    provider wants to return. It is not standardized and is provider-specific.

    Users should generally avoid relying on this field and instead rely on
    accessing relevant information from standardized fields present in
    AIMessage.
    """
