from typing import List, Optional

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration


class ChatResult(BaseModel):
    """Class that contains all results for a single chat model call."""

    generations: List[ChatGeneration]
    """List of the chat generations. This is a List because an input can have multiple 
        candidate generations.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""
