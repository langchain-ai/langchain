from typing import Optional, Sequence

from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.pydantic_v1 import BaseModel, root_validator


class ChatResult(BaseModel):
    """Class that contains all results for a single chat model call."""

    generations: Sequence[ChatGeneration]
    """List of the chat generations. This is a List because an input can have multiple 
        candidate generations.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""

    @root_validator(pre=True)
    def validate_environment(self, values: dict) -> dict:
        values["generations"] = list(values.get("generations", ()))
        return values
