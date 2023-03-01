from typing import List
from langchain.schema import ChatGeneration


def get_buffer_string(messages: List[ChatGeneration]):
    """Get buffer string of messages."""
    return "\n".join([f"{gen.role}: {gen.text}" for gen in messages])
