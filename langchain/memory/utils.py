from typing import List

from langchain.schema import ChatMessage


def get_buffer_string(messages: List[ChatMessage]) -> str:
    """Get buffer string of messages."""
    return "\n".join([f"{gen.role}: {gen.text}" for gen in messages])
