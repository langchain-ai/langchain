from typing import Optional

from langchain.pydantic_v1 import BaseModel

from .messages_chunk import MessagesChunk


class ChoicesChunk(BaseModel):
    delta: MessagesChunk
    index: int = 0
    finish_reason: Optional[str] = None
