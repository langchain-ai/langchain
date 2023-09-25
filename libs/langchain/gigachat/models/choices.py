from typing import Optional

from langchain.pydantic_v1 import BaseModel

from .messages_res import MessagesRes


class Choices(BaseModel):
    message: MessagesRes
    index: int
    finish_reason: Optional[str] = None
