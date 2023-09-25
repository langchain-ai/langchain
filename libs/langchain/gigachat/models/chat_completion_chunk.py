from typing import List

from langchain.pydantic_v1 import BaseModel, Field

from .choices_chunk import ChoicesChunk


class ChatCompletionChunk(BaseModel):
    choices: List[ChoicesChunk]
    created: int
    model: str
    object_: str = Field(alias="object")
