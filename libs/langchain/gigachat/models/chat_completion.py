from typing import List

from langchain.pydantic_v1 import BaseModel, Field

from .choices import Choices
from .usage import Usage


class ChatCompletion(BaseModel):
    choices: List[Choices]
    created: int
    model: str
    usage: Usage
    object_: str = Field(alias="object")
