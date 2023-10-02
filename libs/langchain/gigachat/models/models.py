from typing import List

from langchain.pydantic_v1 import BaseModel, Field

from .model import Model


class Models(BaseModel):
    data: List[Model]
    object_: str = Field(alias="object")
