from langchain.pydantic_v1 import BaseModel, Field


class Model(BaseModel):
    id_: str = Field(alias="id")
    object_: str = Field(alias="object")
    owned_by: str
