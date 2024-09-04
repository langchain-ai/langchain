from langchain_core.pydantic_v1 import BaseModel


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: str
    description: str
    type: str

    class Config:
        arbitrary_types_allowed = True
        frozen = True
