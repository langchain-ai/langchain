from pydantic import BaseModel


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: str
    description: str
    type: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        frozen = True
