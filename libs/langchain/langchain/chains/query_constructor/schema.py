from pydantic import BaseModel, ConfigDict


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: str
    description: str
    type: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
