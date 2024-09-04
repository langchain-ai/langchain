"""Models for the Constitutional AI chain."""

from langchain_core.pydantic_v1 import BaseModel


class ConstitutionalPrinciple(BaseModel):
    """Class for a constitutional principle."""

    critique_request: str
    revision_request: str
    name: str = "Constitutional Principle"
