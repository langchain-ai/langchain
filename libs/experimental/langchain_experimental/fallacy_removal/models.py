"""Models for the Logical Fallacy Chain"""
from langchain_experimental.pydantic_v1 import BaseModel


class LogicalFallacy(BaseModel):
    """Logical fallacy."""

    fallacy_critique_request: str
    fallacy_revision_request: str
    name: str = "Logical Fallacy"
