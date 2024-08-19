"""Models for the Constitutional AI chain."""

from langchain_core.pydantic_v1 import BaseModel


class ConstitutionalPrinciple(BaseModel):
    """Class for a constitutional principle.

    `Constitutional AI principles` are based on the
    [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073) paper.
    """  # noqa: E501

    critique_request: str
    revision_request: str
    name: str = "Constitutional Principle"
