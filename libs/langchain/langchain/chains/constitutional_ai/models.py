"""Models for the Constitutional AI chain."""
try:
    from pydantic.v1 import BaseModel
except:
    from pydantic import BaseModel


class ConstitutionalPrinciple(BaseModel):
    """Class for a constitutional principle."""

    critique_request: str
    revision_request: str
    name: str = "Constitutional Principle"
