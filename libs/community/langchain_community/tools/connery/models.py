from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel


class Validation(BaseModel):
    """Connery Action parameter validation model."""

    required: Optional[bool] = None


class Parameter(BaseModel):
    """Connery Action parameter model."""

    key: str
    title: str
    description: Optional[str] = None
    type: str
    validation: Optional[Validation] = None


class Action(BaseModel):
    """Connery Action model."""

    id: str
    key: str
    title: str
    description: Optional[str] = None
    type: str
    inputParameters: List[Parameter]
    outputParameters: List[Parameter]
    pluginId: str
