from typing import Any, List, Optional

from pydantic import BaseModel


class Validation(BaseModel):
    """Connery Action parameter validation model."""

    required: Optional[bool] = None


class Parameter(BaseModel):
    """Connery Action parameter model."""

    key: str
    title: str
    description: Optional[str] = None
    type: Any
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
