from typing import Callable, Optional, Union

from langchain.pydantic_v1 import BaseModel


class VirtualColumnName(BaseModel):
    """Virtual column name"""

    name: str
    """name of this virtual column name"""
    column: Optional[str] = None
    """real column name to perform function on"""
    func: Callable[[Optional[str]], str] = lambda x: x if x else ""  # noqa: E731
    """virtual column name only accepts function that operates on column name"""

    def __str__(self) -> str:
        return self.name

    def __call__(self) -> str:
        if self.column:
            return self.func(self.column)
        else:
            return self.func(self.name)


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: Union[str, VirtualColumnName]
    description: str
    type: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        frozen = True
