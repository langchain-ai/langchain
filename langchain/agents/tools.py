"""Interface for tools."""
from typing import Callable, NamedTuple, Optional


class Tool(NamedTuple):
    """Interface for tools."""

    name: str
    func: Callable[[str], str]
    description: Optional[str] = None
