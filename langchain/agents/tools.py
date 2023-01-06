"""Interface for tools."""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Tool:
    """Interface for tools."""

    name: str
    func: Callable[[str], str]
    description: Optional[str] = None
    return_direct: bool = False
