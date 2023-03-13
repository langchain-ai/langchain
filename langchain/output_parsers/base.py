from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class BaseOutputParser(BaseModel, ABC):
    """Class to parse the output of an LLM call."""

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call."""

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict
