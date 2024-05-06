from enum import Enum
from typing import Any, Dict, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import root_validator


class EnumOutputParser(BaseOutputParser):
    """Parse an output that is one of a set of values."""

    enum: Type[Enum]
    """The enum to parse. Its values must be strings."""

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        enum = values["enum"]
        if not all(isinstance(e.value, str) for e in enum):
            raise ValueError("Enum values must be strings")
        return values

    @property
    def _valid_values(self) -> List[str]:
        return [e.value for e in self.enum]

    def parse(self, response: str) -> Any:
        try:
            return self.enum(response.strip())
        except ValueError:
            raise OutputParserException(
                f"Response '{response}' is not one of the "
                f"expected values: {self._valid_values}"
            )

    def get_format_instructions(self) -> str:
        return f"Select one of the following options: {', '.join(self._valid_values)}"
