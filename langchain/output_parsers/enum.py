from enum import Enum
from typing import Any, Dict, List, Type

from pydantic import root_validator

from langchain.schema import BaseOutputParser, OutputParserException


class EnumOutputParser(BaseOutputParser):
    enum: Type[Enum]

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
