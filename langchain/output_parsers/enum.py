from enum import Enum
from typing import Any, Type

from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.schema import OutputParserException


class EnumOutputParser(ChoiceOutputParser):
    enum: Type[Enum]

    @classmethod
    def from_enum(cls, enum: Type[Enum], **kwargs):
        assert all(isinstance(e.value, str) for e in enum), "Enum values must be strings"
        return cls(options=[e.value for e in enum], enum=enum, **kwargs)

    def parse(self, response: str) -> Any:
        try:
            selected_value = super().parse(response)
            return self.enum(selected_value)
        except ValueError:
            raise OutputParserException(
                f"Response '{response}' is not one of the expected values: {[e.value for e in self.enum]}"
            )
