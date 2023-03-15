from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from langchain.schema import Fixer, Guardrail, PromptValue, ValidationError


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


class OutputGuardrail(Guardrail, BaseModel):
    output_parser: BaseOutputParser
    fixer: Fixer

    def check(
        self, prompt_value: PromptValue, result: Any
    ) -> Optional[ValidationError]:
        try:
            self.output_parser.parse(result)
            return None
        except Exception as e:
            return ValidationError(text=e)

    def fix(
        self, prompt_value: PromptValue, result: Any, error: ValidationError
    ) -> Any:
        return self.fix(prompt_value, result, error)
