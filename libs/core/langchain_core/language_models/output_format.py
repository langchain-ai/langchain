from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable

_OutputSchema = TypeVar("_OutputSchema")
_FormattedOutput = TypeVar("_FormattedOutput")


class FormattedOutputMixin(Generic[_OutputSchema, _FormattedOutput], ABC):
    """Mixing for language models that offer native output formatting."""

    @abstractmethod
    def with_output_format(
        self, schema: _OutputSchema, **kwargs: Any
    ) -> Runnable[LanguageModelInput, _FormattedOutput]:
        """Implement this if there is a way of steering the model to generate responses that match a given schema."""  # noqa: E501
