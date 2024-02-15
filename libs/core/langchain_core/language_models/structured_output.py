from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable

_OutputSchema = TypeVar("_OutputSchema")


class StructuredOutputMixin(Generic[_OutputSchema], ABC):
    """Mixin for language models that offer native output formatting."""

    @abstractmethod
    def with_structured_output(
        self, schema: _OutputSchema, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Implement this if there is a way of steering the model to generate responses that match a given schema."""  # noqa: E501
