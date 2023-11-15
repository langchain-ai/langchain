from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from langchain_experimental.data_anonymizer.deanonymizer_mapping import MappingDataType
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    exact_matching_strategy,
)

DEFAULT_DEANONYMIZER_MATCHING_STRATEGY = exact_matching_strategy


class AnonymizerBase(ABC):
    """
    Base abstract class for anonymizers.
    It is public and non-virtual because it allows
        wrapping the behavior for all methods in a base class.
    """

    def anonymize(
        self,
        text: str,
        language: Optional[str] = None,
        allow_list: Optional[List[str]] = None,
    ) -> str:
        """Anonymize text"""
        return self._anonymize(text, language, allow_list)

    @abstractmethod
    def _anonymize(
        self, text: str, language: Optional[str], allow_list: Optional[List[str]] = None
    ) -> str:
        """Abstract method to anonymize text"""


class ReversibleAnonymizerBase(AnonymizerBase):
    """
    Base abstract class for reversible anonymizers.
    """

    def deanonymize(
        self,
        text_to_deanonymize: str,
        deanonymizer_matching_strategy: Callable[
            [str, MappingDataType], str
        ] = DEFAULT_DEANONYMIZER_MATCHING_STRATEGY,
    ) -> str:
        """Deanonymize text"""
        return self._deanonymize(text_to_deanonymize, deanonymizer_matching_strategy)

    @abstractmethod
    def _deanonymize(
        self,
        text_to_deanonymize: str,
        deanonymizer_matching_strategy: Callable[[str, MappingDataType], str],
    ) -> str:
        """Abstract method to deanonymize text"""

    @abstractmethod
    def reset_deanonymizer_mapping(self) -> None:
        """Abstract method to reset deanonymizer mapping"""
