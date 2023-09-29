from abc import ABC, abstractmethod
from typing import Optional


class AnonymizerBase(ABC):
    """
    Base abstract class for anonymizers.
    It is public and non-virtual because it allows
        wrapping the behavior for all methods in a base class.
    """

    def anonymize(self, text: str, language: Optional[str] = None) -> str:
        """Anonymize text"""
        return self._anonymize(text, language)

    @abstractmethod
    def _anonymize(self, text: str, language: Optional[str]) -> str:
        """Abstract method to anonymize text"""


class ReversibleAnonymizerBase(AnonymizerBase):
    """
    Base abstract class for reversible anonymizers.
    """

    def deanonymize(self, text: str) -> str:
        """Deanonymize text"""
        return self._deanonymize(text)

    @abstractmethod
    def _deanonymize(self, text: str) -> str:
        """Abstract method to deanonymize text"""
