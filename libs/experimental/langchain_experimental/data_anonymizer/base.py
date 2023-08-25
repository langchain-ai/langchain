from abc import ABC, abstractmethod


class AnonymizerBase(ABC):
    """Base abstract class for anonymizers"""

    def anonymize(self, text: str) -> str:
        """Anonymize text"""
        return self._anonymize(text)

    @abstractmethod
    def _anonymize(self, text: str) -> str:
        """Abstract method to anonymize text"""
