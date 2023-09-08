from abc import ABC, abstractmethod
from typing import List, Tuple


class LanguageDetectorBase(ABC):
    """
    Base abstract class for language detectors.
    """

    def detect_single_language(self, text: str) -> str:
        """Detect language of a single text"""
        return self._detect_single(text)

    @abstractmethod
    def _detect_single(self, text: str) -> str:
        """Abstract method to detect language of a single text"""

    def detect_many_languages(self, text: str) -> List[Tuple[str, float]]:
        """Detect languages of a single text"""
        return self._detect_many(text)

    @abstractmethod
    def _detect_many(self, text: str) -> List[Tuple[str, float]]:
        """Abstract method to detect languages of a single text"""
