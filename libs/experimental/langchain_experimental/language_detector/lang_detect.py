from typing import TYPE_CHECKING, List, Tuple

from langchain_experimental.language_detector.base import LanguageDetectorBase

try:
    import langdetect
except ImportError:
    raise ImportError(
        "Could not import langdetect, please install with " "`pip install langdetect`."
    )


if TYPE_CHECKING:
    import langdetect


class LangDetector(LanguageDetectorBase):
    """
    Language detector based on langdetect package.
    Warning: Results are not deterministic!
    It supports 55 languages out of the box.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = (
            threshold  # Minimum probability to consider a language as detected.
        )

    def _detect_single(self, text: str) -> str:
        """Detects the most probable language of a single text."""
        return langdetect.detect(text)

    def _detect_many(self, text: str) -> List[Tuple[str, float]]:
        """Detects all languages of a single text with a score bigger than threshold.
        Returns them sorted on score in descending order.
        """
        predictions = langdetect.detect_langs(text)
        return [(x.lang, x.prob) for x in predictions if x.prob > self.threshold]
