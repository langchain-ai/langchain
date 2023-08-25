from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from langchain_experimental.data_anonymizer.base import AnonymizerBase
from langchain_experimental.data_anonymizer.utils import pseudoanonymizer_mapping
from typing import Any, Dict, List


class PresidioAnonymizer(AnonymizerBase):
    """Anonymizer using Presidio"""

    def __init__(
        self,
        analyzed_fields: List[str] = None,
        language: str = "en",
        operators: Dict[str, Any] = None,
    ):
        self.analyzed_fields = analyzed_fields
        self.language = language

        if operators is None:
            self.operators = {
                field: OperatorConfig(
                    operator_name="custom", params={"lambda": faker_function}
                )
                for field, faker_function in pseudoanonymizer_mapping.items()
            }

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def _anonymize(self, text: str) -> str:
        results = self._analyzer.analyze(
            text,
            entities=self.analyzed_fields,
            language=self.language,
        )

        return self._anonymizer.anonymize(
            text,
            analyzer_results=results,
            operators=self.operators,
        ).text

    def add_recognizer(self, recognizer: Any) -> None:
        """Add a recognizer to the analyzer"""
        self._analyzer.registry.add_recognizer(recognizer)
