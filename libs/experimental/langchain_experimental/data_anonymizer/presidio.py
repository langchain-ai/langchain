from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_experimental.data_anonymizer.base import AnonymizerBase
from langchain_experimental.data_anonymizer.faker_presidio_mapping import (
    get_pseudoanonymizer_mapping,
)

if TYPE_CHECKING:
    from presidio_analyzer import EntityRecognizer
    from presidio_anonymizer.entities import OperatorConfig


class PresidioAnonymizer(AnonymizerBase):
    """Anonymizer using Microsoft Presidio."""

    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        operators: Optional[Dict[str, OperatorConfig]] = None,
    ):
        """
        Args:
            analyzed_fields: List of fields to detect and then anonymize.
                Defaults to all entities supported by Microsoft Presidio.
            operators: Operators to use for anonymization.
                Operators allow for custom anonymization of detected PII.
                Learn more:
                https://microsoft.github.io/presidio/tutorial/10_simple_anonymization/
        """
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as e:
            raise ImportError(
                "Could not import presidio_analyzer, please install with "
                "`pip install presidio-analyzer`. You will also need to download a "
                "spaCy model to use the analyzer, e.g. "
                "`python -m spacy download en_core_web_lg`."
            ) from e
        try:
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError as e:
            raise ImportError(
                "Could not import presidio_anonymizer, please install with "
                "`pip install presidio-anonymizer`."
            ) from e

        self.analyzed_fields = (
            analyzed_fields
            if analyzed_fields is not None
            else list(get_pseudoanonymizer_mapping().keys())
        )
        self.operators = (
            operators
            if operators is not None
            else {
                field: OperatorConfig(
                    operator_name="custom", params={"lambda": faker_function}
                )
                for field, faker_function in get_pseudoanonymizer_mapping().items()
            }
        )
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def _anonymize(self, text: str) -> str:
        results = self._analyzer.analyze(
            text,
            entities=self.analyzed_fields,
            language="en",
        )

        return self._anonymizer.anonymize(
            text,
            analyzer_results=results,
            operators=self.operators,
        ).text

    def add_recognizer(self, recognizer: EntityRecognizer) -> None:
        """Add a recognizer to the analyzer"""
        self._analyzer.registry.add_recognizer(recognizer)
        self.analyzed_fields.extend(recognizer.supported_entities)

    def add_operators(self, operators: Dict[str, OperatorConfig]) -> None:
        """Add operators to the anonymizer"""
        self.operators.update(operators)
