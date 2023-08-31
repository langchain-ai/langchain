from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from langchain_experimental.data_anonymizer.base import (
    AnonymizerBase,
    ReversibleAnonymizerBase,
)
from langchain_experimental.data_anonymizer.faker_presidio_mapping import (
    get_pseudoanonymizer_mapping,
)
import yaml

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

if TYPE_CHECKING:
    from presidio_analyzer import EntityRecognizer, RecognizerResult
    from presidio_anonymizer.entities import EngineResult


class PresidioAnonymizerBase(AnonymizerBase):
    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        language: str = "en",
        operators: Optional[Dict[str, OperatorConfig]] = None,
        faker_seed: Optional[int] = None,
    ):
        """
        Args:
            analyzed_fields: List of fields to detect and then anonymize.
                Defaults to all entities supported by Microsoft Presidio.
            language: Language to use for analysis. Defaults to english.
            operators: Operators to use for anonymization.
                Operators allow for custom anonymization of detected PII.
                Learn more:
                https://microsoft.github.io/presidio/tutorial/10_simple_anonymization/
            faker_seed: Seed used to initialize faker.
                Defaults to None, in which case faker will be seeded randomly
                and provide random values.
        """
        self.analyzed_fields = (
            analyzed_fields
            if analyzed_fields is not None
            else list(get_pseudoanonymizer_mapping().keys())
        )
        self.language = language
        self.operators = (
            operators
            if operators is not None
            else {
                field: OperatorConfig(
                    operator_name="custom", params={"lambda": faker_function}
                )
                for field, faker_function in get_pseudoanonymizer_mapping(
                    faker_seed
                ).items()
            }
        )
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def add_recognizer(self, recognizer: EntityRecognizer) -> None:
        """Add a recognizer to the analyzer"""
        self._analyzer.registry.add_recognizer(recognizer)
        self.analyzed_fields.extend(recognizer.supported_entities)

    def add_operators(self, operators: Dict[str, OperatorConfig]) -> None:
        """Add operators to the anonymizer"""
        self.operators.update(operators)


class PresidioAnonymizer(PresidioAnonymizerBase):
    def _anonymize(self, text: str) -> str:
        """Anonymize text.
        Each PII entity is replaced with a fake value.
        Each time fake values will be different, as they are generated randomly.
        """
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


class PresidioReversibleAnonymizer(PresidioAnonymizerBase, ReversibleAnonymizerBase):
    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        language: str = "en",
        operators: Optional[Dict[str, OperatorConfig]] = None,
        faker_seed: Optional[int] = None,
    ):
        super().__init__(analyzed_fields, language, operators, faker_seed)
        self._deanonymizer_mapping: Dict[str, Dict[str, str]] = defaultdict(
            lambda: defaultdict(str)
        )

    @property
    def deanonymizer_mapping(self) -> Dict[str, Dict[str, str]]:
        """Return the deanonymizer mapping"""
        return {k: dict(v) for k, v in self._deanonymizer_mapping.items()}

    def _create_update_deanonymizer_mapping(
        self,
        original_text: str,
        analyzer_results: List[RecognizerResult],
        anonymizer_results: EngineResult,
    ) -> None:
        """Creates or updates the mapping used to de-anonymize text.

        This method exploits the results returned by the
        analysis and anonymization processes.

        It constructs a mapping from each anonymized entity
        back to its original text value.

        Mapping will be stored as "deanonymizer_mapping" property.

        Example of "deanonymizer_mapping":
        {
            "PERSON": {
                "<anonymized>": "<original>",
                "John Doe": "Slim Shady"
            },
            "PHONE_NUMBER": {
                "111-111-1111": "555-555-5555"
            }
            ...
        }
        """

        # We are able to zip and loop through both lists because we expect
        # them to return corresponding entities for each identified piece
        # of analyzable data from our input.

        # We sort them by their 'start' attribute because it allows us to
        # match corresponding entities by their position in the input text.
        analyzer_results = sorted(analyzer_results, key=lambda d: d.start)
        anonymizer_results.items = sorted(
            anonymizer_results.items, key=lambda d: d.start
        )

        new_deanonymizer_mapping: defaultdict[str, Dict[str, str]] = defaultdict(dict)

        for analyzed_entity, anonymized_entity in zip(
            analyzer_results, anonymizer_results.items
        ):
            original_value = original_text[analyzed_entity.start : analyzed_entity.end]
            new_deanonymizer_mapping[anonymized_entity.entity_type][
                anonymized_entity.text
            ] = original_value

        for entity_type, values in new_deanonymizer_mapping.items():
            self._deanonymizer_mapping[entity_type].update(values)

    def _anonymize(self, text: str) -> str:
        """Anonymize text.
        Each PII entity is replaced with a fake value.
        Each time fake values will be different, as they are generated randomly.
        At the same time, we will create a mapping from each anonymized entity
        back to its original text value.
        """
        analyzer_results = self._analyzer.analyze(
            text,
            entities=self.analyzed_fields,
            language=self.language,
        )

        filtered_analyzer_results = (
            self._anonymizer._remove_conflicts_and_get_text_manipulation_data(
                analyzer_results
            )
        )

        anonymizer_results = self._anonymizer.anonymize(
            text,
            analyzer_results=analyzer_results,
            operators=self.operators,
        )

        self._create_update_deanonymizer_mapping(
            text, filtered_analyzer_results, anonymizer_results
        )

        return anonymizer_results.text

    def _deanonymize(self, text_to_deanonymize: str) -> str:
        """Deanonymize text.
        Each anonymized entity is replaced with its original value.
        This method exploits the mapping created during the anonymization process.
        """
        if not self._deanonymizer_mapping:
            raise ValueError(
                "Deanonymizer mapping is empty.",
                "Please call anonymize() and anonymize some text first.",
            )

        for entity_type in self._deanonymizer_mapping:
            for anonymized, original in self._deanonymizer_mapping[entity_type].items():
                text_to_deanonymize = text_to_deanonymize.replace(anonymized, original)
        return text_to_deanonymize

    def save_deanonymizer_mapping(self, file_path: Union[Path, str]) -> None:
        """Save the deanonymizer mapping to a JSON or YAML file.

        Args:
            file_path: Path to file to save the mapping to.

        Example:
        .. code-block:: python

            anonymizer.save_deanonymizer_mapping(file_path="path/mapping.json")
        """

        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check file extension
        if save_path.suffix not in [".json", ".yaml"]:
            raise ValueError(f"{save_path} must have an extension of .json or .yaml")

        # Make sure parent directories exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        if save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(self.deanonymizer_mapping, f, indent=2)
        elif save_path.suffix == ".yaml":
            with open(save_path, "w") as f:
                yaml.dump(self.deanonymizer_mapping, f, default_flow_style=False)

    def load_deanonymizer_mapping(self, file_path: Union[Path, str]) -> None:
        """Load the deanonymizer mapping from a JSON or YAML file.

        Args:
            file_path: Path to file to load the mapping from.

        Example:
        .. code-block:: python

            anonymizer.load_deanonymizer_mapping(file_path="path/mapping.json")
        """

        # Convert file to Path object.
        load_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check file extension
        if load_path.suffix not in [".json", ".yaml"]:
            raise ValueError(f"{load_path} must have an extension of .json or .yaml")

        # Load the file
        if load_path.suffix == ".json":
            with open(load_path, "r") as f:
                self._deanonymizer_mapping = json.load(f)
        elif load_path.suffix == ".yaml":
            with open(load_path, "r") as f:
                self._deanonymizer_mapping = yaml.load(f, Loader=yaml.FullLoader)
