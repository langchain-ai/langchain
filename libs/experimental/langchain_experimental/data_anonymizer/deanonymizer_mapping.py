from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from presidio_analyzer import RecognizerResult
from presidio_anonymizer.entities import EngineResult

MappingDataType = Dict[str, Dict[str, str]]


@dataclass
class DeanonymizerMapping:
    mapping: MappingDataType = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(str))
    )

    @property
    def data(self) -> MappingDataType:
        """Return the deanonymizer mapping"""
        return {k: dict(v) for k, v in self.mapping.items()}

    def update(self, new_mapping: MappingDataType) -> None:
        """Update the deanonymizer mapping with new values
        Duplicated values will not be added
        """
        new_values_seen = set()

        for entity_type, values in new_mapping.items():
            for k, v in values.items():
                # Make sure it is not a duplicate value
                if (
                    v not in self.mapping[entity_type].values()
                    and v not in new_values_seen
                ):
                    self.mapping[entity_type][k] = v
                    new_values_seen.update({v})


def create_anonymizer_mapping(
    original_text: str,
    analyzer_results: List[RecognizerResult],
    anonymizer_results: EngineResult,
    reversed: bool = False,
) -> MappingDataType:
    """Creates or updates the mapping used to anonymize and/or deanonymize text.

    This method exploits the results returned by the
    analysis and anonymization processes.

    If reversed is True, it constructs a mapping from each original
    entity to its anonymized value.

    If reversed is False, it constructs a mapping from each
    anonymized entity back to its original text value.

    Example of mapping:
    {
        "PERSON": {
            "<original>": "<anonymized>",
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
    anonymizer_results.items = sorted(anonymizer_results.items, key=lambda d: d.start)

    new_anonymizer_mapping: MappingDataType = defaultdict(dict)

    for analyzed_entity, anonymized_entity in zip(
        analyzer_results, anonymizer_results.items
    ):
        original_value = original_text[analyzed_entity.start : analyzed_entity.end]

        if reversed:
            new_anonymizer_mapping[anonymized_entity.entity_type][
                anonymized_entity.text
            ] = original_value
        else:
            new_anonymizer_mapping[anonymized_entity.entity_type][
                original_value
            ] = anonymized_entity.text

    return new_anonymizer_mapping
