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
        seen_values = set()

        for entity_type, values in new_mapping.items():
            count = len(self.mapping[entity_type]) + 1

            for key, value in values.items():
                if (
                    value not in seen_values
                    and value not in self.mapping[entity_type].values()
                ):
                    new_key = (
                        f"<{entity_type}_{count}>"
                        if key.startswith("<") and key.endswith(">")
                        else key
                    )
                    self.mapping[entity_type][new_key] = value
                    seen_values.add(value)
                    count += 1


def create_anonymizer_mapping(
    original_text: str,
    analyzer_results: List[RecognizerResult],
    anonymizer_results: EngineResult,
    is_reversed: bool = False,
) -> MappingDataType:
    """Creates or updates the mapping used to anonymize and/or deanonymize text.

    This method exploits the results returned by the
    analysis and anonymization processes.

    If is_reversed is True, it constructs a mapping from each original
    entity to its anonymized value.

    If is_reversed is False, it constructs a mapping from each
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

    mapping: MappingDataType = defaultdict(dict)
    count: dict = defaultdict(int)

    for analyzed, anonymized in zip(analyzer_results, anonymizer_results.items):
        original_value = original_text[analyzed.start : analyzed.end]
        anonymized_value = (
            anonymized.text
            if not anonymized.text.startswith("<")
            else f"<{anonymized.entity_type}_{count[anonymized.entity_type] + 1}>"
        )

        entity_type = anonymized.entity_type
        mapping_key = anonymized_value if is_reversed else original_value
        mapping_value = original_value if is_reversed else anonymized_value

        if mapping_key not in mapping[entity_type]:
            mapping[entity_type][mapping_key] = mapping_value
            count[entity_type] += 1

    return mapping
