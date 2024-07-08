import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult
    from presidio_anonymizer.entities import EngineResult

MappingDataType = Dict[str, Dict[str, str]]


def format_duplicated_operator(operator_name: str, count: int) -> str:
    """Format the operator name with the count."""

    clean_operator_name = re.sub(r"[<>]", "", operator_name)
    clean_operator_name = re.sub(r"_\d+$", "", clean_operator_name)

    if operator_name.startswith("<") and operator_name.endswith(">"):
        return f"<{clean_operator_name}_{count}>"
    else:
        return f"{clean_operator_name}_{count}"


@dataclass
class DeanonymizerMapping:
    """Deanonymizer mapping."""

    mapping: MappingDataType = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(str))
    )

    @property
    def data(self) -> MappingDataType:
        """Return the deanonymizer mapping."""
        return {k: dict(v) for k, v in self.mapping.items()}

    def update(self, new_mapping: MappingDataType) -> None:
        """Update the deanonymizer mapping with new values.

        Duplicated values will not be added
        If there are multiple entities of the same type, the mapping will
        include a count to differentiate them. For example, if there are
        two names in the input text, the mapping will include NAME_1 and NAME_2.
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
                        format_duplicated_operator(key, count)
                        if key in self.mapping[entity_type]
                        else key
                    )

                    self.mapping[entity_type][new_key] = value
                    seen_values.add(value)
                    count += 1


def create_anonymizer_mapping(
    original_text: str,
    analyzer_results: List["RecognizerResult"],
    anonymizer_results: "EngineResult",
    is_reversed: bool = False,
) -> MappingDataType:
    """Create or update the mapping used to anonymize and/or
     deanonymize a text.

    This method exploits the results returned by the
    analysis and anonymization processes.

    If is_reversed is True, it constructs a mapping from each original
    entity to its anonymized value.

    If is_reversed is False, it constructs a mapping from each
    anonymized entity back to its original text value.

    If there are multiple entities of the same type, the mapping will
    include a count to differentiate them. For example, if there are
    two names in the input text, the mapping will include NAME_1 and NAME_2.

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
    analyzer_results.sort(key=lambda d: d.start)
    anonymizer_results.items.sort(key=lambda d: d.start)

    mapping: MappingDataType = defaultdict(dict)
    count: dict = defaultdict(int)

    for analyzed, anonymized in zip(analyzer_results, anonymizer_results.items):
        original_value = original_text[analyzed.start : analyzed.end]
        entity_type = anonymized.entity_type

        if is_reversed:
            cond = original_value in mapping[entity_type].values()
        else:
            cond = original_value in mapping[entity_type]

        if cond:
            continue

        if (
            anonymized.text in mapping[entity_type].values()
            or anonymized.text in mapping[entity_type]
        ):
            anonymized_value = format_duplicated_operator(
                anonymized.text, count[entity_type] + 2
            )
            count[entity_type] += 1
        else:
            anonymized_value = anonymized.text

        mapping_key, mapping_value = (
            (anonymized_value, original_value)
            if is_reversed
            else (original_value, anonymized_value)
        )

        mapping[entity_type][mapping_key] = mapping_value

    return mapping
