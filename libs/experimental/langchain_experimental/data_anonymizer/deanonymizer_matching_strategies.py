from typing import Dict


def default_matching_strategy(
    text: str, deanonymizer_mapping: Dict[str, Dict[str, str]]
) -> str:
    """
    Default matching strategy for deanonymization.
    It replaces all the anonymized entities with the original ones.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones"""

    # Iterate over all the entities (PERSON, EMAIL_ADDRESS, etc.)
    for entity_type in deanonymizer_mapping:
        # Iterate over all the anonymized entities for the current entity type
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            # Replace all the occurrences of the anonymized entity with the original one
            text = text.replace(anonymized, original)
    return text
