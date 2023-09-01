from langchain_experimental.data_anonymizer.presidio import MappingDataType


def default_matching_strategy(text: str, deanonymizer_mapping: MappingDataType) -> str:
    """
    Default matching strategy for deanonymization.
    It replaces all the anonymized entities with the original ones.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones"""

    # Iterate over all the entities (PERSON, EMAIL_ADDRESS, etc.)
    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            text = text.replace(anonymized, original)
    return text
