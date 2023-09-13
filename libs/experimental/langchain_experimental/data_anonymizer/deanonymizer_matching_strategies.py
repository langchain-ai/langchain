import re
from langchain_experimental.data_anonymizer.deanonymizer_mapping import MappingDataType


def exact_matching_strategy(text: str, deanonymizer_mapping: MappingDataType) -> str:
    """
    Exact matching strategy for deanonymization.
    It replaces all the anonymized entities with the original ones.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones"""

    # Iterate over all the entities (PERSON, EMAIL_ADDRESS, etc.)
    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            text = text.replace(anonymized, original)
    return text


def case_insensitive_matching_strategy(
    text: str, deanonymizer_mapping: MappingDataType
) -> str:
    """
    Case insensitive matching strategy for deanonymization.
    It replaces all the anonymized entities with the original ones irrespective of their letter case.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones
    """

    # Iterate over all the entities (PERSON, EMAIL_ADDRESS, etc.)
    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            # Use regular expressions for case-insensitive matching and replacing
            text = re.sub(anonymized, original, text, flags=re.IGNORECASE)
    return text


def ngram_fuzzy_matching_strategy(
    text: str, deanonymizer_mapping: MappingDataType, fuzzy_threshold: int = 80
) -> str:
    """
    N-gram fuzzy matching strategy for deanonymization.
    It replaces all the anonymized entities with the original ones.
    It uses fuzzy matching to find the position of the anonymized entity in the text.
    It generates n-grams of the same length as the anonymized entity from the text and
    uses fuzzy matching to find the position of the anonymized entity in the text.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones
        fuzzy_threshold: fuzzy matching threshold
    """

    def generate_ngrams(words_list, n):
        """Generate n-grams from a list of words"""
        return [
            " ".join(words_list[i : i + n]) for i in range(len(words_list) - (n - 1))
        ]

    try:
        from fuzzywuzzy import fuzz

    except ImportError as e:
        raise ImportError(
            "Could not import fuzzywuzzy, please install with "
            "`pip install fuzzywuzzy`."
        ) from e

    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            anonymized_words = anonymized.split()
            text_words = text.split()

            # Generate text segments of the same length as the anonymized entity
            segments = generate_ngrams(text_words, len(anonymized_words))

            for i, segment in enumerate(segments):
                # Fuzzy match the segment with the anonymized entity
                if fuzz.ratio(anonymized.lower(), segment.lower()) > fuzzy_threshold:
                    text_words[i : i + len(anonymized_words)] = original.split()

            text = " ".join(text_words)

    return text
