import re
from typing import List

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
    It replaces all the anonymized entities with the original ones
        irrespective of their letter case.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones

    Examples of matching:
        keanu reeves -> Keanu Reeves
        JOHN F. KENNEDY -> John F. Kennedy
    """

    # Iterate over all the entities (PERSON, EMAIL_ADDRESS, etc.)
    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            # Use regular expressions for case-insensitive matching and replacing
            text = re.sub(anonymized, original, text, flags=re.IGNORECASE)
    return text


def fuzzy_matching_strategy(
    text: str, deanonymizer_mapping: MappingDataType, max_l_dist: int = 3
) -> str:
    """
    Fuzzy matching strategy for deanonymization.
    It uses fuzzy matching to find the position of the anonymized entity in the text.
    It replaces all the anonymized entities with the original ones.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones
        max_l_dist: maximum Levenshtein distance between the anonymized entity and the
            text segment to consider it a match

    Examples of matching:
        Kaenu Reves -> Keanu Reeves
        John F. Kennedy -> John Kennedy
    """

    try:
        from fuzzysearch import find_near_matches
    except ImportError as e:
        raise ImportError(
            "Could not import fuzzysearch, please install with "
            "`pip install fuzzysearch`."
        ) from e

    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            matches = find_near_matches(anonymized, text, max_l_dist=max_l_dist)
            new_text = ""
            last_end = 0
            for m in matches:
                # add the text that isn't part of a match
                new_text += text[last_end : m.start]
                # add the replacement text
                new_text += original
                last_end = m.end
            # add the remaining text that wasn't part of a match
            new_text += text[last_end:]
            text = new_text

    return text


def combined_exact_fuzzy_matching_strategy(
    text: str, deanonymizer_mapping: MappingDataType, max_l_dist: int = 3
) -> str:
    """
    RECOMMENDED STRATEGY.
    Combined exact and fuzzy matching strategy for deanonymization.

    Args:
        text: text to deanonymize
        deanonymizer_mapping: mapping between anonymized entities and original ones
        max_l_dist: maximum Levenshtein distance between the anonymized entity and the
            text segment to consider it a match

    Examples of matching:
        Kaenu Reves -> Keanu Reeves
        John F. Kennedy -> John Kennedy
    """
    text = exact_matching_strategy(text, deanonymizer_mapping)
    text = fuzzy_matching_strategy(text, deanonymizer_mapping, max_l_dist)
    return text


def ngram_fuzzy_matching_strategy(
    text: str,
    deanonymizer_mapping: MappingDataType,
    fuzzy_threshold: int = 85,
    use_variable_length: bool = True,
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
        use_variable_length: whether to use (n-1, n, n+1)-grams or just n-grams
    """

    def generate_ngrams(words_list: List[str], n: int) -> list:
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

    text_words = text.split()
    replacements = []
    matched_indices: List[int] = []

    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            anonymized_words = anonymized.split()

            if use_variable_length:
                gram_lengths = [
                    len(anonymized_words) - 1,
                    len(anonymized_words),
                    len(anonymized_words) + 1,
                ]
            else:
                gram_lengths = [len(anonymized_words)]
            for n in gram_lengths:
                if n > 0:  # Take only positive values
                    segments = generate_ngrams(text_words, n)
                    for i, segment in enumerate(segments):
                        if (
                            fuzz.ratio(anonymized.lower(), segment.lower())
                            > fuzzy_threshold
                            and i not in matched_indices
                        ):
                            replacements.append((i, n, original))
                            # Add the matched segment indices to the list
                            matched_indices.extend(range(i, i + n))

    # Sort replacements by index in reverse order
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements in reverse order to not affect subsequent indices
    for start, length, replacement in replacements:
        text_words[start : start + length] = replacement.split()

    return " ".join(text_words)
