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


def generate_ngrams(words_list, n):
    """Generate n-grams from a list of words"""
    return [" ".join(words_list[i : i + n]) for i in range(len(words_list) - (n - 1))]


def fuzzy_matching_strategy(
    text: str, deanonymizer_mapping: MappingDataType, fuzzy_threshold: int = 80
) -> str:
    from fuzzywuzzy import fuzz

    for entity_type in deanonymizer_mapping:
        for anonymized, original in deanonymizer_mapping[entity_type].items():
            # Split the anonymized entity and text into words
            anonymized_words = anonymized.split()
            text_words = text.split()

            # Generate text segments of the same length as the anonymized entity
            segments = generate_ngrams(text_words, len(anonymized_words))

            # Iterate over each segment
            for i, segment in enumerate(segments):
                # Fuzzy match the segment with the anonymized entity
                if fuzz.ratio(anonymized.lower(), segment.lower()) > fuzzy_threshold:
                    # Replace the words in the original text
                    text_words[i : i + len(anonymized_words)] = original.split()

            # Join the words back into text
            text = " ".join(text_words)

    return text
