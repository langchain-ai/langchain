from typing import Iterator, List

import pytest

from . import is_libcublas_available


@pytest.fixture(scope="module", autouse=True)
def check_spacy_model() -> Iterator[None]:
    import spacy

    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip(reason="Spacy model 'en_core_web_lg' not installed")
    yield


@pytest.fixture(scope="module", autouse=True)
def check_libcublas() -> Iterator[None]:
    if not is_libcublas_available():
        pytest.skip(reason="libcublas.so is not available")
    yield


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
@pytest.mark.parametrize(
    "analyzed_fields,should_contain",
    [(["PERSON"], False), (["PHONE_NUMBER"], True), (None, False)],
)
def test_anonymize(analyzed_fields: List[str], should_contain: bool) -> None:
    """Test anonymizing a name in a simple sentence"""
    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    text = "Hello, my name is John Doe."
    anonymizer = PresidioAnonymizer(analyzed_fields=analyzed_fields)
    anonymized_text = anonymizer.anonymize(text)
    assert ("John Doe" in anonymized_text) == should_contain


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
@pytest.mark.parametrize(
    "analyzed_fields,should_contain",
    [(["PERSON"], True), (["PHONE_NUMBER"], True), (None, True)],
)
def test_anonymize_allow_list(analyzed_fields: List[str], should_contain: bool) -> None:
    """Test anonymizing a name in a simple sentence"""
    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    text = "Hello, my name is John Doe."
    anonymizer = PresidioAnonymizer(analyzed_fields=analyzed_fields)
    anonymized_text = anonymizer.anonymize(text, allow_list=["John Doe"])
    assert ("John Doe" in anonymized_text) == should_contain


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_anonymize_multiple() -> None:
    """Test anonymizing multiple items in a sentence"""
    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    text = "John Smith's phone number is 313-666-7440 and email is johnsmith@gmail.com"
    anonymizer = PresidioAnonymizer()
    anonymized_text = anonymizer.anonymize(text)
    for phrase in ["John Smith", "313-666-7440", "johnsmith@gmail.com"]:
        assert phrase not in anonymized_text


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_check_instances() -> None:
    """Test anonymizing multiple items in a sentence"""
    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    text = (
        "This is John Smith. John Smith works in a bakery." "John Smith is a good guy"
    )
    anonymizer = PresidioAnonymizer(["PERSON"], faker_seed=42)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text.count("Connie Lawrence") == 3

    # New name should be generated
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text.count("Connie Lawrence") == 0


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_anonymize_with_custom_operator() -> None:
    """Test anonymize a name with a custom operator"""
    from presidio_anonymizer.entities import OperatorConfig

    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    custom_operator = {"PERSON": OperatorConfig("replace", {"new_value": "NAME"})}
    anonymizer = PresidioAnonymizer(operators=custom_operator)

    text = "Jane Doe was here."

    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == "NAME was here."


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_add_recognizer_operator() -> None:
    """
    Test add recognizer and anonymize a new type of entity and with a custom operator
    """
    from presidio_analyzer import PatternRecognizer
    from presidio_anonymizer.entities import OperatorConfig

    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    anonymizer = PresidioAnonymizer(analyzed_fields=[])
    titles_list = ["Sir", "Madam", "Professor"]
    custom_recognizer = PatternRecognizer(
        supported_entity="TITLE", deny_list=titles_list
    )
    anonymizer.add_recognizer(custom_recognizer)

    # anonymizing with custom recognizer
    text = "Madam Jane Doe was here."
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == "<TITLE> Jane Doe was here."

    # anonymizing with custom recognizer and operator
    custom_operator = {"TITLE": OperatorConfig("replace", {"new_value": "Dear"})}
    anonymizer.add_operators(custom_operator)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == "Dear Jane Doe was here."


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_non_faker_values() -> None:
    """Test anonymizing multiple items in a sentence without faker values"""
    from langchain_experimental.data_anonymizer import PresidioAnonymizer

    text = (
        "My name is John Smith. Your name is Adam Smith. Her name is Jane Smith."
        "Our names are: John Smith, Adam Smith, Jane Smith."
    )
    expected_result = (
        "My name is <PERSON>. Your name is <PERSON_2>. Her name is <PERSON_3>."
        "Our names are: <PERSON>, <PERSON_2>, <PERSON_3>."
    )
    anonymizer = PresidioAnonymizer(add_default_faker_operators=False)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == expected_result


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_exact_matching_strategy() -> None:
    """
    Test exact matching strategy for deanonymization.
    """
    from langchain_experimental.data_anonymizer import (
        deanonymizer_matching_strategies as dms,
    )

    deanonymizer_mapping = {
        "PERSON": {"Maria Lynch": "Slim Shady"},
        "PHONE_NUMBER": {"7344131647": "313-666-7440"},
        "EMAIL_ADDRESS": {"wdavis@example.net": "real.slim.shady@gmail.com"},
        "CREDIT_CARD": {"213186379402654": "4916 0387 9536 0861"},
    }

    text = (
        "Are you Maria Lynch? I found your card with number 213186379402654. "
        "Is this your phone number: 7344131647? "
        "Is this your email address: wdavis@example.net"
    )

    deanonymized_text = dms.exact_matching_strategy(text, deanonymizer_mapping)

    for original_value in [
        "Slim Shady",
        "313-666-7440",
        "real.slim.shady@gmail.com",
        "4916 0387 9536 0861",
    ]:
        assert original_value in deanonymized_text


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_best_matching_strategy() -> None:
    """
    Test exact matching strategy for deanonymization.
    """
    from langchain_experimental.data_anonymizer import (
        deanonymizer_matching_strategies as dms,
    )

    deanonymizer_mapping = {
        "PERSON": {"Maria Lynch": "Slim Shady"},
        "PHONE_NUMBER": {"7344131647": "313-666-7440"},
        "EMAIL_ADDRESS": {"wdavis@example.net": "real.slim.shady@gmail.com"},
        "CREDIT_CARD": {"213186379402654": "4916 0387 9536 0861"},
    }

    # Changed some values:
    # - "Maria Lynch" -> "Maria K. Lynch"
    # - "7344131647" -> "734-413-1647"
    # - "213186379402654" -> "2131 8637 9402 654"
    # - "wdavis@example.net" -> the same to test exact match
    text = (
        "Are you Maria K. Lynch? I found your card with number 2131 8637 9402 654. "
        "Is this your phone number: 734-413-1647?"
        "Is this your email address: wdavis@example.net"
    )

    deanonymized_text = dms.combined_exact_fuzzy_matching_strategy(
        text, deanonymizer_mapping
    )

    for original_value in [
        "Slim Shady",
        "313-666-7440",
        "real.slim.shady@gmail.com",
        "4916 0387 9536 0861",
    ]:
        assert original_value in deanonymized_text
