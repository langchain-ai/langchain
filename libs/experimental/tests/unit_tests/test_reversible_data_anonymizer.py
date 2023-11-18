import os
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
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = "Hello, my name is John Doe."
    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=analyzed_fields)
    anonymized_text = anonymizer.anonymize(text)
    assert ("John Doe" in anonymized_text) == should_contain


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
@pytest.mark.parametrize(
    "analyzed_fields,should_contain",
    [(["PERSON"], True), (["PHONE_NUMBER"], True), (None, True)],
)
def test_anonymize_allow_list(analyzed_fields: List[str], should_contain: bool) -> None:
    """Test anonymizing a name in a simple sentence"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = "Hello, my name is John Doe."
    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=analyzed_fields)
    anonymized_text = anonymizer.anonymize(text, allow_list=["John Doe"])
    assert ("John Doe" in anonymized_text) == should_contain


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_anonymize_multiple() -> None:
    """Test anonymizing multiple items in a sentence"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = "John Smith's phone number is 313-666-7440 and email is johnsmith@gmail.com"
    anonymizer = PresidioReversibleAnonymizer()
    anonymized_text = anonymizer.anonymize(text)
    for phrase in ["John Smith", "313-666-7440", "johnsmith@gmail.com"]:
        assert phrase not in anonymized_text


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_check_instances() -> None:
    """Test anonymizing multiple items in a sentence"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = (
        "This is John Smith. John Smith works in a bakery." "John Smith is a good guy"
    )
    anonymizer = PresidioReversibleAnonymizer(["PERSON"], faker_seed=42)
    anonymized_text = anonymizer.anonymize(text)
    persons = list(anonymizer.deanonymizer_mapping["PERSON"].keys())
    assert len(persons) == 1

    anonymized_name = persons[0]
    assert anonymized_text.count(anonymized_name) == 3

    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text.count(anonymized_name) == 3
    assert anonymizer.deanonymizer_mapping["PERSON"][anonymized_name] == "John Smith"

    text = "This is Jane Smith"
    anonymized_text = anonymizer.anonymize(text)
    persons = list(anonymizer.deanonymizer_mapping["PERSON"].keys())
    assert len(persons) == 2


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_anonymize_with_custom_operator() -> None:
    """Test anonymize a name with a custom operator"""
    from presidio_anonymizer.entities import OperatorConfig

    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    custom_operator = {"PERSON": OperatorConfig("replace", {"new_value": "NAME"})}
    anonymizer = PresidioReversibleAnonymizer(operators=custom_operator)

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

    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=[])
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
    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=[])
    anonymizer.add_recognizer(custom_recognizer)
    custom_operator = {"TITLE": OperatorConfig("replace", {"new_value": "Dear"})}
    anonymizer.add_operators(custom_operator)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == "Dear Jane Doe was here."


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_deanonymizer_mapping() -> None:
    """Test if deanonymizer mapping is correctly populated"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    anonymizer = PresidioReversibleAnonymizer(
        analyzed_fields=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"]
    )

    anonymizer.anonymize("Hello, my name is John Doe and my number is 444 555 6666.")

    # ["PERSON", "PHONE_NUMBER"]
    assert len(anonymizer.deanonymizer_mapping.keys()) == 2
    assert "John Doe" in anonymizer.deanonymizer_mapping.get("PERSON", {}).values()
    assert (
        "444 555 6666"
        in anonymizer.deanonymizer_mapping.get("PHONE_NUMBER", {}).values()
    )

    text_to_anonymize = (
        "And my name is Jane Doe, my email is jane@gmail.com and "
        "my credit card is 4929 5319 6292 5362."
    )
    anonymizer.anonymize(text_to_anonymize)

    # ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"]
    assert len(anonymizer.deanonymizer_mapping.keys()) == 4
    assert "Jane Doe" in anonymizer.deanonymizer_mapping.get("PERSON", {}).values()
    assert (
        "jane@gmail.com"
        in anonymizer.deanonymizer_mapping.get("EMAIL_ADDRESS", {}).values()
    )
    assert (
        "4929 5319 6292 5362"
        in anonymizer.deanonymizer_mapping.get("CREDIT_CARD", {}).values()
    )


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_deanonymize() -> None:
    """Test deanonymizing a name in a simple sentence"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = "Hello, my name is John Doe."
    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=["PERSON"])
    anonymized_text = anonymizer.anonymize(text)
    deanonymized_text = anonymizer.deanonymize(anonymized_text)
    assert deanonymized_text == text


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_save_load_deanonymizer_mapping() -> None:
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    anonymizer = PresidioReversibleAnonymizer(analyzed_fields=["PERSON"])
    anonymizer.anonymize("Hello, my name is John Doe.")
    try:
        anonymizer.save_deanonymizer_mapping("test_file.json")
        assert os.path.isfile("test_file.json")

        anonymizer = PresidioReversibleAnonymizer()
        anonymizer.load_deanonymizer_mapping("test_file.json")

        assert "John Doe" in anonymizer.deanonymizer_mapping.get("PERSON", {}).values()

    finally:
        os.remove("test_file.json")


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_non_faker_values() -> None:
    """Test anonymizing multiple items in a sentence without faker values"""
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

    text = (
        "My name is John Smith. Your name is Adam Smith. Her name is Jane Smith."
        "Our names are: John Smith, Adam Smith, Jane Smith."
    )
    expected_result = (
        "My name is <PERSON>. Your name is <PERSON_2>. Her name is <PERSON_3>."
        "Our names are: <PERSON>, <PERSON_2>, <PERSON_3>."
    )
    anonymizer = PresidioReversibleAnonymizer(add_default_faker_operators=False)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == expected_result
