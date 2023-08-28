from typing import Dict, List

import pytest
from presidio_analyzer import PatternRecognizer
from presidio_anonymizer.entities import OperatorConfig

from langchain_experimental.data_anonymizer import PresidioAnonymizer


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
@pytest.mark.parametrize(
    "analyzed_fields,should_contain",
    [(["PERSON"], False), (["PHONE_NUMBER"], True), (None, False)],
)
def test_anonymize(analyzed_fields: List[str], should_contain: bool) -> None:
    """Test anonymizing a name in a simple sentence"""
    text = "Hello, my name is John Doe."
    anonymizer = PresidioAnonymizer(analyzed_fields=analyzed_fields)
    anonymized_text = anonymizer.anonymize(text)
    assert ("John Doe" in anonymized_text) == should_contain


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_anonymize_multiple() -> None:
    """Test anonymizing multiple items in a sentence"""
    text = "John Smith's phone number is 313-666-7440 and email is johnsmith@gmail.com"
    anonymizer = PresidioAnonymizer()
    anonymized_text = anonymizer.anonymize(text)
    for phrase in ["John Smith", "313-666-7440", "johnsmith@gmail.com"]:
        assert phrase not in anonymized_text


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
@pytest.mark.parametrize(
    "custom_operator,expected",
    [
        (
            {"PERSON": OperatorConfig("replace", {"new_value": "<name>"})},
            "<name> was here.",
        )
    ],
)
def test_anonymize_with_custom_operator(
    custom_operator: Dict[str, OperatorConfig], expected: str
) -> None:
    """Test anonymize a name with a custom operator"""
    text = "Jane Doe was here."
    anonymizer = PresidioAnonymizer(operators=custom_operator)
    anonymized_text = anonymizer.anonymize(text)
    assert anonymized_text == expected


@pytest.mark.requires("presidio_analyzer", "presidio_anonymizer", "faker")
def test_add_recognizer_operator() -> None:
    """
    Test add recognizer and anonymize a new type of entity and with a custom operator
    """
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
