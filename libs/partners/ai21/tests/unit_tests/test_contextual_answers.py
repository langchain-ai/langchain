from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from langchain_ai21 import AI21ContextualAnswers
from langchain_ai21.contextual_answers import ContextualAnswerInput
from tests.unit_tests.conftest import DUMMY_API_KEY


@pytest.mark.parametrize(
    ids=[
        "when_no_context__should_raise_exception",
        "when_no_question__should_raise_exception",
        "when_question_is_an_empty_string__should_raise_exception",
        "when_context_is_an_empty_string__should_raise_exception",
        "when_context_is_an_empty_list",
    ],
    argnames="input",
    argvalues=[
        ({"question": "What is the capital of France?"}),
        ({"context": "Paris is the capital of France"}),
        ({"question": "", "context": "Paris is the capital of France"}),
        ({"context": "", "question": "some question?"}),
        ({"context": [], "question": "What is the capital of France?"}),
    ],
)
def test_invoke__on_bad_input(
    input: ContextualAnswerInput,
    mock_client_with_contextual_answers: Mock,
) -> None:
    tsm = AI21ContextualAnswers(
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        client=mock_client_with_contextual_answers,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError) as error:
        tsm.invoke(input)

    assert (
        error.value.args[0]
        == f"Input must contain a 'context' and 'question' fields. Got {input}"
    )


@pytest.mark.parametrize(
    ids=[
        "when_context_is_not_str_or_list_of_docs_or_str",
    ],
    argnames="input",
    argvalues=[
        ({"context": 1242, "question": "What is the capital of France?"}),
    ],
)
def test_invoke__on_context_bad_input(
    input: ContextualAnswerInput, mock_client_with_contextual_answers: Mock
) -> None:
    tsm = AI21ContextualAnswers(
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        client=mock_client_with_contextual_answers,
    )

    with pytest.raises(ValueError) as error:
        tsm.invoke(input)

    assert (
        error.value.args[0] == f"Expected input to be a list of strings or Documents."
        f" Received {type(input)}"
    )


@pytest.mark.parametrize(
    ids=[
        "when_context_is_a_list_of_strings",
        "when_context_is_a_list_of_documents",
        "when_context_is_a_string",
    ],
    argnames="input",
    argvalues=[
        (
            {
                "context": ["Paris is the capital of france"],
                "question": "What is the capital of France?",
            }
        ),
        (
            {
                "context": [Document(page_content="Paris is the capital of france")],
                "question": "What is the capital of France?",
            }
        ),
        (
            {
                "context": "Paris is the capital of france",
                "question": "What is the capital of France?",
            }
        ),
    ],
)
def test_invoke__on_good_input(
    input: ContextualAnswerInput, mock_client_with_contextual_answers: Mock
) -> None:
    tsm = AI21ContextualAnswers(
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        client=mock_client_with_contextual_answers,
    )

    response = tsm.invoke(input)
    assert isinstance(response, str)
