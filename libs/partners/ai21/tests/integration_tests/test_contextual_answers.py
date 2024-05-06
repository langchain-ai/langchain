from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from langchain_ai21.contextual_answers import (
    ANSWER_NOT_IN_CONTEXT_RESPONSE,
    AI21ContextualAnswers,
)

context = """
Albert Einstein German: 14 March 1879 – 18 April 1955) 
was a German-born theoretical physicist who is widely held
 to be one of the greatest and most influential scientists 
"""


_GOOD_QUESTION = "When did Albert Einstein born?"
_BAD_QUESTION = "What color is Yoda's light saber?"
_EXPECTED_PARTIAL_RESPONSE = "March 14, 1879"


def test_invoke__when_good_question() -> None:
    llm = AI21ContextualAnswers()

    response = llm.invoke(
        {"context": context, "question": _GOOD_QUESTION},
        config={"metadata": {"name": "I AM A TEST"}},
    )

    assert response != ANSWER_NOT_IN_CONTEXT_RESPONSE


def test_invoke__when_bad_question__should_return_answer_not_in_context() -> None:
    llm = AI21ContextualAnswers()

    response = llm.invoke(input={"context": context, "question": _BAD_QUESTION})

    assert response == ANSWER_NOT_IN_CONTEXT_RESPONSE


def test_invoke__when_response_if_no_answer_passed__should_use_it() -> None:
    response_if_no_answer_found = "This should be the response"
    llm = AI21ContextualAnswers()

    response = llm.invoke(
        input={"context": context, "question": _BAD_QUESTION},
        response_if_no_answer_found=response_if_no_answer_found,
    )

    assert response == response_if_no_answer_found


def test_invoke_when_used_in_a_simple_chain_with_no_vectorstore() -> None:
    tsm = AI21ContextualAnswers()

    chain: Runnable = tsm | StrOutputParser()

    response = chain.invoke(
        {"context": context, "question": _GOOD_QUESTION},
    )

    assert response != ANSWER_NOT_IN_CONTEXT_RESPONSE
