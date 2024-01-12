"""Test LLM Bash functionality."""
import sys
from typing import Type

import pytest

from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_chain import (
    ContextQAEvalChain,
    CotQAEvalChain,
    QAEvalChain,
    _parse_string_eval_output,
)
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_eval_chain() -> None:
    """Test a simple eval chain."""
    example = {"query": "What's my name", "answer": "John Doe"}
    prediction = {"result": "John Doe"}
    fake_qa_eval_chain = QAEvalChain.from_llm(FakeLLM())

    outputs = fake_qa_eval_chain.evaluate([example, example], [prediction, prediction])
    assert outputs[0] == outputs[1]
    assert fake_qa_eval_chain.output_key in outputs[0]
    assert outputs[0][fake_qa_eval_chain.output_key] == "foo"


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
@pytest.mark.parametrize("chain_cls", [ContextQAEvalChain, CotQAEvalChain])
def test_context_eval_chain(chain_cls: Type[ContextQAEvalChain]) -> None:
    """Test a simple eval chain."""
    example = {
        "query": "What's my name",
        "context": "The name of this person is John Doe",
    }
    prediction = {"result": "John Doe"}
    fake_qa_eval_chain = chain_cls.from_llm(FakeLLM())

    outputs = fake_qa_eval_chain.evaluate([example, example], [prediction, prediction])
    assert outputs[0] == outputs[1]
    assert "text" in outputs[0]
    assert outputs[0]["text"] == "foo"


@pytest.mark.parametrize("chain_cls", [QAEvalChain, ContextQAEvalChain, CotQAEvalChain])
def test_implements_string_evaluator_protocol(
    chain_cls: Type[LLMChain],
) -> None:
    assert issubclass(chain_cls, StringEvaluator)


@pytest.mark.parametrize("chain_cls", [QAEvalChain, ContextQAEvalChain, CotQAEvalChain])
def test_returns_expected_results(
    chain_cls: Type[LLMChain],
) -> None:
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"}, sequential_responses=True
    )
    chain = chain_cls.from_llm(fake_llm)  # type: ignore
    results = chain.evaluate_strings(
        prediction="my prediction", reference="my reference", input="my input"
    )
    assert results["score"] == 1


@pytest.mark.parametrize(
    "output,expected",
    [
        (
            """ GRADE: CORRECT

QUESTION: according to the passage, what is the main reason that the author wrote this passage?
STUDENT ANSWER: to explain the importance of washing your hands
TRUE ANSWER: to explain the importance of washing your hands
GRADE:""",  # noqa: E501
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
        (
            """ Here is my step-by-step reasoning to grade the student's answer:

1. The question asks who founded the Roanoke settlement.

2. The context states that the grade incorrect answer is Walter Raleigh. 

3. The student's answer is "Sir Walter Raleigh".

4. The student's answer matches the context, which states the answer is Walter Raleigh. 

5. The addition of "Sir" in the student's answer does not contradict the context. It provides extra detail about Walter Raleigh's title, but the core answer of Walter Raleigh is still correct.

6. Therefore, the student's answer contains the same factual information as the true answer, so it should be graded as correct.

GRADE: CORRECT""",  # noqa: E501
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
        (
            """  CORRECT

QUESTION: who was the first president of the united states?
STUDENT ANSWER: George Washington 
TRUE ANSWER: George Washington was the first president of the United States.
GRADE:""",
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
        (
            """The student's answer is "Regent's Park," which matches the correct answer given in the context. Therefore, the student's answer is CORRECT.""",  # noqa: E501
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
    ],
)
def test_qa_output_parser(output: str, expected: dict) -> None:
    expected["reasoning"] = output.strip()
    assert _parse_string_eval_output(output) == expected
