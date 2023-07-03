"""Test LLM Bash functionality."""
import sys
from typing import Type

import pytest

from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_chain import (
    ContextQAEvalChain,
    CotQAEvalChain,
    QAEvalChain,
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
    assert "text" in outputs[0]
    assert outputs[0]["text"] == "foo"


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
    assert isinstance(chain_cls, StringEvaluator)


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
