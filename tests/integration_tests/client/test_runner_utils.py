import sys
from typing import Iterator
from uuid import uuid4

import pytest
from langchainplus_sdk import LangChainPlusClient as Client

from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.client.runner_utils import run_on_dataset
from langchain.evaluation import EvaluatorType
from langchain.evaluation.run_evaluators.loading import load_run_evaluators_for_model
from langchain.llms.openai import OpenAI


@pytest.fixture(
    scope="module",
)
def dataset_name() -> Iterator[str]:
    import pandas as pd

    client = Client()
    df = pd.DataFrame(
        [
            {"question": "5", "answer": 5.0},
            {"question": "5 + 3", "answer": 8.0},
            {"question": "2^3.171", "answer": 9.006708689094099},
            {"question": "  2 ^3.171 ", "answer": 9.006708689094099},
        ]
    )

    uid = str(uuid4())[-8:]
    _dataset_name = f"lcp integration tests - {uid}"
    client.upload_dataframe(
        df,
        name=_dataset_name,
        input_keys=["question"],
        output_keys=["answer"],
        description="Integration test dataset",
    )
    yield _dataset_name


def test_chat_model(dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    evaluators = load_run_evaluators_for_model(
        [EvaluatorType.QA, EvaluatorType.CRITERIA], llm, reference_key="answer"
    )
    results = run_on_dataset(
        dataset_name,
        llm,
        run_evaluators=evaluators,
    )
    print("CHAT", results, file=sys.stderr)


def test_llm(dataset_name: str) -> None:
    llm = OpenAI(temperature=0)
    evaluators = load_run_evaluators_for_model(
        [EvaluatorType.QA, EvaluatorType.CRITERIA], llm, reference_key="answer"
    )
    results = run_on_dataset(
        dataset_name,
        llm,
        run_evaluators=evaluators,
    )
    print("LLM", results, file=sys.stderr)


def test_chain(dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    evaluators = load_run_evaluators_for_model(
        [EvaluatorType.QA, EvaluatorType.CRITERIA], chain, reference_key="answer"
    )
    results = run_on_dataset(
        dataset_name,
        lambda: chain,
        run_evaluators=evaluators,
    )
    print("CHAIN", results, file=sys.stderr)
