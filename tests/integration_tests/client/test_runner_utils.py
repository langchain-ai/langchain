import sys
from typing import Iterator, List
from uuid import uuid4

import pytest
from langsmith import Client as Client
from langsmith.schemas import DataType

from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.client.runner_utils import InputFormatError, run_on_dataset
from langchain.evaluation import EvaluatorType
from langchain.evaluation.run_evaluators import RunEvalConfig
from langchain.llms.openai import OpenAI


@pytest.fixture(
    scope="module",
)
def kv_dataset_name() -> Iterator[str]:
    import pandas as pd

    client = Client()
    df = pd.DataFrame(
        {
            "some_input": [
                "What's the capital of California?",
                "What's the capital of Nevada?",
                "What's the capital of Oregon?",
                "What's the capital of Washington?",
            ],
            "other_input": [
                "a",
                "b",
                "c",
                "d",
            ],
            "some_output": ["Sacramento", "Carson City", "Salem", "Olympia"],
            "other_output": ["e", "f", "g", "h"],
        }
    )

    uid = str(uuid4())[-8:]
    _dataset_name = f"lcp kv dataset integration tests - {uid}"
    client.upload_dataframe(
        df,
        name=_dataset_name,
        input_keys=["some_input", "other_input"],
        output_keys=["some_output", "other_output"],
        description="Integration test dataset",
    )
    yield _dataset_name


def test_chat_model(kv_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(kv_dataset_name, llm, evaluation=eval_config)
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match language model"
    ):
        run_on_dataset(kv_dataset_name, llm, evaluation=eval_config)

    def input_mapper(d: dict) -> dict:
        return {"input": d["some_input"]}

    results = run_on_dataset(
        kv_dataset_name, llm, evaluation=eval_config, input_mapper=input_mapper
    )
    print("CHAT", results, file=sys.stderr)


def test_llm(kv_dataset_name: str) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(kv_dataset_name, llm, evaluation=eval_config)
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match language model"
    ):
        run_on_dataset(kv_dataset_name, llm, evaluation=eval_config)

    def input_mapper(d: dict) -> dict:
        return {"input": d["some_input"]}

    results = run_on_dataset(
        kv_dataset_name, llm, evaluation=eval_config, input_mapper=input_mapper
    )
    print("LLM", results, file=sys.stderr)


def test_chain(kv_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(kv_dataset_name, lambda: chain, evaluation=eval_config)
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match chain input keys"
    ):
        run_on_dataset(kv_dataset_name, lambda: chain, evaluation=eval_config)

    def input_mapper(d: dict) -> dict:
        return {"input": d["some_input"]}

    results = run_on_dataset(
        kv_dataset_name,
        lambda: chain,
        evaluation=eval_config,
        input_mapper=input_mapper,
    )
    print("CHAIN", results, file=sys.stderr)


### Testing Chat Datasets


@pytest.fixture(
    scope="module",
)
def chat_dataset_name() -> Iterator[str]:
    def _create_message(txt: str, role: str = "human") -> List[dict]:
        return [{"type": role, "data": {"content": txt}}]

    import pandas as pd

    client = Client()
    df = pd.DataFrame(
        {
            "input": [
                _create_message(txt)
                for txt in (
                    "What's the capital of California?",
                    "What's the capital of Nevada?",
                    "What's the capital of Oregon?",
                    "What's the capital of Washington?",
                )
            ],
            "output": [
                _create_message(txt, role="ai")[0]
                for txt in ("Sacramento", "Carson City", "Salem", "Olympia")
            ],
        }
    )

    uid = str(uuid4())[-8:]
    _dataset_name = f"lcp chat dataset integration tests - {uid}"
    ds = client.create_dataset(
        _dataset_name, description="Integration test dataset", data_type=DataType.chat
    )
    for row in df.itertuples():
        client.create_example(
            dataset_id=ds.id,
            inputs={"input": row.input},
            outputs={"output": row.output},
        )
    yield _dataset_name


def test_chat_model_on_chat_dataset(chat_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    results = run_on_dataset(chat_dataset_name, llm, evaluation=eval_config)
    print("CHAT", results, file=sys.stderr)


def test_llm_on_chat_dataset(chat_dataset_name: str) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    results = run_on_dataset(
        chat_dataset_name,
        llm,
        evaluation=eval_config,
    )
    print("LLM", results, file=sys.stderr)


def test_chain_on_chat_dataset(chat_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=chat"
    ):
        run_on_dataset(
            chat_dataset_name,
            lambda: chain,
            evaluation=eval_config,
        )


@pytest.fixture(
    scope="module",
)
def llm_dataset_name() -> Iterator[str]:
    import pandas as pd

    client = Client()
    df = pd.DataFrame(
        {
            "input": [
                "What's the capital of California?",
                "What's the capital of Nevada?",
                "What's the capital of Oregon?",
                "What's the capital of Washington?",
            ],
            "output": ["Sacramento", "Carson City", "Salem", "Olympia"],
        }
    )

    uid = str(uuid4())[-8:]
    _dataset_name = f"lcp llm dataset integration tests - {uid}"
    client.upload_dataframe(
        df,
        name=_dataset_name,
        input_keys=["input"],
        output_keys=["output"],
        description="Integration test dataset",
        data_type=DataType.llm,
    )
    yield _dataset_name


def test_chat_model_on_llm_dataset(llm_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    results = run_on_dataset(llm_dataset_name, llm, evaluation=eval_config)
    print("CHAT", results, file=sys.stderr)


def test_llm_on_llm_dataset(llm_dataset_name: str) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    results = run_on_dataset(
        llm_dataset_name,
        llm,
        evaluation=eval_config,
    )
    print("LLM", results, file=sys.stderr)


def test_chain_on_llm_dataset(llm_dataset_name: str) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=chat"
    ):
        run_on_dataset(
            llm_dataset_name,
            lambda: chain,
            evaluation=eval_config,
        )
