import functools
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import pytest
from langchainplus_sdk import LangChainPlusClient, RunEvaluator
from langchainplus_sdk.schemas import Example

from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.run_evaluators.implementations import (
    get_criteria_evaluator,
    get_qa_evaluator,
)
from langchain.sql_database import SQLDatabase
from langchain.callbacks.tracers.run_stack import RunStackCallbackHandler
from langchain.callbacks.tracers.schemas import Run

_DIR = Path(__file__).parent.resolve()
_TEST_RUN_ID = uuid4().hex
_CLIENT = LangChainPlusClient()
_EVAL_LLM = ChatOpenAI(temperature=0)
_EVALUATORS = [
    get_qa_evaluator(
        _EVAL_LLM, input_key="query", prediction_key="result", answer_key="answer"
    ),
    get_criteria_evaluator(
        _EVAL_LLM, "helpfulness", input_key="query", prediction_key="result"
    ),
]

dataset_examples = [
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "How many albums are there"},
        "outputs": {"answer": "347"},
        "id": "b82f3498-5f2c-4e02-9bc3-799e88c7859a",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {
            "query": "How many more Protected AAC audio files are there than Protected MPEG-4 video file?"
        },
        "outputs": {"answer": "23"},
        "id": "baafdea1-ed98-4eda-9161-f953ca771377",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "What is the most common media type?"},
        "outputs": {"answer": "Purchased AAC audio file"},
        "id": "f1748c31-5cd4-4409-a88e-35412a8c7cb5",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "What is the most common media type?"},
        "outputs": {"answer": "MPEG audio file"},
        "id": "fc076f71-fbff-4401-aa86-a66a46cf7de5",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "What is the most common genre of songs?"},
        "outputs": {"answer": "Rock"},
        "id": "14f7bf41-8037-4499-b3fc-ac5429a3aee0",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "Where is Mark Telus from?"},
        "outputs": {"answer": "Edmonton, Canada"},
        "id": "e7481287-cc6a-46e1-a42a-e071f1371127",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "How many employees are also customers?"},
        "outputs": {"answer": "None"},
        "id": "070c3639-9e69-4810-a22d-9b2897e22e67",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "What are some example tracks by Bach?"},
        "outputs": {
            "answer": "'Concerto for 2 Violins in D Minor, BWV 1043: I. Vivace', 'Aria Mit 30 Veränderungen, BWV 988 'Goldberg Variations': Aria', and 'Suite for Solo Cello No. 1 in G Major, BWV 1007: I. Prélude'"
        },
        "id": "99a9f7ee-4fe7-4bf0-bec2-bf8e1a60c798",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {
            "query": "What are some example tracks by composer Johann Sebastian Bach?"
        },
        "outputs": {
            "answer": "'Concerto for 2 Violins in D Minor, BWV 1043: I. Vivace', 'Aria Mit 30 Veränderungen, BWV 988 'Goldberg Variations': Aria', and 'Suite for Solo Cello No. 1 in G Major, BWV 1007: I. Prélude'"
        },
        "id": "3f683e5e-314c-4d8c-98f9-6456e1c27e70",
    },
    {
        "dataset_id": "ad07b2df-2c73-4ca7-98ca-e67bc606536f",
        "inputs": {"query": "How many employees are there?"},
        "outputs": {"answer": "8"},
        "id": "85b3ae48-26c0-495f-95bf-7fa9cf271897",
    },
]


@pytest.fixture(scope="module")
def database() -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{_DIR}/data/Chinook.db")


@pytest.fixture(scope="module")
def chain_to_test(database: SQLDatabase) -> SQLDatabaseChain:
    llm = ChatOpenAI(temperature=0.0)
    return SQLDatabaseChain.from_llm(llm, database)


@pytest.fixture(scope="module", params=dataset_examples)
def run_example_pair(request, chain_to_test: SQLDatabaseChain) -> Tuple[Run, Example]:
    example = Example(**request.param)
    run_stack = RunStackCallbackHandler()
    with tracing_v2_enabled(
        session_name=f"test_chain_on_example-{_TEST_RUN_ID}", example_id=example.id
    ):
        chain_to_test(example.inputs, callbacks=[run_stack])
        return (run_stack.pop(), example)


@pytest.mark.parametrize("evaluator", _EVALUATORS)
def test_chain_on_example(
    evaluator: RunEvaluator, run_example_pair: Tuple[Run, Example]
) -> None:
    run, example = run_example_pair
    evaluation_result = _CLIENT.evaluate_run(run, evaluator, reference_example=example)
    assert (
        evaluation_result.score == 1
    ), f"My Chain failed evaluation {evaluation_result.key}\n\n{evaluation_result}"
