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
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
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


@pytest.fixture(scope="module")
def database() -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{_DIR}/data/Chinook.db")


@pytest.fixture(scope="module")
def chain_to_test(database: SQLDatabase) -> SQLDatabaseChain:
    llm = ChatOpenAI(temperature=0.0)
    return SQLDatabaseChain.from_llm(llm, database)


@pytest.fixture(
    scope="module", params=_CLIENT.list_examples(dataset_name="sql-qa-chinook")
)
def run_example_pair(request, chain_to_test: SQLDatabaseChain) -> Tuple[Run, Example]:
    example: Example = request.param
    # TODO: Add context manager for this
    run_stack = RunCollectorCallbackHandler()
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
