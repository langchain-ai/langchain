from pathlib import Path
from uuid import uuid4

import pytest
from langchainplus_sdk import LangChainPlusClient
from langchainplus_sdk.schemas import Example

from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.run_evaluators.implementations import (
    get_criteria_evaluator,
    get_qa_evaluator,
)
from langchain.schema import RUN_KEY
from langchain.sql_database import SQLDatabase

from langchain.client.runner_utils import run_on_dataset

_DIR = Path(__file__).parent.resolve()
_TEST_RUN_ID = uuid4().hex
_CLIENT = LangChainPlusClient()
_EVAL_LLM = ChatOpenAI(temperature=0)
_DATASET_NAME = "sql-qa-chinook"
_EVALUATORS = [
    get_qa_evaluator(
        _EVAL_LLM, input_key="query", prediction_key="response", answer_key="answer"
    ),
    get_criteria_evaluator(
        _EVAL_LLM, "helpfulness", input_key="query", prediction_key="response"
    ),
]


@pytest.fixture(scope="module")
def database() -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{_DIR}/data/Chinook.db")


@pytest.fixture(scope="module")
def chain_to_test(database: SQLDatabase) -> SQLDatabaseChain:
    llm = ChatOpenAI(temperature=0.0)
    return SQLDatabaseChain.from_llm(llm, database)


@pytest.fixture(scope="module")
def chain_run_results(chain_to_test: SQLDatabaseChain) -> dict:
    results = run_on_dataset(
        lambda: chain_to_test,
        dataset_name=_DATASET_NAME,
        session_name=f"test_chain_on_example-{_TEST_RUN_ID}",
    )
    return results


@pytest.fixture(scope="module")
def chain_run_individual_results(chain_run_results: dict):
    return chain_run_results["results"]


@pytest.mark.parametrize("result", chain_run_individual_results, indirect=True)
def test_chain_run(result) -> None:
    assert result is not None, "Chain run failed"


@pytest.mark.parametrize("evaluator", _EVALUATORS)
def test_evaluators(chain_run_results: dict, evaluator) -> None:
    session_name = chain_run_results["session_name"]
    for run in _CLIENT.list_runs(session_name):
        evaluation_result = _CLIENT.evaluate_run(run, evaluator)
        if evaluation_result.score != 1:
            raise ValueError(
                f"My Chain failed evaluation {evaluation_result.key} for run: {run}\n\n{evaluation_result}"
            )
