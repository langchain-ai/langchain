from pathlib import Path
from uuid import uuid4

import pytest
from langchainplus_sdk import LangChainPlusClient

from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.run_evaluators.implementations import (
    get_criteria_evaluator,
    get_qa_evaluator,
)
from langchain.schema import RUN_KEY
from langchain.sql_database import SQLDatabase

_DIR = Path(__file__).parent.resolve()
_TEST_RUN_ID = uuid4().hex
_CLIENT = LangChainPlusClient()
_EVAL_LLM = ChatOpenAI(temperature=0)
_DATASET_NAME = "sql-qa-chinook"
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


def test_chain_on_example(chain_to_test: SQLDatabaseChain) -> None:
    run_ids = []
    for example in _CLIENT.list_examples(dataset_name=_DATASET_NAME):
        with tracing_v2_enabled(
            session_name=f"test_chain_on_example-{_TEST_RUN_ID}", example_id=example.id
        ):
            result = chain_to_test(example.inputs, include_run_info=True)
            run_ids.append(result[RUN_KEY].run_id)
    failures = []
    for run_id in run_ids:
        for evaluator in _EVALUATORS:
            evaluation_result = _CLIENT.evaluate_run(run_id, evaluator)
            if evaluation_result.score != 1:
                failures.append(
                    ValueError(
                        f"My Chain failed evaluation {evaluation_result.key}\n\n{evaluation_result}"
                    )
                )
    if failures:
        raise ValueError(
            f"Chain failed {len(failures)} out of {len(run_ids)} evaluations"
        )
