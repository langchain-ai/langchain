from typing import Iterator, List
from uuid import uuid4

import pytest
from langsmith import Client as Client
from langsmith.schemas import DataType

from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import EvaluatorType
from langchain.llms.openai import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.smith.evaluation import InputFormatError
from langchain.smith.evaluation.runner_utils import arun_on_dataset


def _check_all_feedback_passed(_project_name: str, client: Client) -> None:
    # Assert that all runs completed, all feedback completed, and that the
    # chain or llm passes for the feedback provided.
    runs = list(client.list_runs(project_name=_project_name, execution_order=1))
    if not runs:
        # Queue delays. We are mainly just smoke checking rn.
        return
    feedback = list(client.list_feedback(run_ids=[run.id for run in runs]))
    if not feedback:
        return
    assert all([f.score == 1 for f in feedback])


@pytest.fixture
def eval_project_name() -> str:
    return f"lcp integration tests - {str(uuid4())[-8:]}"


@pytest.fixture(scope="module")
def client() -> Client:
    return Client()


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


def test_chat_model(
    kv_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match language model"
    ):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )

    def input_mapper(d: dict) -> List[BaseMessage]:
        return [HumanMessage(content=d["some_input"])]

    run_on_dataset(
        client=client,
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        input_mapper=input_mapper,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_llm(kv_dataset_name: str, eval_project_name: str, client: Client) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match language model"
    ):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )

    def input_mapper(d: dict) -> str:
        return d["some_input"]

    run_on_dataset(
        client=client,
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        input_mapper=input_mapper,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_chain(kv_dataset_name: str, eval_project_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match chain input keys"
    ):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
            client=client,
        )

    def input_mapper(d: dict) -> dict:
        return {"input": d["some_input"]}

    with pytest.raises(
        InputFormatError,
        match=" match the chain's expected input keys.",
    ):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=lambda: input_mapper | chain,
            client=client,
            evaluation=eval_config,
        )

    def right_input_mapper(d: dict) -> dict:
        return {"question": d["some_input"]}

    run_on_dataset(
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=lambda: right_input_mapper | chain,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


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


def test_chat_model_on_chat_dataset(
    chat_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=chat_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        client=client,
        project_name=eval_project_name,
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_llm_on_chat_dataset(
    chat_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=chat_dataset_name,
        llm_or_chain_factory=llm,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_chain_on_chat_dataset(chat_dataset_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=chat"
    ):
        run_on_dataset(
            dataset_name=chat_dataset_name,
            client=client,
            llm_or_chain_factory=lambda: chain,
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


def test_chat_model_on_llm_dataset(
    llm_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        client=client,
        dataset_name=llm_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_llm_on_llm_dataset(
    llm_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        client=client,
        dataset_name=llm_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_chain_on_llm_dataset(llm_dataset_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=llm"
    ):
        run_on_dataset(
            client=client,
            dataset_name=llm_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
        )


@pytest.fixture(
    scope="module",
)
def kv_singleio_dataset_name() -> Iterator[str]:
    import pandas as pd

    client = Client()
    df = pd.DataFrame(
        {
            "the wackiest input": [
                "What's the capital of California?",
                "What's the capital of Nevada?",
                "What's the capital of Oregon?",
                "What's the capital of Washington?",
            ],
            "unthinkable output": ["Sacramento", "Carson City", "Salem", "Olympia"],
        }
    )

    uid = str(uuid4())[-8:]
    _dataset_name = f"lcp singleio kv dataset integration tests - {uid}"
    client.upload_dataframe(
        df,
        name=_dataset_name,
        input_keys=["the wackiest input"],
        output_keys=["unthinkable output"],
        description="Integration test dataset",
    )
    yield _dataset_name


def test_chat_model_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        client=client,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_llm_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=llm,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


def test_chain_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=lambda: chain,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


@pytest.mark.asyncio
async def test_runnable_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    runnable = (
        ChatPromptTemplate.from_messages([("human", "{the wackiest input}")])
        | ChatOpenAI()
    )
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    await arun_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=runnable,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


@pytest.mark.asyncio
async def test_arb_func_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    runnable = (
        ChatPromptTemplate.from_messages([("human", "{the wackiest input}")])
        | ChatOpenAI()
    )

    def my_func(x: dict) -> str:
        return runnable.invoke(x).content

    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    await arun_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=my_func,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)
