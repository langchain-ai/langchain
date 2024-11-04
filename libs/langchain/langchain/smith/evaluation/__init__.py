"""LangSmith evaluation utilities.

This module provides utilities for evaluating Chains and other language model
applications using LangChain evaluators and LangSmith.

For more information on the LangSmith API, see the `LangSmith API documentation <https://docs.smith.langchain.com/docs/>`_.

**Example**

.. code-block:: python

    from langsmith import Client
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.smith import EvaluatorType, RunEvalConfig, run_on_dataset

    def construct_chain():
        llm = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            llm,
            "What's the answer to {your_input_key}"
        )
        return chain

    evaluation_config = RunEvalConfig(
        evaluators=[
            EvaluatorType.QA,  # "Correctness" against a reference answer
            EvaluatorType.EMBEDDING_DISTANCE,
            RunEvalConfig.Criteria("helpfulness"),
            RunEvalConfig.Criteria({
                "fifth-grader-score": "Do you have to be smarter than a fifth grader to answer this question?"
            }),
        ]
    )

    client = Client()
    run_on_dataset(
        client,
        "<my_dataset_name>",
        construct_chain,
        evaluation=evaluation_config
    )

**Attributes**

- ``arun_on_dataset``: Asynchronous function to evaluate a chain or other LangChain component over a dataset.
- ``run_on_dataset``: Function to evaluate a chain or other LangChain component over a dataset.
- ``RunEvalConfig``: Class representing the configuration for running evaluation.
- ``StringRunEvaluatorChain``: Class representing a string run evaluator chain.
- ``InputFormatError``: Exception raised when the input format is incorrect.

"""  # noqa: E501

from langchain.smith.evaluation.config import RunEvalConfig
from langchain.smith.evaluation.runner_utils import (
    InputFormatError,
    arun_on_dataset,
    run_on_dataset,
)
from langchain.smith.evaluation.string_run_evaluator import StringRunEvaluatorChain

__all__ = [
    "InputFormatError",
    "arun_on_dataset",
    "run_on_dataset",
    "StringRunEvaluatorChain",
    "RunEvalConfig",
]
