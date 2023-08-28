"""**LangSmith** utilities.

This module provides utilities for connecting to `LangSmith <https://smith.langchain.com/>`_. For more information on LangSmith, see the `LangSmith documentation <https://docs.smith.langchain.com/>`_.

**Evaluation**

LangSmith helps you evaluate Chains and other language model application components using a number of LangChain evaluators.
An example of this is shown below, assuming you've created a LangSmith dataset called ``<my_dataset_name>``:

.. code-block:: python

    from langsmith import Client
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.smith import RunEvalConfig, run_on_dataset

    # Chains may have memory. Passing in a constructor function lets the
    # evaluation framework avoid cross-contamination between runs.
    def construct_chain():
        llm = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            llm,
            "What's the answer to {your_input_key}"
        )
        return chain

    # Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    evaluation_config = RunEvalConfig(
        evaluators=[
            "qa",  # "Correctness" against a reference answer
            "embedding_distance",
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
        evaluation=evaluation_config,
    )

You can also create custom evaluators by subclassing the
:class:`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`
or LangSmith's `RunEvaluator` classes.

.. code-block:: python

    from typing import Optional
    from langchain.evaluation import StringEvaluator

    class MyStringEvaluator(StringEvaluator):
        
        @property
        def requires_input(self) -> bool:
            return False
        
        @property
        def requires_reference(self) -> bool:
            return True
        
        @property
        def evaluation_name(self) -> str:
            return "exact_match"
        
        def _evaluate_strings(self, prediction, reference=None, input=None, **kwargs) -> dict:
            return {"score": prediction == reference}


    evaluation_config = RunEvalConfig(
        custom_evaluators = [MyStringEvaluator()],
    )

    run_on_dataset(
        client,
        "<my_dataset_name>",
        construct_chain,
        evaluation=evaluation_config,
    )    

**Primary Functions**

- :func:`arun_on_dataset <langchain.smith.evaluation.runner_utils.arun_on_dataset>`: Asynchronous function to evaluate a chain, agent, or other LangChain component over a dataset.
- :func:`run_on_dataset <langchain.smith.evaluation.runner_utils.run_on_dataset>`: Function to evaluate a chain, agent, or other LangChain component over a dataset.
- :class:`RunEvalConfig <langchain.smith.evaluation.config.RunEvalConfig>`: Class representing the configuration for running evaluation. You can select evaluators by :class:`EvaluatorType <langchain.evaluation.schema.EvaluatorType>` or config, or you can pass in `custom_evaluators`
"""  # noqa: E501
from langchain.smith.evaluation import (
    RunEvalConfig,
    arun_on_dataset,
    run_on_dataset,
)

__all__ = [
    "arun_on_dataset",
    "run_on_dataset",
    "ChoicesOutputParser",
    "RunEvalConfig",
]
