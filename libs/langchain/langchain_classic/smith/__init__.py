"""**LangSmith** utilities.

This module provides utilities for connecting to
[LangSmith](https://docs.langchain.com/langsmith/home).

**Evaluation**

LangSmith helps you evaluate Chains and other language model application components
using a number of LangChain evaluators.
An example of this is shown below, assuming you've created a LangSmith dataset
called `<my_dataset_name>`:

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.smith import RunEvalConfig, run_on_dataset


# Chains may have memory. Passing in a constructor function lets the
# evaluation framework avoid cross-contamination between runs.
def construct_chain():
    model = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(model, "What's the answer to {your_input_key}")
    return chain


# Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
evaluation_config = RunEvalConfig(
    evaluators=[
        "qa",  # "Correctness" against a reference answer
        "embedding_distance",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria(
            {
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }
        ),
    ]
)

client = Client()
run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```

You can also create custom evaluators by subclassing the
`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`
or LangSmith's `RunEvaluator` classes.

```python
from typing import Optional
from langchain_classic.evaluation import StringEvaluator


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

    def _evaluate_strings(
        self, prediction, reference=None, input=None, **kwargs
    ) -> dict:
        return {"score": prediction == reference}


evaluation_config = RunEvalConfig(
    custom_evaluators=[MyStringEvaluator()],
)

run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```

**Primary Functions**

- `arun_on_dataset <langchain.smith.evaluation.runner_utils.arun_on_dataset>`:
    Asynchronous function to evaluate a chain, agent, or other LangChain component over
    a dataset.
- `run_on_dataset <langchain.smith.evaluation.runner_utils.run_on_dataset>`:
    Function to evaluate a chain, agent, or other LangChain component over a dataset.
- `RunEvalConfig <langchain.smith.evaluation.config.RunEvalConfig>`:
    Class representing the configuration for running evaluation.
    You can select evaluators by
    `EvaluatorType <langchain.evaluation.schema.EvaluatorType>` or config,
    or you can pass in `custom_evaluators`.
"""

from langchain_classic.smith.evaluation import (
    RunEvalConfig,
    arun_on_dataset,
    run_on_dataset,
)

__all__ = [
    "RunEvalConfig",
    "arun_on_dataset",
    "run_on_dataset",
]
