# Confident AI

>[Confident AI](https://www.confident-ai.com/) is the cloud platform for [DeepEval](https://deepeval.com/), the most widely adopted open-source framework to evaluate LLM applications such as RAG pipelines, agentics, chatbots, or even just an LLM itself.


## Tracing LangChain

Confident AI provides a [`CallbackHandler`](https://documentation.confident-ai.com/docs/llm-tracing/integrations/langchain) that can be used to trace LangChain’s execution. Therefore, you can trace the entire execution of your agent by creating nested spans of every step that your agent takes to complete the task. 

### Quickstart

Install the following packages:

```bash
pip install -U deepeval langgraph langchain langchain-openai
```

Log in with your [API key](https://app.confident-ai.com/auth/signup) and configure DeepEval’s `CallbackHandler` as a callback:

```python
import os
import time
from langgraph.prebuilt import create_react_agent
 
from deepeval.integrations.langchain.callback import CallbackHandler
import deepeval
 
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
deepeval.login_with_confident_api_key("<your-confident-api-key>")
 
def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"
 
agent = create_react_agent(
    model="openai:gpt-4o-mini",  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)
 
def my_llm_app(input: str) -> str:

    result = agent.invoke(
        input = {"messages": [{"role": "user", "content": input}]}, 
        config = {"callbacks": [CallbackHandler()]}
    )

    return result["messages"][-1].content

print(my_llm_app("what is the weather in sf"))
time.sleep(5) # Wait for the trace to be published
```

Run your agent by executing the script.

```bash
python main.py
```

### View traces in the Observatory
On the Confident AI dashboard, go to Observatory > Traces and click on the latest trace. 

![Confident AI Observatory]([https://github.com/spike-spiegel-21/confident-docs/blob/mayank/img/public/img/Screenshot%202025-07-09%20at%204.38.13%E2%80%AFPM.png?raw=true](https://raw.githubusercontent.com/spike-spiegel-21/confident-docs/refs/heads/mayank/img/public/img/langgraph_trace.png))


## Run end-to-end evals on your agent
You can run end-to-end evaluations on the overall inputs and outputs of your agent using these simple steps. 

1. Create a metric locally via DeepEval (more info [here](https://documentation.confident-ai.com/docs/llm-evaluation/metrics/create-locally) if unsure):
2. Pull your dataset to create some test cases for evaluation, read about using datasets on Confident AI [here](https://documentation.confident-ai.com/docs/dataset-editor/introduction).
3. Run evaluations. DeepEval will pass in the inputs of each individual golden in your dataset to invoke your agent.


### Evals In CI/CD

```python
import pytest
 
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import assert_test
from main import my_llm_app
 
 
dataset = EvaluationDataset()
dataset.pull(alias="your-dataset-alias") # Pull dataset from Confident AI
 
@pytest.mark.parametrize("golden", dataset.goldens) # Loop through goldens
def test_llm_app(golden: Golden):
    test_case = LLMTestCase(
      input=golden.input, 
      actual_output=my_llm_app(golden.input) # Replace with your LLM app
    )
 
    # Replace with your metrics
    assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])
```

### Evals In Python script

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
from main import my_llm_app
 
dataset = EvaluationDataset()
dataset.pull(alias="your-dataset-alias") # Pull dataset from Confident AI
 
# Process each golden in your dataset
for golden in dataset.goldens:
    input = golden.input
    test_case = LLMTestCase(input=input, actual_output=my_llm_app(input))
    dataset.test_cases.append(test_case)
 
# Run an evaluation
evaluate(
    test_cases=dataset.test_cases,
    metrics=[AnswerRelevancyMetric()], # Replace with your metrics
)
```
