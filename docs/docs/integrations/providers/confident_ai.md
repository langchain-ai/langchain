# Confident AI

>[Confident AI](https://www.confident-ai.com/) is the cloud platform for [DeepEval](https://deepeval.com/), the most widely adopted open-source framework to evaluate LLM applications such as RAG pipelines, agentics, chatbots, or even just an LLM itself.


## Tracing LangGraph

Confident AI provides a [`CallbackHandler`](https://documentation.confident-ai.com/docs/llm-tracing/integrations/langchain) that can be used to trace LangGraph’s execution. Therefore, you can trace the entire execution of your agent by creating nested spans of every step that your agent takes to complete the task. 

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

![Confident AI Observatory](https://confident-docs.s3.us-east-1.amazonaws.com/llm-tracing%3Alangchain.png)


## Run end-to-end evals on your agent
With DeepEval, you can run end-to-end evaluations on your agent by creating metrics, datasets. Supply metrics to the CallbackHandler. Then, use the dataset generator to invoke your LangGraph agent for each golden.

> Task completion metric is the only supported metric for end-to-end evaluations. Read more about metrics [here](https://deepeval.com/docs/metrics-task-completion).

### Example (synchronous)
```python
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import Golden, EvaluationDataset
 
...
 
# Create a metric
task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)
 
# Create goldens
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
 
dataset = EvaluationDataset(goldens=goldens)
 
# Run evaluation for each golden
for golden in dataset.evals_iterator():
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler(metrics=[task_completion])]}
    )
```

### Example (asynchronous)

```python
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.evaluate import test_run
 
...
 
# Create a metric
task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)
 
# Create goldens
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
 
dataset = EvaluationDataset(goldens=goldens)
 
# Run evaluation for each golden
for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent.ainvoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={"callbacks": [CallbackHandler(metrics=[task_completion])]}
        )
    )
    test_run.append(task)
```

## Running Evals in Production

Confident AI provides metric collection on the platform. Running evals in production is same as running it locally. However, might want it when you want evaluate your agents in production. Refer to the [documentation](https://documentation.confident-ai.com/docs/llm-evaluation/metrics/run-evals-in-production) for more details.


```python
from deepeval.integrations.langchain import CallbackHandler
...
 
# Invoke your agent with the metric collection name
agent.invoke(
    input = {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config = {"callbacks": [
        CallbackHandler(metric_collection="<metric-collection-name-with-task-completion>")
    ]}
)
```
