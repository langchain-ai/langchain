# Confident AI

>[Confident AI](https://www.confident-ai.com/) is the cloud platform for [DeepEval](https://deepeval.com/), the most widely adopted open-source framework to evaluate LLM applications such as RAG pipelines, agentics, chatbots, or even just an LLM itself.


## Tracing LangChain

Confident AI provides a [`CallbackHandler`](https://documentation.confident-ai.com/docs/llm-tracing/integrations/langchain) that can be used to trace LangChain’s execution. Therefore, you can trace the entire execution of your agent by creating nested spans of every step that your agent takes to complete the task. 

### Quickstart

Install the following packages:

```bash
pip install -U deepeval langchain langchain-openai
```

Log in with your [API key](https://app.confident-ai.com/auth/signup) and configure DeepEval’s `CallbackHandler` as a callback:

```python
import os
import time
from langchain.chat_models import init_chat_model
 
from deepeval import login_with_confident_api_key
from deepeval.integrations.langchain.callback import CallbackHandler
 
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
login_with_confident_api_key("<your-confident-api-key>")
 
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b
 
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools([multiply])
 
llm_with_tools.invoke(
    "What is 3 * 12?", 
    config = {"callbacks": [CallbackHandler()]}
)
 
time.sleep(5) # Wait for the trace to be published
```

### View traces in the Observatory
On the Confident AI dashboard, go to Observatory > Traces and click on the latest trace. 

![Confident AI Observatory](https://github.com/spike-spiegel-21/confident-docs/blob/mayank/img/public/img/Screenshot%202025-07-09%20at%204.38.13%E2%80%AFPM.png?raw=true)

