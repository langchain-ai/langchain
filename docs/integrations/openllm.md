# OpenLLM

This page demonstrates how to use [OpenLLM](https://github.com/bentoml/OpenLLM) with LangChain.

## Installation

To get started with OpenLLM, one can install it via PyPI:

```bash
pip install openllm
```

## Wrappers

### LLM

There is a OpenLLM Wrapper for both local [Runner](https://docs.bentoml.org/en/latest/concepts/runner.html)
and hosted-LLM on remote server.

To use the local Runner wrapper:

```python
from langchain.llms import OpenLLM

llm = OpenLLM.for_model("dolly-v2", model_id='databricks/dolly-v2-7b', device_map='auto')

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
```

To use it with a hosted-LLM on remote server:

```python
from langchain.llms import OpenLLM

llm = OpenLLM.for_model(server_url='http://123.23.123.1:3000', server_type='grpc')

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
```
