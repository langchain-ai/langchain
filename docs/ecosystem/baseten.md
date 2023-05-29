# Baseten

Learn how to use LangChain with models deployed on Baseten.

## Installation and setup

- Create a [Baseten](https://baseten.co) account and [API key](https://docs.baseten.co/settings/api-keys).
- Install the Baseten Python client with `pip install baseten`
- Use your API key to authenticate with `baseten login`

## Invoking a model

Baseten integrates with LangChain through the LLM module, which provides a standardized and interoperable interface for models that are deployed on your Baseten workspace.

You can deploy foundation models like LLaMA and FLAN-T5 with one click from the [Baseten model library](https://app.baseten.co/explore/) or if you have your own model, [deploy it with this tutorial](https://docs.baseten.co/deploying-models/deploy).

In this example, we'll work with LLaMA. [Deploy LLaMA here](https://app.baseten.co/explore/llama) and follow along with the deployed [model's ID](https://docs.baseten.co/managing-models/manage).

```python
from langchain.llms import Baseten

llama = Baseten(model="MODEL_ID", verbose=True)

llama("Answer this question: What animals grow wool for making sweaters?")
```
