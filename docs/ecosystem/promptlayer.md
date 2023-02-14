# PromptLayer

This page covers how to use [PromptLayer](https://www.promptlayer.com) within LangChain.
It is broken into two parts: installation and setup, and then references to specific PromptLayer wrappers.

## Installation and Setup

If you want to work with PromptLayer:
- Install the promptlayer python library `pip install promptlayer`
- Create a PromptLayer account
- Create an api token and set it as an environment variable (`PROMPTLAYER_API_KEY`)

## Wrappers

### LLM

There exists an PromptLayer OpenAI LLM wrapper, which you can access with
```python
from langchain.llms import PromptLayerOpenAI
```

To tag your requests, use the argument `pl_tags` when instanializing the LLM
```python
from langchain.llms import PromptLayerOpenAI
llm = PromptLayerOpenAI(pl_tags=["langchain-requests", "chatbot"])
```

This LLM is identical to the [OpenAI LLM](./openai), except that
- all your requests will be logged to your PromptLayer account
- you can add `pl_tags` when instantializing to tag your requests on PromptLayer

