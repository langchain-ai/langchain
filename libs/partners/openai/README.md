# langchain-openai

This package contains the LangChain integrations for OpenAI through their `openai` SDK.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install langchain-openai
```
- Get an OpenAI api key and set it as an environment variable (`OPENAI_API_KEY`)


## LLM

See a [usage example](http://python.langchain.com/docs/integrations/llms/openai).

```python
from langchain_openai import OpenAI
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:
```python
from langchain_openai import AzureOpenAI
```
For a more detailed walkthrough of the `Azure` wrapper, see [here](http://python.langchain.com/docs/integrations/llms/azure_openai)


## Chat model

See a [usage example](http://python.langchain.com/docs/integrations/chat/openai).

```python
from langchain_openai import ChatOpenAI
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:
```python
from langchain_openai import AzureChatOpenAI
```
For a more detailed walkthrough of the `Azure` wrapper, see [here](http://python.langchain.com/docs/integrations/chat/azure_chat_openai)


## Text Embedding Model

See a [usage example](http://python.langchain.com/docs/integrations/text_embedding/openai)

```python
from langchain_openai import OpenAIEmbeddings
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:
```python
from langchain_openai import AzureOpenAIEmbeddings
```
For a more detailed walkthrough of the `Azure` wrapper, see [here](https://python.langchain.com/docs/integrations/text_embedding/azureopenai)