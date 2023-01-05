# OpenAI

This page covers how to use the OpenAI ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific OpenAI wrappers.

## Installation and Setup
- Install the Python SDK with `pip install openai`
- Get an OpenAI api key and set it as an environment variable (`OPENAI_API_KEY`)
- If you want to use OpenAI's tokenizer (only available for Python 3.9+), install it with `pip install tiktoken`

## Wrappers

### LLM

There exists an OpenAI LLM wrapper, which you can access with 
```python
from langchain.llms import OpenAI
```

If you are using a model hosted on Azure, you should use different wrapper for that:
```python
from langchain.llms import AzureOpenAI
```
For a more detailed walkthrough of the Azure wrapper, see [this notebook](../modules/llms/integrations/azure_openai_example.ipynb)



### Embeddings

There exists an OpenAI Embeddings wrapper, which you can access with 
```python
from langchain.embeddings import OpenAIEmbeddings
```
For a more detailed walkthrough of this, see [this notebook](../modules/utils/combine_docs_examples/embeddings.ipynb)


### Tokenizer

There are several places you can use the `tiktoken` tokenizer. By default, it is used to count tokens
for OpenAI LLMs.

You can also use it to count tokens when splitting documents with 
```python
from langchain.text_splitter import CharacterTextSplitter
CharacterTextSplitter.from_tiktoken_encoder(...)
```
For a more detailed walkthrough of this, see [this notebook](../modules/utils/combine_docs_examples/textsplitter.ipynb)

### Moderation
You can also access the OpenAI content moderation endpoint with

```python
from langchain.chains import OpenAIModerationChain
```
For a more detailed walkthrough of this, see [this notebook](../modules/chains/examples/moderation.ipynb)
