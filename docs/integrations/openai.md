# OpenAI

>[OpenAI](https://en.wikipedia.org/wiki/OpenAI) is American artificial intelligence (AI) research laboratory 
> consisting of the non-profit `OpenAI Incorporated`
> and its for-profit subsidiary corporation `OpenAI Limited Partnership`. 
> `OpenAI` conducts AI research with the declared intention of promoting and developing a friendly AI. 
> `OpenAI` systems run on an `Azure`-based supercomputing platform from `Microsoft`.

>The [OpenAI API](https://platform.openai.com/docs/models) is powered by a diverse set of models with different capabilities and price points.
> 
>[ChatGPT](https://chat.openai.com) is the Artificial Intelligence (AI) chatbot developed by `OpenAI`.

## Installation and Setup
- Install the Python SDK with
```bash
pip install openai
```
- Get an OpenAI api key and set it as an environment variable (`OPENAI_API_KEY`)
- If you want to use OpenAI's tokenizer (only available for Python 3.9+), install it
```bash
pip install tiktoken
```


## LLM

```python
from langchain.llms import OpenAI
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:
```python
from langchain.llms import AzureOpenAI
```
For a more detailed walkthrough of the `Azure` wrapper, see [this notebook](../modules/models/llms/integrations/azure_openai_example.ipynb)


## Text Embedding Model

```python
from langchain.embeddings import OpenAIEmbeddings
```
For a more detailed walkthrough of this, see [this notebook](../modules/models/text_embedding/examples/openai.ipynb)


## Chat Model

```python
from langchain.chat_models import ChatOpenAI
```
For a more detailed walkthrough of this, see [this notebook](../modules/models/chat/integrations/openai.ipynb)


## Tokenizer

There are several places you can use the `tiktoken` tokenizer. By default, it is used to count tokens
for OpenAI LLMs.

You can also use it to count tokens when splitting documents with 
```python
from langchain.text_splitter import CharacterTextSplitter
CharacterTextSplitter.from_tiktoken_encoder(...)
```
For a more detailed walkthrough of this, see [this notebook](../modules/indexes/text_splitters/examples/tiktoken.ipynb)

## Chain

See a [usage example](../modules/chains/examples/moderation.ipynb).

```python
from langchain.chains import OpenAIModerationChain
```

## Document Loader

See a [usage example](../modules/indexes/document_loaders/examples/chatgpt_loader.ipynb).

```python
from langchain.document_loaders.chatgpt import ChatGPTLoader
```

## Retriever

See a [usage example](../modules/indexes/retrievers/examples/chatgpt-plugin.ipynb).

```python
from langchain.retrievers import ChatGPTPluginRetriever
```
