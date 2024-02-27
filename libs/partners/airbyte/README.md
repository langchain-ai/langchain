# langchain-airbyte

This package contains the LangChain integration with Airbyte

## Installation

```bash
pip install -U langchain-airbyte
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatAirbyte` class exposes chat models from Airbyte.

```python
from langchain_airbyte import ChatAirbyte

llm = ChatAirbyte()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`AirbyteEmbeddings` class exposes embeddings from Airbyte.

```python
from langchain_airbyte import AirbyteEmbeddings

embeddings = AirbyteEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`AirbyteLLM` class exposes LLMs from Airbyte.

```python
from langchain_airbyte import AirbyteLLM

llm = AirbyteLLM()
llm.invoke("The meaning of life is")
```
