# langchain-nomic

This package contains the LangChain integration with Nomic

## Installation

```bash
pip install -U langchain-nomic
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatNomic` class exposes chat models from Nomic.

```python
from langchain_nomic import ChatNomic

llm = ChatNomic()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`NomicEmbeddings` class exposes embeddings from Nomic.

```python
from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`NomicLLM` class exposes LLMs from Nomic.

```python
from langchain_nomic import NomicLLM

llm = NomicLLM()
llm.invoke("The meaning of life is")
```
