# langchain-nomic

This package contains the LangChain integration with Nomic

## Installation

```bash
pip install -U langchain-nomic
```

And you should configure credentials by setting the following environment variables:

* `NOMIC_API_KEY`: your nomic API key

## Embeddings

`NomicEmbeddings` class exposes embeddings from Nomic.

```python
from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings()
embeddings.embed_query("What is the meaning of life?")