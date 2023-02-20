# Cohere

This page covers how to use the Cohere ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Cohere wrappers.

## Installation and Setup
- Install the Python SDK with `pip install cohere`
- Get an Cohere api key and set it as an environment variable (`COHERE_API_KEY`)

## Wrappers

### LLM

There exists an Cohere LLM wrapper, which you can access with 
```python
from langchain.llms import Cohere
```

### Embeddings

There exists an Cohere Embeddings wrapper, which you can access with 
```python
from langchain.embeddings import CohereEmbeddings
```
For a more detailed walkthrough of this, see [this notebook](../modules/indexes/examples/embeddings.ipynb)
