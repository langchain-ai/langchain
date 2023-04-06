# Marqo

This page covers how to use the Marqo ecosystem within LangChain.

**What is Marqo?**

Marqo is a tensor search engine that uses embeddings stored in in-memory HNSW indexes to achieve cutting edge search speeds. Marqo can scale to hundred-million document indexes with horizontal index sharding and allows for async and non-blocking data upload and search. Marqo uses the latest machine learning models from PyTorch, Huggingface, OpenAI and more you can start with a pre-configured model or bring your own. The built in ONNX support and conversion allows for faster inference and higher throughput on both CPU and GPU support.

Because marqo include its own inference your documents and queries can have a mix of text and images, you don't need to provide embeddings!

Deployment of Marqo is flexible, you can get started yourself with our docker image or (contact use about our managed cloud offering!)[https://www.marqo.ai/pricing]

## Installation and Setup
- Install the Python SDK with `pip install marqo`

## Wrappers

### VectorStore

There exists a wrapper around Marqo indexes, allowing you to use it as a knowledgestore within the vectorstore framework. Marqo lets you select from a range of models for generating embeddings and exposes some preprocessing configurations, Marqo can also work with multimodel indexes, documents and queries can both have a mix of images and text, for more information refer to [our documentation](https://docs.marqo.ai/latest/).

Marqo can be used locally with our docker image, [see our getting started.](https://docs.marqo.ai/latest/)

To import this vectorstore:
```python
from langchain.vectorstores import Marqo
```

For a more detailed walkthrough of the Marqo wrapper and some of its unique features, see [this notebook](../modules/indexes/vectorstores/examples/marqo.ipynb)
