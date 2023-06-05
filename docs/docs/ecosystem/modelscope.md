# ModelScope

This page covers how to use the modelscope ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific modelscope wrappers.

## Installation and Setup

* Install the Python SDK with `pip install modelscope`

## Wrappers

### Embeddings

There exists a modelscope Embeddings wrapper, which you can access with 

```python
from langchain.embeddings import ModelScopeEmbeddings
```

For a more detailed walkthrough of this, see [this notebook](../modules/models/text_embedding/examples/modelscope_hub.ipynb)
