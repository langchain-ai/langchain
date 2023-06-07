# scikit-learn

This page covers how to use the scikit-learn package within LangChain.
It is broken into two parts: installation and setup, and then references to specific scikit-learn wrappers.

## Installation and Setup

- Install the Python package with `pip install scikit-learn`

## Wrappers

### VectorStore

`SKLearnVectorStore` provides a simple wrapper around the nearest neighbor implementation in the
scikit-learn package, allowing you to use it as a vectorstore.

To import this vectorstore:

```python
from langchain.vectorstores import SKLearnVectorStore
```

For a more detailed walkthrough of the SKLearnVectorStore wrapper, see [this notebook](../modules/indexes/vectorstores/examples/sklearn.ipynb).
