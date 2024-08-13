# langchain-databricks

This package contains the LangChain integration with Databricks

## Installation

```bash
pip install -U langchain-databricks
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatDatabricks` class exposes chat models from Databricks.

```python
from langchain_databricks import ChatDatabricks

llm = ChatDatabricks()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`DatabricksEmbeddings` class exposes embeddings from Databricks.

```python
from langchain_databricks import DatabricksEmbeddings

embeddings = DatabricksEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`DatabricksLLM` class exposes LLMs from Databricks.

```python
from langchain_databricks import DatabricksLLM

llm = DatabricksLLM()
llm.invoke("The meaning of life is")
```
