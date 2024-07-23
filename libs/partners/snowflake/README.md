# langchain-snowflake

This package contains the LangChain integration with Snowflake

## Installation

```bash
pip install -U langchain-snowflake
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSnowflake` class exposes chat models from Snowflake.

```python
from langchain_snowflake import ChatSnowflake

llm = ChatSnowflake()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SnowflakeEmbeddings` class exposes embeddings from Snowflake.

```python
from langchain_snowflake import SnowflakeEmbeddings

embeddings = SnowflakeEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SnowflakeLLM` class exposes LLMs from Snowflake.

```python
from langchain_snowflake import SnowflakeLLM

llm = SnowflakeLLM()
llm.invoke("The meaning of life is")
```
