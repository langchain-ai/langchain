# langchain-neo4j

This package contains the LangChain integration with Neo4j

## Installation

```bash
pip install -U langchain-neo4j
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatNeo4j` class exposes chat models from Neo4j.

```python
from langchain_neo4j import ChatNeo4j

llm = ChatNeo4j()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`Neo4jEmbeddings` class exposes embeddings from Neo4j.

```python
from langchain_neo4j import Neo4jEmbeddings

embeddings = Neo4jEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`Neo4jLLM` class exposes LLMs from Neo4j.

```python
from langchain_neo4j import Neo4jLLM

llm = Neo4jLLM()
llm.invoke("The meaning of life is")
```
