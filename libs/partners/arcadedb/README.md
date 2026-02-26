# langchain-arcadedb

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-arcadedb?label=%20)](https://pypi.org/project/langchain-arcadedb/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-arcadedb)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-arcadedb)](https://pypistats.org/packages/langchain-arcadedb)

## Quick Install

```bash
pip install langchain-arcadedb
```

## What is this?

This package contains the LangChain integration with [ArcadeDB](https://arcadedb.com),
the multi-model database with native Bolt protocol support and 97.8% OpenCypher
compatibility. It provides a graph store that works as a drop-in replacement for
Neo4j-based graph integrations.

Key features:

- **Bolt protocol** — connects via the standard Neo4j Python driver
- **APOC-free** — schema introspection and document import use pure Cypher
- **GraphStore protocol** — works with `GraphCypherQAChain` out of the box
- **Graph document import** — batched `MERGE` operations grouped by type

## Quick Start

```python
from langchain_arcadedb import ArcadeDBGraph

graph = ArcadeDBGraph(
    url="bolt://localhost:7687",
    username="root",
    password="playwithdata",
    database="mydb",
)

# Schema is auto-detected
print(graph.get_schema)

# Run Cypher queries
result = graph.query("MATCH (n:Person) RETURN n.name AS name")
```

## Use with GraphCypherQAChain

```python
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

answer = chain.invoke({"query": "Who does Alice know?"})
```

## Starting ArcadeDB with Bolt

```bash
docker run --rm -p 2480:2480 -p 7687:7687 \
    -e JAVA_OPTS="-Darcadedb.server.plugins=Bolt:com.arcadedb.bolt.BoltProtocolPlugin" \
    -e arcadedb.server.rootPassword=playwithdata \
    arcadedata/arcadedb:latest
```

## Documentation

- [ArcadeDB Documentation](https://docs.arcadedb.com)
- [LangChain Documentation](https://docs.langchain.com)
