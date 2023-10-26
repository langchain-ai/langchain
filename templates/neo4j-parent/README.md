# Parent Document Retriever with Neo4j Vector Index

This template allows you to balance precise embeddings and context retention by splitting documents into smaller chunks and retrieving their original or larger text information.

## Set up Environment

You need to define the following environment variables

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Installation

```bash
# from inside your LangServe instance
poe add neo4j-parent
```
