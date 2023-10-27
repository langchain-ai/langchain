# Neo4j Knowledge Graph: Enhanced mapping from text to database using a full-text index

This template allows you to chat with Neo4j graph database in natural language, using an OpenAI LLM.
The addition of the full-text index ensures efficient mapping of values from text to database for more precise Cypher statement generation.

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
poe add neo4j-cypher
```
