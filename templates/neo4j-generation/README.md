# Graph Generation Chain for Neo4j Knowledge Graph

Harness the power of natural language understanding and convert plain text into structured knowledge graphs with the Graph Generation Chain.
This system integrates with the Neo4j graph database using OpenAI's LLM.
By leveraging OpenAI Functions capabilities, the Graph Generation Chain efficiently extracts graph structure from text.

## Set up Environment

You need to define the following environment variables

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Installation

To get started with the Graph Generation Chain:

```bash
# from inside your LangServe instance
poe add neo4j-generation
```