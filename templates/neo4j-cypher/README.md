# Neo4j Knowledge Graph with OpenAI LLMs

This template allows you to chat with Neo4j graph database in natural language, using an OpenAI LLM.
Its primary purpose is to convert a natural language question into a Cypher query (which is used to query Neo4j databases), 
execute the query, and then provide a natural language response based on the query's results.

## Neo4j database

There are a number of ways to set up a Neo4j database.

### Neo4j Aura

Neo4j AuraDB is a fully managed cloud graph database service.
Create a free instance on [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database?utm_source=langchain&utm_content=langserve).
When you initiate a free database instance, you'll receive credentials to access the database.

##  Environment variables

You need to define the following environment variables

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Populating with data

If you want to populate the DB with some example data, you can run `python ingest.py`.
This script will populate the database with sample movie data.

## Installation

```bash
# from inside your LangServe instance
poe add neo4j-cypher
```
