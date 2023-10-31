# Neo4j Knowledge Graph: Enhanced mapping from text to database using a full-text index

This template allows you to chat with Neo4j graph database in natural language, using an OpenAI LLM.
Its primary purpose is to convert a natural language question into a Cypher query (which is used to query Neo4j databases), 
execute the query, and then provide a natural language response based on the query's results.
The addition of the full-text index ensures efficient mapping of values from text to database for more precise Cypher statement generation.
In this example, full-text index is used to map names of people and movies from the user's query with corresponding database entries.

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
Additionally, it will create an full-text index named `entity`, which is used to 
map person and movies from user input to database values for precise Cypher statement generation.

## Installation

```bash
# from inside your LangServe instance
poe add neo4j-cypher-ft
```
