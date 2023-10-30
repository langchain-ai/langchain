# Parent Document Retriever with Neo4j Vector Index

This template allows you to balance precise embeddings and context retention by splitting documents into smaller chunks and retrieving their original or larger text information.
Using a Neo4j vector index, the template queries child nodes using vector similarity search and retrieves the corresponding parent's text by defining an appropriate `retrieval_query` parameter.

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
The script process and stores sections of the text from the file `dune.txt` into a Neo4j graph database.
First, the text is divided into larger chunks ("parents") and then further subdivided into smaller chunks ("children"), where both parent and child chunks overlap slightly to maintain context.
After storing these chunks in the database, embeddings for the child nodes are computed using OpenAI's embeddings and stored back in the graph for future retrieval or analysis.
Additionally, a vector index named `retrieval` is created for efficient querying of these embeddings.

## Installation

```bash
# from inside your LangServe instance
poe add neo4j-parent
```
