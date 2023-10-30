# Graph Generation Chain for Neo4j Knowledge Graph

Harness the power of natural language understanding of LLMs and convert plain text into structured knowledge graphs with the Graph Generation Chain.
This chain uses OpenAI's LLM to construct a knowledge graph in Neo4j.
Leveraging OpenAI Functions capabilities, the Graph Generation Chain efficiently extracts structured information from text.
The chain has the following input parameters:

* text (str): The input text from which the information will be extracted to construct the graph.
* allowed_nodes (Optional[List[str]]): A list of node labels to guide the extraction process.
                                If not provided, extraction won't have specific restriction on node labels.
* allowed_relationships (Optional[List[str]]): A list of relationship types to guide the extraction process.
                                If not provided, extraction won't have specific restriction on relationship types.

Find more details in [this blog post](https://blog.langchain.dev/constructing-knowledge-graphs-from-text-using-openai-functions/).

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

## Installation

To get started with the Graph Generation Chain:

```bash
# from inside your LangServe instance
poe add neo4j-generation
```