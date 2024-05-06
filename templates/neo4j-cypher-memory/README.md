
# neo4j-cypher-memory

This template allows you to have conversations with a Neo4j graph database in natural language, using an OpenAI LLM.
It transforms a natural language question into a Cypher query (used to fetch data from Neo4j databases), executes the query, and provides a natural language response based on the query results.
Additionally, it features a conversational memory module that stores the dialogue history in the Neo4j graph database.
The conversation memory is uniquely maintained for each user session, ensuring personalized interactions.
To facilitate this, please supply both the `user_id` and `session_id` when using the conversation chain.

![Workflow diagram illustrating the process of a user asking a question, generating a Cypher query, retrieving conversational history, executing the query on a Neo4j database, generating an answer, and storing conversational memory.](https://raw.githubusercontent.com/langchain-ai/langchain/master/templates/neo4j-cypher-memory/static/workflow.png "Neo4j Cypher Memory Workflow Diagram")

## Environment Setup

Define the following environment variables:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Neo4j database setup

There are a number of ways to set up a Neo4j database.

### Neo4j Aura

Neo4j AuraDB is a fully managed cloud graph database service.
Create a free instance on [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database?utm_source=langchain&utm_content=langserve).
When you initiate a free database instance, you'll receive credentials to access the database.

## Populating with data

If you want to populate the DB with some example data, you can run `python ingest.py`.
This script will populate the database with sample movie data.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-cypher-memory
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-cypher-memory
```

And add the following code to your `server.py` file:
```python
from neo4j_cypher_memory import chain as neo4j_cypher_memory_chain

add_routes(app, neo4j_cypher_memory_chain, path="/neo4j-cypher-memory")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/neo4j_cypher_memory/playground](http://127.0.0.1:8000/neo4j_cypher/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-cypher-memory")
```
