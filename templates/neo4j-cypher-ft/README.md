
# neo4j-cypher-ft

This template allows you to interact with a Neo4j graph database using natural language, leveraging OpenAI's LLM. 

Its main function is to convert natural language questions into Cypher queries (the language used to query Neo4j databases), execute these queries, and provide natural language responses based on the query's results. 

The package utilizes a full-text index for efficient mapping of text values to database entries, thereby enhancing the generation of accurate Cypher statements. 

In the provided example, the full-text index is used to map names of people and movies from the user's query to corresponding database entries.

![Workflow diagram showing the process from a user asking a question to generating an answer using the Neo4j knowledge graph and full-text index.](https://raw.githubusercontent.com/langchain-ai/langchain/master/templates/neo4j-cypher-ft/static/workflow.png "Neo4j Cypher Workflow Diagram")

## Environment Setup

The following environment variables need to be set:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

Additionally, if you wish to populate the DB with some example data, you can run `python ingest.py`.
This script will populate the database with sample movie data and create a full-text index named `entity`, which is used to map person and movies from user input to database values for precise Cypher statement generation.


## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-cypher-ft
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-cypher-ft
```

And add the following code to your `server.py` file:
```python
from neo4j_cypher_ft import chain as neo4j_cypher_ft_chain

add_routes(app, neo4j_cypher_ft_chain, path="/neo4j-cypher-ft")
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

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/neo4j-cypher-ft/playground](http://127.0.0.1:8000/neo4j-cypher-ft/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-cypher-ft")
```
