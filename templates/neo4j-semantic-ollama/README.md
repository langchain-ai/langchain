# neo4j-semantic-ollama

This template is designed to implement an agent capable of interacting with a graph database like Neo4j through a semantic layer using Mixtral as a JSON-based agent.
The semantic layer equips the agent with a suite of robust tools, allowing it to interact with the graph database based on the user's intent.
Learn more about the semantic layer template in the [corresponding blog post](https://medium.com/towards-data-science/enhancing-interaction-between-language-models-and-graph-databases-via-a-semantic-layer-0a78ad3eba49) and specifically about [Mixtral agents with Ollama](https://blog.langchain.dev/json-based-agents-with-ollama-and-langchain/).

![Diagram illustrating the workflow of the Neo4j semantic layer with an agent interacting with tools like Information, Recommendation, and Memory, connected to a knowledge graph.](https://raw.githubusercontent.com/langchain-ai/langchain/master/templates/neo4j-semantic-ollama/static/workflow.png "Neo4j Semantic Layer Workflow Diagram")

## Tools

The agent utilizes several tools to interact with the Neo4j graph database effectively:

1. **Information tool**:
   - Retrieves data about movies or individuals, ensuring the agent has access to the latest and most relevant information.
2. **Recommendation Tool**:
   - Provides movie recommendations based upon user preferences and input.
3. **Memory Tool**:
   - Stores information about user preferences in the knowledge graph, allowing for a personalized experience over multiple interactions.
4. **Smalltalk Tool**:
   - Allows an agent to deal with smalltalk.

## Environment Setup

Before using this template, you need to set up Ollama and Neo4j database.

1. Follow instructions [here](https://python.langchain.com/docs/integrations/chat/ollama) to download Ollama.

2. Download your LLM of interest:

    * This package uses `mixtral`: `ollama pull mixtral`
    * You can choose from many LLMs [here](https://ollama.ai/library)

You need to define the following environment variables

```
OLLAMA_BASE_URL=<YOUR_OLLAMA_URL>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Populating with data

If you want to populate the DB with an example movie dataset, you can run `python ingest.py`.
The script import information about movies and their rating by users.
Additionally, the script creates two [fulltext indices](https://neo4j.com/docs/cypher-manual/current/indexes-for-full-text-search/), which are used to map information from user input to the database.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-semantic-ollama
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-semantic-ollama
```

And add the following code to your `server.py` file:
```python
from neo4j_semantic_layer import agent_executor as neo4j_semantic_agent

add_routes(app, neo4j_semantic_agent, path="/neo4j-semantic-ollama")
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
We can access the playground at [http://127.0.0.1:8000/neo4j-semantic-ollama/playground](http://127.0.0.1:8000/neo4j-semantic-ollama/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-semantic-ollama")
```
