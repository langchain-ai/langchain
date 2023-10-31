# neo4j-generation

The neo4j-generation template is designed to convert plain text into structured knowledge graphs. 

By using OpenAI's language model, it can efficiently extract structured information from text and construct a knowledge graph in Neo4j. 

This package is flexible and allows users to guide the extraction process by specifying a list of node labels and relationship types.

For more details on the functionality and capabilities of this package, please refer to [this blog post](https://blog.langchain.dev/constructing-knowledge-graphs-from-text-using-openai-functions/).

## Environment Setup

You need to set the following environment variables:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-generation
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-generation
```

And add the following code to your `server.py` file:
```python
from neo4j_generation import chain as neo4j_generation_chain

add_routes(app, neo4j_generation_chain, path="/neo4j-generation")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
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
We can access the playground at [http://127.0.0.1:8000/neo4j-generation/playground](http://127.0.0.1:8000/neo4j-generation/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-generation")
```
