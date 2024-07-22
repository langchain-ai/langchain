
# neo4j-parent

This template allows you to balance precise embeddings and context retention by splitting documents into smaller chunks and retrieving their original or larger text information. 

Using a Neo4j vector index, the package queries child nodes using vector similarity search and retrieves the corresponding parent's text by defining an appropriate `retrieval_query` parameter.

## Environment Setup

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


## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-parent
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-parent
```

And add the following code to your `server.py` file:
```python
from neo4j_parent import chain as neo4j_parent_chain

add_routes(app, neo4j_parent_chain, path="/neo4j-parent")
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
We can access the playground at [http://127.0.0.1:8000/neo4j-parent/playground](http://127.0.0.1:8000/neo4j-parent/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-parent")
```
