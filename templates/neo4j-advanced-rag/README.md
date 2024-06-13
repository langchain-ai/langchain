# neo4j-advanced-rag

This template allows you to balance precise embeddings and context retention by implementing advanced retrieval strategies.

## Strategies

1. **Typical RAG**:
   - Traditional method where the exact data indexed is the data retrieved.
2. **Parent retriever**:
   - Instead of indexing entire documents, data is divided into smaller chunks, referred to as Parent and Child documents.
   - Child documents are indexed for better representation of specific concepts, while parent documents is retrieved to ensure context retention.
3. **Hypothetical Questions**:
     - Documents are processed to determine potential questions they might answer.
     - These questions are then indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
4. **Summaries**:
     - Instead of indexing the entire document, a summary of the document is created and indexed.
     - Similarly, the parent document is retrieved in a RAG application.

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
For every parent node, hypothetical questions and summaries are generated, embedded, and added to the database. 
Additionally, a vector index for each retrieval strategy is created for efficient querying of these embeddings.

*Note that ingestion can take a minute or two due to LLMs velocity of generating hypothetical questions and summaries.*

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package neo4j-advanced-rag
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add neo4j-advanced-rag
```

And add the following code to your `server.py` file:
```python
from neo4j_advanced_rag import chain as neo4j_advanced_chain

add_routes(app, neo4j_advanced_chain, path="/neo4j-advanced-rag")
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
We can access the playground at [http://127.0.0.1:8000/neo4j-advanced-rag/playground](http://127.0.0.1:8000/neo4j-advanced-rag/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/neo4j-advanced-rag")
```
