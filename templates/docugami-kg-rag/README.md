
# docugami-kg-rag

This template contains a reference architecture for Retrieval Augmented Generation against a set of documents using Docugami's XML Knowledge Graph (KG-RAG).

## Environment Setup

You need to set some required environment variables before using your new app based on this template. These are used to index as well as run the application, and exceptions are raised if the following required environment variables are not set:

1. `OPENAI_API_KEY`: from the OpenAI platform.
2. `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`: from pinecone.io
3. `DOCUGAMI_API_KEY`: from the [Docugami Developer Playground](https://help.docugami.com/home/docugami-api)

```shell
export OPENAI_API_KEY=...

export PINECONE_API_KEY=...
export PINECONE_ENVIRONMENT=...

export DG_ENVIRONMENT=Production
export DOCUGAMI_API_KEY=...
```

Note that Pinecone only allows one index for your free starter project.

## Usage

### Indexing

Before you can run your app, you need to build your index in Pinecone.io. See [cli.py](./docugami_kg_rag/cli.py) which you can directly run via `poetry run index` after setting the environment variables as specified above. The CLI will query docsets in the workspace corresponding to your `DOCUGAMI_API_KEY` and let you pick which one(s) you want to index. When this is done, you can check to make your index is created and populated in Pinecone.io.

Indexing in this template uses the Docugami Loader for LangChain to create semantic chunks out of your documents. Refer to this [documentation](https://python.langchain.com/docs/integrations/document_loaders/docugami) for details.


### Creating app
To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package docugami-kg-rag
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add docugami-kg-rag
```

And add the following code to your `server.py` file:
```python
from docugami_kg_rag import chain as docugami_kg_rag_chain

add_routes(app, docugami_kg_rag, path="/docugami-kg-rag")
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

### Running app
If you are inside the app directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/docugami-kg-rag/playground](http://127.0.0.1:8000/docugami-kg-rag/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/docugami-kg-rag")
```
