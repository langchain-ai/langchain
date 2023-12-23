# rag-pathway

This template performs RAG using Pathway and OpenAI.

Pathway is a data streaming tool that can be utilized as vector store that is always up to date.
Once the engine is listening to your local files, Drive, Dropbox, s3 or any other connector, 
You don't have to run any ETL jobs, embedding services or LLM Api.

Your answers to questions will be always up to date with latest changes.
You can listen to many sources simultaneously. 

You can also use Pathway with the [self-query retrieval technique](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query).

## Environment Setup

Set the OPENAI_API_KEY environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-pathway
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-pathway
```

And add the following code to your `server.py` file:
```python
from rag_pathway import chain as rag_pathway_chain

add_routes(app, rag_pathway_chain, path="/rag-pathway")
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
We can access the playground at [http://127.0.0.1:8000/rag-pathway/playground](http://127.0.0.1:8000/rag-pathway/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-pathway")
```
