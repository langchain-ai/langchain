# rag-aerospike

This template performs RAG using Aerospike Vector Search (AVS), HuggingFace embeddings, and an OpenAI LLM. The data set is a draft of the [Aerospike Up and Running book](https://aerospike.com/files/ebooks/aerospike-up-and-running-early-release3.pdf) which is loaded, tokenized, then embedded using the all-MiniLM-L6-v2 sentence transformer. The context and embeddings are stored in the Aerospike Vector Search LangChain vector store.

The chain exposed in this example shows basic usage of the Aerospike Vector Search LangChain vector store as a retriever for RAG applications.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the [OpenAI](https://platform.openai.com) models:
Set `AVS_HOST` (default: localhost) and `AVS_PORT` (default: 5000) to the address for your AVS deployment.
Set `AVS_NAMESPACE` (default: test) to the Aerospike namespace to store vector data and indexes in.
Set `DATASOURCE` (default: https://aerospike.com/files/ebooks/aerospike-up-and-running-early-release3.pdf) to a URL or file path of a PDF you would like to index. The text from the PDF will be used as context in the RAG application.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-aerospike
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-aerospike
```

And add the following code to your `server.py` file:
```python
from rag_aerospike import chain as rag_aerospike_chain

add_routes(app, rag_aerospike_chain, path="/rag-aerospike")
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
We can access the playground at [http://127.0.0.1:8000/rag-aerospike/playground](http://127.0.0.1:8000/rag-aerospike/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-aerospike")
```