# rag-conversation-zep

This template demonstrates building a RAG conversation app using Zep. 

Included in this template:
- Populating a [Zep Document Collection](https://docs.getzep.com/sdk/documents/) with a set of documents (a Collection is analogous to an index in other Vector Databases).
- Using Zep's [integrated embedding](https://docs.getzep.com/deployment/embeddings/) functionality to embed the documents as vectors.
- Configuring a LangChain [ZepVectorStore Retriever](https://docs.getzep.com/sdk/documents/) to retrieve documents using Zep's built, hardware accelerated in [Maximal Marginal Relevance](https://docs.getzep.com/sdk/search_query/) (MMR) re-ranking.
- Prompts, a simple chat history data structure, and other components required to build a RAG conversation app.
- The RAG conversation chain.

## About [Zep - Fast, scalable building blocks for LLM Apps](https://www.getzep.com/)
Zep is an open source platform for productionizing LLM apps. Go from a prototype built in LangChain or LlamaIndex, or a custom app, to production in minutes without rewriting code.

Key Features:

- Fast! Zep’s async extractors operate independently of the your chat loop, ensuring a snappy user experience.
- Long-term memory persistence, with access to historical messages irrespective of your summarization strategy.
- Auto-summarization of memory messages based on a configurable message window. A series of summaries are stored, providing flexibility for future summarization strategies.
- Hybrid search over memories and metadata, with messages automatically embedded on creation.
- Entity Extractor that automatically extracts named entities from messages and stores them in the message metadata.
- Auto-token counting of memories and summaries, allowing finer-grained control over prompt assembly.
- Python and JavaScript SDKs.

Zep project: https://github.com/getzep/zep | Docs: https://docs.getzep.com/

## Environment Setup

Set up a Zep service by following the [Quick Start Guide](https://docs.getzep.com/deployment/quickstart/).

## Ingesting Documents into a Zep Collection

Run `python ingest.py` to ingest the test documents into a Zep Collection. Review the file to modify the Collection name and document source.


## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-conversation-zep
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-conversation-zep
```

And add the following code to your `server.py` file:
```python
from rag_conversation_zep import chain as rag_conversation_zep_chain

add_routes(app, rag_conversation_zep_chain, path="/rag-conversation-zep")
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
We can access the playground at [http://127.0.0.1:8000/rag-conversation-zep/playground](http://127.0.0.1:8000/rag-conversation-zep/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-conversation-zep")
```