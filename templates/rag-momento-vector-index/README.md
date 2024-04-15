# rag-momento-vector-index

This template performs RAG using Momento Vector Index (MVI) and OpenAI.

> MVI: the most productive, easiest to use, serverless vector index for your data. To get started with MVI, simply sign up for an account. There's no need to handle infrastructure, manage servers, or be concerned about scaling. MVI is a service that scales automatically to meet your needs. Combine with other Momento services such as Momento Cache to cache prompts and as a session store or Momento Topics as a pub/sub system to broadcast events to your application.

To sign up and access MVI, visit the [Momento Console](https://console.gomomento.com/).

## Environment Setup

This template uses Momento Vector Index as a vectorstore and requires that `MOMENTO_API_KEY`, and `MOMENTO_INDEX_NAME` are set.

Go to the [console](https://console.gomomento.com/) to get an API key.

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-momento-vector-index
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-momento-vector-index
```

And add the following code to your `server.py` file:

```python
from rag_momento_vector_index import chain as rag_momento_vector_index_chain

add_routes(app, rag_momento_vector_index_chain, path="/rag-momento-vector-index")
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
We can access the playground at [http://127.0.0.1:8000/rag-momento-vector-index/playground](http://127.0.0.1:8000/rag-momento-vector-index/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-momento-vector-index")
```

## Indexing Data

We have included a sample module to index data. That is available at `rag_momento_vector_index/ingest.py`. You will see a commented out line in `chain.py` that invokes this. Uncomment to use.
