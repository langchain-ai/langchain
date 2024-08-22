
# rag-ollama-multi-query

This template performs RAG using Ollama and OpenAI with a multi-query retriever. 

The multi-query retriever is an example of query transformation, generating multiple queries from different perspectives based on the user's input query. 

For each query, it retrieves a set of relevant documents and takes the unique union across all queries for answer synthesis.

We use a private, local LLM for the narrow task of query generation to avoid excessive calls to a larger LLM API.

See an example trace for Ollama LLM performing the query expansion [here](https://smith.langchain.com/public/8017d04d-2045-4089-b47f-f2d66393a999/r).

But we use OpenAI for the more challenging task of answer syntesis (full trace example [here](https://smith.langchain.com/public/ec75793b-645b-498d-b855-e8d85e1f6738/r)).

## Environment Setup

To set up the environment, you need to download Ollama. 

Follow the instructions [here](https://python.langchain.com/docs/integrations/chat/ollama). 

You can choose the desired LLM with Ollama. 

This template uses `zephyr`, which can be accessed using `ollama pull zephyr`.

There are many other options available [here](https://ollama.ai/library).

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first install the LangChain CLI:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this package, do:

```shell
langchain app new my-app --package rag-ollama-multi-query
```

To add this package to an existing project, run:

```shell
langchain app add rag-ollama-multi-query
```

And add the following code to your `server.py` file:

```python
from rag_ollama_multi_query import chain as rag_ollama_multi_query_chain

add_routes(app, rag_ollama_multi_query_chain, path="/rag-ollama-multi-query")
```

(Optional) Now, let's configure LangSmith. LangSmith will help us trace, monitor, and debug LangChain applications. You can sign up for LangSmith [here](https://smith.langchain.com/). If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server running locally at [http://localhost:8000](http://localhost:8000)

You can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
You can access the playground at [http://127.0.0.1:8000/rag-ollama-multi-query/playground](http://127.0.0.1:8000/rag-ollama-multi-query/playground)

To access the template from code, use:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-ollama-multi-query")
```