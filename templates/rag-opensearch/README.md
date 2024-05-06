# rag-opensearch

This Template performs RAG using [OpenSearch](https://python.langchain.com/docs/integrations/vectorstores/opensearch).

## Environment Setup

Set the following environment variables. 

- `OPENAI_API_KEY` -  To access OpenAI Embeddings and Models.

And optionally set the OpenSearch ones if not using defaults:

- `OPENSEARCH_URL` - URL of the hosted OpenSearch Instance
- `OPENSEARCH_USERNAME` - User name for the OpenSearch instance
- `OPENSEARCH_PASSWORD` - Password for the OpenSearch instance
- `OPENSEARCH_INDEX_NAME` - Name of the index 

To run the default OpenSeach instance in docker, you can use the command
```shell
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:latest
```

Note: To load dummy index named `langchain-test` with dummy documents, run `python dummy_index_setup.py` in the package

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-opensearch
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-opensearch
```

And add the following code to your `server.py` file:
```python
from rag_opensearch import chain as rag_opensearch_chain

add_routes(app, rag_opensearch_chain, path="/rag-opensearch")
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
We can access the playground at [http://127.0.0.1:8000/rag-opensearch/playground](http://127.0.0.1:8000/rag-opensearch/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-opensearch")
```