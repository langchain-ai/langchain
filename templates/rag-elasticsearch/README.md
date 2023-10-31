
# rag-elasticsearch

This template performs RAG using ElasticSearch.

It relies on sentence transformer `MiniLM-L6-v2` for embedding passages and questions.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

To connect to your Elasticsearch instance, use the following environment variables:

```bash
export ELASTIC_CLOUD_ID = <ClOUD_ID>
export ELASTIC_USERNAME = <ClOUD_USERNAME>
export ELASTIC_PASSWORD = <ClOUD_PASSWORD>
```
For local development with Docker, use:

```bash
export ES_URL = "http://localhost:9200"
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-elasticsearch
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-elasticsearch
```

And add the following code to your `server.py` file:
```python
from rag_elasticsearch import chain as rag_elasticsearch_chain

add_routes(app, rag_elasticsearch_chain, path="/rag-elasticsearch")
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
We can access the playground at [http://127.0.0.1:8000/rag-elasticsearch/playground](http://127.0.0.1:8000/rag-elasticsearch/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-elasticsearch")
```

For loading the fictional workplace documents, run the following command from the root of this repository:

```bash
python ./data/load_documents.py
```

However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).  
