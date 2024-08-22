
# rag-matching-engine

This template performs RAG using Google Cloud Platform's Vertex AI with the matching engine.

It will utilize a previously created index to retrieve relevant documents or contexts based on user-provided questions. 

## Environment Setup

An index should be created before running the code. 

The process to create this index can be found [here](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-qa/question_answering_documents_langchain_matching_engine.ipynb).

Environment variables for Vertex should be set:
```
PROJECT_ID
ME_REGION
GCS_BUCKET
ME_INDEX_ID
ME_ENDPOINT_ID
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-matching-engine
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-matching-engine
```

And add the following code to your `server.py` file:
```python
from rag_matching_engine import chain as rag_matching_engine_chain

add_routes(app, rag_matching_engine_chain, path="/rag-matching-engine")
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
We can access the playground at [http://127.0.0.1:8000/rag-matching-engine/playground](http://127.0.0.1:8000/rag-matching-engine/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-matching-engine")
```

For more details on how to connect to the template, refer to the Jupyter notebook `rag_matching_engine`.