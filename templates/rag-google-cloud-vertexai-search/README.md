# rag-google-cloud-vertexai-search

This template is an application that utilizes Google Vertex AI Search, a machine learning powered search service, and
PaLM 2 for Chat (chat-bison). The application uses a Retrieval chain to answer questions based on your documents.

For more context on building RAG applications with Vertex AI Search,
check [here](https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction).

## Environment Setup

Before using this template, please ensure that you are authenticated with Vertex AI Search. See the authentication
guide: [here](https://cloud.google.com/generative-ai-app-builder/docs/authentication).

You will also need to create:

- A search application [here](https://cloud.google.com/generative-ai-app-builder/docs/create-engine-es)
- A data store [here](https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es)

A suitable dataset to test this template with is the Alphabet Earnings Reports, which you can
find [here](https://abc.xyz/investor/). The data is also available
at `gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs`.

Set the following environment variables:

* `GOOGLE_CLOUD_PROJECT_ID` - Your Google Cloud project ID.
* `DATA_STORE_ID` - The ID of the data store in Vertex AI Search, which is a 36-character alphanumeric value found on
  the data store details page.
* `MODEL_TYPE` - The model type for Vertex AI Search.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-google-cloud-vertexai-search
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-google-cloud-vertexai-search
```

And add the following code to your `server.py` file:

```python
from rag_google_cloud_vertexai_search.chain import chain as rag_google_cloud_vertexai_search_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")
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

This will start the FastAPI app with a server running locally at
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground
at [http://127.0.0.1:8000/rag-google-cloud-vertexai-search/playground](http://127.0.0.1:8000/rag-google-cloud-vertexai-search/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-google-cloud-vertexai-search")
```
