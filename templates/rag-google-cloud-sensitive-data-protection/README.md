# rag-google-cloud-sensitive-data-protection

This template is an application that utilizes Google Vertex AI Search, a machine learning powered search service, and
PaLM 2 for Chat (chat-bison). The application uses a Retrieval chain to answer questions based on your documents.

This template is an application that utilizes Google Sensitive Data Protection, a service for detecting and redacting
sensitive data in text, and PaLM 2 for Chat (chat-bison), although you can use any model.

For more context on using Sensitive Data Protection,
check [here](https://cloud.google.com/dlp/docs/sensitive-data-protection-overview).

## Environment Setup

Before using this template, please ensure that you enable the [DLP API](https://console.cloud.google.com/marketplace/product/google/dlp.googleapis.com)
and [Vertex AI API](https://console.cloud.google.com/marketplace/product/google/aiplatform.googleapis.com) in your Google Cloud
project.

For some common environment troubleshooting steps related to Google Cloud, see the bottom
of this readme.

Set the following environment variables:

* `GOOGLE_CLOUD_PROJECT_ID` - Your Google Cloud project ID.
* `MODEL_TYPE` - The model type for Vertex AI Search (e.g. `chat-bison`)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-google-cloud-sensitive-data-protection
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-google-cloud-sensitive-data-protection
```

And add the following code to your `server.py` file:

```python
from rag_google_cloud_sensitive_data_protection.chain import chain as rag_google_cloud_sensitive_data_protection_chain

add_routes(app, rag_google_cloud_sensitive_data_protection_chain, path="/rag-google-cloud-sensitive-data-protection")
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

runnable = RemoteRunnable("http://localhost:8000/rag-google-cloud-sensitive-data-protection")
```
```

# Troubleshooting Google Cloud

You can set your `gcloud` credentials with their CLI using `gcloud auth application-default login`

You can set your `gcloud` project with the following commands
```bash
gcloud config set project <your project>
gcloud auth application-default set-quota-project <your project>
export GOOGLE_CLOUD_PROJECT_ID=<your project>
```
