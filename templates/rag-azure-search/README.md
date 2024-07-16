# rag-azure-search

This template performs RAG on documents using [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search) as the vectorstore and Azure OpenAI chat and embedding models.

For additional details on RAG with Azure AI Search, refer to [this notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/azuresearch.ipynb).


## Environment Setup

***Prerequisites:*** Existing [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search) and [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/overview) resources.

***Environment Variables:***

To run this template, you'll need to set the following environment variables:

***Required:***

- AZURE_SEARCH_ENDPOINT - The endpoint of the Azure AI Search service.
- AZURE_SEARCH_KEY - The API key for the Azure AI Search service.
- AZURE_OPENAI_ENDPOINT - The endpoint of the Azure OpenAI service.
- AZURE_OPENAI_API_KEY - The API key for the Azure OpenAI service.
- AZURE_EMBEDDINGS_DEPLOYMENT - Name of the Azure OpenAI deployment to use for embeddings.
- AZURE_CHAT_DEPLOYMENT - Name of the Azure OpenAI deployment to use for chat.

***Optional:***

- AZURE_SEARCH_INDEX_NAME - Name of an existing Azure AI Search index to use. If not provided, an index will be created with name "rag-azure-search".
- OPENAI_API_VERSION - Azure OpenAI API version to use. Defaults to "2023-05-15". 

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-azure-search
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-azure-search
```

And add the following code to your `server.py` file:
```python
from rag_azure_search import chain as rag_azure_search_chain

add_routes(app, rag_azure_search_chain, path="/rag-azure-search")
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
We can access the playground at [http://127.0.0.1:8000/rag-azure-search/playground](http://127.0.0.1:8000/rag-azure-search/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-azure-search")
```