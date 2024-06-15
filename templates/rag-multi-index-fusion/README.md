# RAG with Multiple Indexes (Fusion)

A QA application that queries multiple domain-specific retrievers and selects the most relevant documents from across all retrieved results.

## Environment Setup

This application queries PubMed, ArXiv, Wikipedia, and [Kay AI](https://www.kay.ai) (for SEC filings).

You will need to create a free Kay AI account and [get your API key here](https://www.kay.ai).
Then set environment variable:

```bash
export KAY_API_KEY="<YOUR_API_KEY>"
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-multi-index-fusion
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-multi-index-fusion
```

And add the following code to your `server.py` file:
```python
from rag_multi_index_fusion import chain as rag_multi_index_fusion_chain

add_routes(app, rag_multi_index_fusion_chain, path="/rag-multi-index-fusion")
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
We can access the playground at [http://127.0.0.1:8000/rag-multi-index-fusion/playground](http://127.0.0.1:8000/rag-multi-index-fusion/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-multi-index-fusion")
```
