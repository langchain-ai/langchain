
# rag-biomedical

This template performs RAG over clinical trial data from [Clinical Trials .gov](https://classic.clinicaltrials.gov/api/gui/ref/download_all).

It builds a vectorstore from a sub-set of cinical trial data using `build_db.ipynb` with specified metadata fields.

It then uses [self-query retriever](https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query) to query using these metadata filters.

## Database

The vectorstore is created using `build_db.ipynb`. 

Also see more general context on biomedical RAG in `biomedical_rag_introduction.ipynb`.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-biomedical
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-biomedical
```

And add the following code to your `server.py` file:
```python
from rag_biomedical import chain as rag_biomedical_chain

add_routes(app, rag_biomedical_chain, path="/rag-biomedical")
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
We can access the playground at [http://127.0.0.1:8000/rag-biomedical/playground](http://127.0.0.1:8000/rag-biomedical/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-biomedical")
```