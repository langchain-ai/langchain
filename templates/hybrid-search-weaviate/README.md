# Hybrid Search in Weaviate
This template shows you how to use the hybrid search feature in Weaviate. Hybrid search combines multiple search algorithms to improve the accuracy and relevance of search results. 

Weaviate uses both sparse and dense vectors to represent the meaning and context of search queries and documents. The results use a combination of `bm25` and vector search ranking to return the top results. 

##  Configurations
Connect to your hosted Weaviate Vectorstore by setting a few env variables in `chain.py`:

* `WEAVIATE_ENVIRONMENT`
* `WEAVIATE_API_KEY`

You will also need to set your `OPENAI_API_KEY` to use the OpenAI models.

## Get Started 
To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package hybrid-search-weaviate
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add hybrid-search-weaviate
```

And add the following code to your `server.py` file:
```python
from hybrid_search_weaviate import chain as hybrid_search_weaviate_chain

add_routes(app, hybrid_search_weaviate_chain, path="/hybrid-search-weaviate")
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
We can access the playground at [http://127.0.0.1:8000/hybrid-search-weaviate/playground](http://127.0.0.1:8000/hybrid-search-weaviate/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/hybrid-search-weaviate")
```
