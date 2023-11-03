
# hyde

This template HyDE with RAG. 

Hyde is a retrieval method that stands for Hypothetical Document Embeddings (HyDE). It is a method used to enhance retrieval by generating a hypothetical document for an incoming query. 

The document is then embedded, and that embedding is utilized to look up real documents that are similar to the hypothetical document. 

The underlying concept is that the hypothetical document may be closer in the embedding space than the query. 

For a more detailed description, see the paper [here](https://arxiv.org/abs/2212.10496).

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package hyde
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add hyde
```

And add the following code to your `server.py` file:
```python
from hyde.chain import chain as hyde_chain

add_routes(app, hyde_chain, path="/hyde")
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
We can access the playground at [http://127.0.0.1:8000/hyde/playground](http://127.0.0.1:8000/hyde/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/hyde")
```

