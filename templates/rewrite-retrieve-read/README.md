
# rewrite_retrieve_read

**rewrite_retrieve_read** is a method proposed in the paper [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2305.14283.pdf). It aims to optimize queries for Large Language Models (LLMs), especially in real-world scenarios, by prompting an LLM to rewrite the queries before conducting retrieval-augmented reading. This package demonstrates how you can easily implement this method using the LangChain Expression Language.

## Environment Setup

TODO: What environment variables need to be set (if any)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rewrite_retrieve_read
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rewrite_retrieve_read
```

And add the following code to your `server.py` file:
```python
from rewrite_retrieve_read.chain import chain as rewrite_retrieve_read.chain_chain

add_routes(app, rewrite_retrieve_read.chain_chain, path="/rewrite-retrieve-read")
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
We can access the playground at [http://127.0.0.1:8000/rewrite_retrieve_read/playground](http://127.0.0.1:8000/rewrite_retrieve_read/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rewrite_retrieve_read")
```
