
# rag-redis

This template performs RAG using Redis and OpenAI on financial 10k filings docs (for Nike). 

It relies on the sentence transformer `all-MiniLM-L6-v2` for embedding chunks of the pdf and user questions.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

The following Redis environment variables need to be set:

```bash
export REDIS_HOST = <YOUR REDIS HOST>
export REDIS_PORT = <YOUR REDIS PORT>
export REDIS_USER = <YOUR REDIS USER NAME>
export REDIS_PASSWORD = <YOUR REDIS PASSWORD>
```

## Supported Settings
We use a variety of environment variables to configure this application

| Environment Variable | Description                       | Default Value |
|----------------------|-----------------------------------|---------------|
| `DEBUG`            | Enable or disable Langchain debugging logs       | True         |
| `REDIS_HOST`           | Hostname for the Redis server     | "localhost"   |
| `REDIS_PORT`           | Port for the Redis server         | 6379          |
| `REDIS_USER`           | User for the Redis server         | "" |
| `REDIS_PASSWORD`       | Password for the Redis server     | "" |
| `REDIS_URL`            | Full URL for connecting to Redis  | `None`, Constructed from user, password, host, and port if not provided |
| `INDEX_NAME`           | Name of the vector index          | "rag-redis"   |

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-redis
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-redis
```

And add the following code to your `server.py` file:
```python
from rag_redis.chain import chain as rag_redis_chain

add_routes(app, rag_redis_chain, path="/rag-redis")
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
We can access the playground at [http://127.0.0.1:8000/rag-redis/playground](http://127.0.0.1:8000/rag-redis/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-redis")
```