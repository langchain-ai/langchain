
# rag-clickhouse

This template performs RAG with no reliance on external APIs. 

It utilizes Ollama the LLM, GPT4All for embeddings, and ClickHouse for the vectorstore.

The vectorstore is created in `chain.py` and by default indexes a [popular blog posts on Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) for question-answering. 

## Environment Setup

To set up the environment, you need to download Ollama. 

Follow the instructions [here](https://python.langchain.com/docs/integrations/chat/ollama). 

You can choose the desired LLM with Ollama. 

This template uses `mistral`, which can be accessed using `ollama pull mistral`.

There are also [other models available](https://ollama.ai/library).

This package also uses [GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all) embeddings.

You'll also need to install ClickHouse:

```bash
curl https://clickhouse.com/ | sh
./clickhouse server
```

Once you've done that, connect to the server:

```bash
./clickhouse client -m
```

And then create the following table:

```sql
SET allow_experimental_object_type=1;
```

```sql
CREATE TABLE IF NOT EXISTS default.clickhouse_vector_search_example
(
    `id` Nullable(String),
    `document` Nullable(String),
    `embedding` Array(Float32),
    `metadata` JSON,
    `uuid` UUID DEFAULT generateUUIDv4(),
    CONSTRAINT cons_vec_len CHECK length(embedding) = 384
)
ENGINE = MergeTree
ORDER BY uuid;
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project (called `my-app`) and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-clickhouse
```

If you want to add this to an existing project, you can run:

```shell
langchain app add rag-clickhouse
```

And add the following code to your `server.py` file:
```python
from rag_clickhouse import chain as rag_clickhouse

add_routes(app, rag_clickhouse, path="/rag-clickhouse")
```

(Optional) Let's now configure LangSmith. LangSmith will help us trace, monitor and debug LangChain applications. 
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
We can access the playground at [http://127.0.0.1:8000/rag-clickhouse/playground](http://127.0.0.1:8000/rag_clickhouse/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-clickhouse")
```

The package will create and add documents to the vector database in `chain.py`. 
By default, it will load a popular blog post on agents. However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).
