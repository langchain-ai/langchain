# sql-pgvector

This template enables user to use `pgvector` for combining postgreSQL with semantic search / RAG. 

It uses [PGVector](https://github.com/pgvector/pgvector) extension as shown in the [RAG empowered SQL cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb)

## Environment Setup

If you are using `ChatOpenAI` as your LLM, make sure the `OPENAI_API_KEY` is set in your environment. You can change both the LLM and embeddings model inside `chain.py`

And you can configure configure the following environment variables
for use by the template (defaults are in parentheses)

- `POSTGRES_USER` (postgres)
- `POSTGRES_PASSWORD` (test)
- `POSTGRES_DB` (vectordb)
- `POSTGRES_HOST` (localhost)
- `POSTGRES_PORT` (5432)

If you don't have a postgres instance, you can run one locally in docker:

```bash
docker run \
  --name some-postgres \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  postgres:16
```

And to start again later, use the `--name` defined above:
```bash
docker start some-postgres
```

### PostgreSQL Database setup

Apart from having `pgvector` extension enabled, you will need to do some setup before being able to run semantic search within your SQL queries.

In order to run RAG over your postgreSQL database you will need to generate the embeddings for the specific columns you want. 

This process is covered in the [RAG empowered SQL cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb), but the overall approach consist of:
1. Querying for unique values in the column
2. Generating embeddings for those values
3. Store the embeddings in a separate column or in an auxiliary table.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package sql-pgvector
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add sql-pgvector
```

And add the following code to your `server.py` file:
```python
from sql_pgvector import chain as sql_pgvector_chain

add_routes(app, sql_pgvector_chain, path="/sql-pgvector")
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
We can access the playground at [http://127.0.0.1:8000/sql-pgvector/playground](http://127.0.0.1:8000/sql-pgvector/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/sql-pgvector")
```
