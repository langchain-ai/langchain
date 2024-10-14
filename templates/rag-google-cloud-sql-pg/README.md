# RAG - Google Cloud SQL for PostgreSQL

This template performs RAG using `Cloud SQL for PostgreSQL` and `VertexAI`.
Learn more about the methods used in this template from [Cloud SQL for PostgreSQL for LangChain](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/README.rst)

## Environment Setup

To run this template, you will need to setup a Cloud SQL instance and store vectors into a database. Learn more about initializing an `PostgresVectorStore` from the [Google Cloud SQL Vector Store Getting Started](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/vector_store.ipynb)

Set these environments to run the template
  * `PROJECT_ID`
  * `REGION`
  * `INSTANCE_ID`
  * `DATABASE_ID`
  * `TABLE_NAME`
  * `DB_USER`
  * `DB_PASSWORD`

This uses public IP is required for local testing but not required when deployed to Google Cloud VPC network.

This template needs a user and password to access Postgres. Use these to create a new user and grant DB priviledges.
[Add an IAM user or service account to a database instance](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users) and
[grant database privileges to the IAM user](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users#grant-db-privileges).

Make sure you have authorized access for Google Cloud with your credentials. For more information on this, take a look at [Gcloud Auth Login](https://cloud.google.com/sdk/gcloud/reference/auth/login)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-google-cloud-sql-pg
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-google-cloud-sql-pg
```

And add the following code to your `server.py` file:

```python
from rag_google_cloud_sql_pg.chain import chain as rag_google_cloud_sql_pg_chain

add_routes(app, rag_google_cloud_sql_pg_chain, path="/rag-google-cloud-sql-pg")
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
at [http://127.0.0.1:8000/rag-google-cloud-sql-pg/playground](http://127.0.0.1:8000/rag-google-cloud-sql-pg/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-google-cloud-sql-pg")