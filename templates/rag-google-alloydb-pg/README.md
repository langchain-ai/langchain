# RAG - Google AlloyDB for PostgreSQL

This template performs RAG using `AlloyDB for PostgreSQL` and `VertexAI`.
Learn more about the methods used in this template from [AlloyDB for PostgreSQL for LangChain](https://github.com/googleapis/langchain-google-alloydb-pg-python).

## Environment Setup

To run this template, you will need to setup an AlloyDB instance and store vectors into the database. Learn more about initializing an `AlloyDBVectorStore` from the [Google AlloyDB Vector Store Getting Started](https://github.com/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/vector_store.ipynb).

* Set these environments to run the template:
    * `PROJECT_ID`
    * `REGION`
    * `CLUSTER_ID`
    * `INSTANCE_ID`
    * `DATABASE_ID`
    * `TABLE_NAME`
    * `DB_USER`
    * `DB_PASSWORD`

* Enable the [Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

* This template uses public IP address to connect to the AlloyDB database. A public IP address is required for local testing, but not required when deployed into a Google Cloud VPC network. Learn how to customize the IP address type for the [`AlloyDBEngine`](https://cloud.google.com/python/docs/reference/langchain-google-alloydb-pg/latest/langchain_google_alloydb_pg.engine.AlloyDBEngine).

* This template uses [built-in database authentication](https://cloud.google.com/alloydb/docs/database-users/about) via a username and a password to quickly authenticate local database users. It is recommend to use IAM database authentication via local user or service account credentials. Learn how to customize the service account for the [`AlloyDBEngine`](https://cloud.google.com/python/docs/reference/langchain-google-alloydb-pg/latest/langchain_google_alloydb_pg.engine.AlloyDBEngine). Next, add this account as a new database user and grant privileges:
  * [Add an IAM user or service account to a cluster](https://cloud.google.com/alloydb/docs/manage-iam-authn#create-user).
  * [Grant appropriate database permissions to IAM users](https://cloud.google.com/alloydb/docs/manage-iam-authn#grant-privileges).

* To run this template locally, make sure you have set up [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) in your environment.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-google-alloydb-pg
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-google-alloydb-pg
```

And add the following code to your `server.py` file:

```python
from rag_google_alloydb_pg.chain import chain as rag_google_alloydb_pg_chain

add_routes(app, rag_google_alloydb_pg_chain, path="/rag-google-alloydb-pg")
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
at [http://127.0.0.1:8000/rag-google-alloydb-pg/playground](http://127.0.0.1:8000/rag-google-alloydb-pg/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-google-alloydb-pg")