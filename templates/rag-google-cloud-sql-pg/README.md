# RAG - Google Cloud SQL for PostgreSQL

This template performs RAG using `Cloud SQL for PostgreSQL` and `VertexAI`.

## Environment Setup

Use the following templates to deploy Retrieval Augmented Generation (RAG) applications with an CloudSQL for Postgres database.

1. In the Google Cloud console, on the project selector page, select or [create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
2. [Make sure that billing is enabled for your Google Cloud project](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console).
3. [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets).
4.  Enable [AI Platform, Cloud SQL Admin, and Service Networking APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,sqladmin.googleapis.com,servicenetworking.googleapis.com&_ga=2.92928541.1293093187.1719511698-1945987529.1719351858)
5. [Create a Cloud SQL for Postgres instance.](https://cloud.google.com/sql/docs/postgres/create-instance)
6. [Configure Public IP.](https://cloud.google.com/sql/docs/postgres/configure-ip)
7. [Create a database.](https://cloud.google.com/sql/docs/postgres/create-manage-databases)
8. Create a [vector store table](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/vector_store.ipynb).
9. Grant IAM permissions, `roles/cloudsql.client`, `roles/aiplatform.user`, and `serviceusage.serviceUsageConsumer` to the AI Platform Reasoning Engine Service Agent service account: `service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com` to connect to the Cloud SQL instance.
10. (Optional) [Add an IAM user or service account to a database instance](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users) and
[grant database privileges to the IAM user](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users#grant-db-privileges).
11. Use [`create_embeddings.py`](create_embeddings.py) to add data to your vector store.

Set these environments to run the template
  * `DATABASE_ID`
  * `INSTANCE_ID`
  * `DB_PASSWORD`
  * `PROJECT_ID`
  * `REGION`
  * `TABLE_NAME`
  * `DB_USER`

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