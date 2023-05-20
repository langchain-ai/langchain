# Databricks

This page covers how to use the [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) within LangChain.
It is broken into two parts: installation and setup, and then references to specific SQLDatabase wrappers.

## Installation and Setup
- Install the Databricks SQL Connector for Python with `pip install databricks-sql-connector`

## Wrappers

### SQLDatabase
You can connect to [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) using the `SQLDatabase` wrapper.

To import `SQLDatabase`:
```python
from langchain import SQLDatabase
```

To connect to a Databricks SQL warehouse or a Databricks cluster:
```python
SQLDatabase.from_databricks(
    catalog: str,
    schema: str,
    host: Optional[str] = None,
    api_token: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    engine_args: Optional[dict] = None,
    **kwargs: Any)
```
#### Required Parameters
* `catalog`: The catalog name in the Databricks database.
* `schema`: The schema name in the catalog.

#### Optional Parameters
There following parameters are optional. When executing the method in a Databricks notebook, you don't need to provide them in most of the cases.
* `host`: The Databricks workspace hostname, excluding
    `https://` part. Defaults to 'DATABRICKS_HOST' environment variable or current workspace if in a Databricks notebook.
* `api_token`: The Databricks personal access token for
    accessing the Databricks SQL warehouse or the cluster. Defaults to 'DATABRICKS_API_TOKEN' environment variable or a temporary one is generated if in a Databricks notebook.
* `warehouse_id`: The warehouse ID in the Databricks SQL.
* `cluster_id`: The cluster ID in the Databricks Runtime. If running in a Databricks notebook
    and both 'warehouse_id' and 'cluster_id' are None, it uses the ID of the
    cluster the notebook is attached to.
* `engine_args`: The arguments to be used when connecting
    Databricks.
* `**kwargs`: Additional keyword arguments for the `SQLDatabase.from_uri` method.

### Examples
For a more detailed walkthrough of the `SQLDatabase` wrapper, see 
