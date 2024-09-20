# langchain-snowflake

This package contains the LangChain integration with Snowflake

## Installation

```bash
pip install -U langchain-snowflake
```

You can authenticate by one of the following ways:

1. Configure credentials by setting the following environment variables:

`SNOWFLAKE_USERNAME` should hold your Snowflake username.

`SNOWFLAKE_PASSWORD` should hold your Snowflake password.

`SNOWFLAKE_DATABASE` should hold a Snowflake database to operate in.

`SNOWFLAKE_SCHEMA` should hold the name of the schema to operate in.

`SNOWFLAKE_ROLE` should contain the name of the appropriate Snowflake role.

`SNOWFLAKE_AUTHENTICATOR` should contain the name of Snowflake authentication method, if not using username/password authentication.

Any of these paramaters can also be passed directly into the `CortexSearchRetriever` constructor. Those not passed will be inferred from the environment. For instance:

```python
from langchain_snowflake.search_retriever import CortexSearchRetriever

search = CortexSearchRetriever(
        role="snowflake_role",
        database="your_db",
        schema="your_schema",
        service="your_cortex_search_service_name",
        search_column="search_column_name",
)
```

Here, `role`, `database`, and `schema` are passed directly, while `SNOWFLAKE_USERNAME`, `SNOWFLAKE_PASSWORD`, and `SNOWFLAKE_AUTHENTICATOR` are inferred from the environment.

If the `SNOWFLAKE_AUTHENTICATOR` environment variable or `authenticator` property is set to `externalbrowser`, the `SNOWFLAKE_PASSWORD`/`password` need not be provided. `externalbrowser` auth will prompt to log in through an browser popup instead.

2. Alternatively, you can pass in a `snowflake.snowpark.Session` directly into the constructor. See [the Snowflake docs](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session) on how to create such a session.

```python
from langchain_snowflake.search_retriever import CortexSearchRetriever
from snowflake.snowpark import Session

# Create a snowflake session
snowflake_session = Session.builder.config(...).create()

search = CortexSearchRetriever(
        session=snowflake_session,
        service="your_cortex_search_service_name",
        search_column="search_column_name",
)
```

## Search Retriever

`CortexSearchRetriever` is a Third Party Retriever that allows users to utilize their Cortex Search Service within Langchain.

Given the service name and columns to search, the retriever will return matches found within the dataset with which the Cortex Search Service is associated.

```python
from langchain_snowflake.search_retriever import CortexSearchRetriever

search = CortexSearchRetriever(
        service="your_cortex_search_service_name",
        search_column="<search_column_name>",
        columns=["<col1>", "<col2>"],
        filter={"@eq": {"<column>": "<value>"}},
        limit=5
)

query="foo bar"
result = search.invoke(query)
for doc in result:
    print(doc)
```

The class requires the arguments below:
`service` corresponds to the name of your Cortex Search Service.

`search_column` is the search column of the Cortex Search Service.

`columns` corresponds to the columns to return in the search. If null or an empty list, only the `search_column` will be returned.

`filter` is an optional argument corresponding to any filters that should be utilized when searching.

`limit` corresponds to the number of search hits to return.
