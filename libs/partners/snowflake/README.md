# langchain-snowflake

This package contains the LangChain integration with Snowflake

## Installation

```bash
pip install -U langchain-snowflake
```

And you should configure credentials by setting the following environment variables:

`SNOWFLAKE_USERNAME` should hold your Snowflake username.

`SNOWFLAKE_PASSWORD` should hold your Snowflake password.

`SNOWFLAKE_DATABASE` should hold a Snowflake database to operate in.

`SNOWFLAKE_SCHEMA` should hold the name of the schema to operate in.

`SNOWFLAKE_ROLE` should contain the name of the appropriate Snowflake role.

`SNOWFLAKE_WAREHOUSE` should contain the name of the Snowflake warehouse you would like to use.

Alternatively, if `SNOWFLAKE_PASSWORD` is not specified, the `AUTHENTICATOR` variable may be set to `externalbrowser` to log in through the externalbrowser instead.

These paramaters can also be passed directly into the class described below.

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
