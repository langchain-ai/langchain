# langchain-snowflake

This package contains the LangChain integration with Snowflake

## Installation

```bash
pip install -U langchain-snowflake
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

`SNOWFLAKE_USERNAME` should hold your Snowflake username.

`SNOWFLAKE_PASSWORD` should hold your Snowflake password.

`SNOWFLAKE_DATABASE` should hold a Snowflake database to operate in.

`SNOWFLAKE_SCHEMA` should hold the name of the schema to operate in.

`SNOWFLAKE_ROLE` should contain the name of the appropriate Snowflake role.

`SNOWFLAKE_WAREHOUSE` should contain the name of the Snowflake warehouse you would like to use.

Alternatively, if `SNOWFLAKE_PASSWORD` is not specified, the `AUTHENTICATOR` variable may be set to `externalbrowser` to log in through the externalbrowser instead.

These paramaters can also be passed directly into either of the classes below.

## Chat Models

`ChatSnowflakeCortex` class exposes chat models from Snowflake.

```python
from langchain_snowflake.chat import ChatSnowflakeCortex

llm = ChatSnowflakeCortex(model="snowflake-arctic")
llm.invoke("Sing a ballad of LangChain.")
```

`model` is the choice of LLM to use when generating response. See the documentation [here](https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex) for model options.

`temperature`, `max_tokens`, and `top_p` can be passed as additional arguments if they are to be changed.

`temperature` refers to model temperature and should be a number between 0.0 and 1.0. By default it is set to 0.7.

 `max_tokens` refers to the maximum number of tokens to output in the response, and is set to `None` by default.
 
`top_p` adjusts the number of options for predicted tokens based on cumulative properties and should be a number between 0.0 and 1.0. By default it is set to `None`.

## Search Retriever
`CortexSearchRetriever` is a Third Party Retriever that allows users to utilize their Cortex Search Service within Langchain.

Given the service name and columns to search, the retriever will return matches found within the database with which the Cortex Search Service is associated.

```python
from langchain_snowflake.search_retriever import CortexSearchRetriever

search = CortexSearchRetriever(
        service="your_cortex_search_service_name",
        columns=["<col1>", "<col2>"],
        filter={"@eq": {"<column>": "<value>"}},
        limit=5
)

kwargs={"search_column": "main_column_to_search"}
query="foo bar"
result = search.invoke(query, **kwargs)
for doc in result:
    print(doc)
```

The class requires the arguments below:
`service` corresponds to the name of your Cortex Search Service.

`columns` corresponds to the columns to return in the search.

`filter` is an optional argument corresponding to any filters that should be utilized when searching.

`limit` corresponds to the number of search hits to return.
