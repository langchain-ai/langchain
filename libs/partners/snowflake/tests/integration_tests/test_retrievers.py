"""Test Snowflake Cortex Search retriever.

You need to create a Cortex Search service in Snowflake with the specified
columns below to run the integration tests.
Follow the instructions in the example notebook:
`snowflake_cortex_search.ipynb` to set up the service and configure
authentication.

Set the following environment variables before the tests:
export SNOWFLAKE_ACCOUNT=<snowflake_account>
export SNOWFLAKE_USERNAME=<snowflake_username>
export SNOWFLAKE_PASSWORD=<snowflake_password>
export SNOWFLAKE_DATABASE=<snowflake_database>
export SNOWFLAKE_SCHEMA=<snowflake_schema>
export SNOWFLAKE_ROLE=<snowflake_role>
export SNOWFLAKE_CORTEX_SEARCH_SERVICE=<cortex_search_service>
"""

import pytest
from langchain_core.documents import Document
from langchain_snowflake import CortexSearchRetriever, CortexSearchRetrieverError


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke() -> None:
    """Test the invoke() method."""

    columns = ["name", "description", "era", "diet", "height_meters"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "filter": {"@eq": {"era": "Jurassic"}},
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        for column in columns:
            if column == search_column:
                continue
            assert column in doc.metadata
        # Validate the filter was passed through correctly
        assert doc.metadata["era"] == "Jurassic"


# Test no columns or filter
@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_columns_or_filter() -> None:
    """Test the invoke() method with no columns or filter."""

    kwargs = {
        "search_service": "dinosaur_svc",
        "search_column": "description",
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content


# Test no search column
@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_search_column() -> None:
    """Test the invoke() method with no search column."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "limit": 10,
    }

    with pytest.raises(CortexSearchRetrieverError):
        CortexSearchRetriever(**kwargs)


# Test no service name
@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_service_name() -> None:
    """Test the invoke() method with no search column."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "columns": columns,
        "limit": 10,
        "search_column": "description",
    }

    with pytest.raises(CortexSearchRetrieverError):
        CortexSearchRetriever(**kwargs)
