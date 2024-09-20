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

import os
from unittest import mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

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


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_constructor_no_search_column() -> None:
    """Test the constructor with no search column name provided."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "limit": 10,
    }

    with pytest.raises(ValidationError):
        CortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_retriever_no_service_name() -> None:
    """Test the constructor with no search service name provided."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "columns": columns,
        "limit": 10,
        "search_column": "description",
    }

    with pytest.raises(ValidationError):
        CortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_invalid_filter() -> None:
    """Test the invoke() method with an invalid filter object."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "columns": columns,
        "search_service": "dinosaur_svc",
        "limit": 10,
        "search_column": "description",
        "filter": {"@eq": ["era", "Jurassic"]},
    }

    retriever = CortexSearchRetriever(**kwargs)

    with pytest.raises(CortexSearchRetrieverError):
        retriever.invoke("dinosaur with a large tail")


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_limit() -> None:
    """Test the invoke() method with an overridden limit."""

    columns = ["name", "description", "era", "diet", "height_meters"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 1,
    }

    retriever = CortexSearchRetriever(**kwargs)

    new_limit = 2

    documents = retriever.invoke("dinosaur with a large tail", limit=new_limit)
    assert len(documents) == new_limit


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_filter() -> None:
    """Test the invoke() method with an overridden filter."""

    columns = ["name", "description", "era", "diet", "height_meters"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
        "filter": {"@eq": {"era": "Jurassic"}},
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail", filter=None)
    assert len(documents) == 10

    observed_eras = set()
    for doc in documents:
        observed_eras.add(doc.metadata["era"])

    # Since we overrode the default filter with None, we should see more than one era.
    assert len(observed_eras) > 1


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_columns() -> None:
    """Test the invoke() method with overridden columns."""

    columns = ["description", "era"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail", columns=["description"])

    assert len(documents) == 10

    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert "era" not in doc.metadata


@pytest.mark.skip(
    """This test requires a Snowflake account with externalbrowser authentication 
    enabled."""
)
@pytest.mark.requires("snowflake.core")
@mock.patch.dict(os.environ, {"SNOWFLAKE_PASSWORD": ""})
def test_snowflake_cortex_search_constructor_externalbrowser_authenticator() -> None:
    """Test the constructor with external browser authenticator."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": "description",
        "limit": 10,
        "authenticator": "externalbrowser",
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
