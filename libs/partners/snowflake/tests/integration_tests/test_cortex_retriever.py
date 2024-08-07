from langchain_snowflake.search_retriever import (
    CortexSearchRetriever,
)
from langchain_core.documents import Document


def test_cortex_search_invoke() -> None:
    """Test valid call to Azure AI Search.

    In order to run this test, you should provide
    your Snowflake account details and the path to your
    Snowflake Cortex Search Service. These can be
    specified as direct parameters or as environment variables.
    """
    retriever = CortexSearchRetriever()

    kwargs = {"search_column": ""}
    documents = retriever.invoke("what is langchain?", **kwargs)
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = CortexSearchRetriever(top_k=1)
    kwargs = {"search_column": ""}
    documents = retriever.invoke("what is langchain?", **kwargs)
    assert len(documents) <= 1
