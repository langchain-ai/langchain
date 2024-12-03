"""Integration testing of retrievers/self_query/base.py

########################################
# Here is a Weaviate example:
# Launch the Weaviate container
cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up

# Run the test
pytest -sv tests/integration_tests/retrievers/self_query/test_base.py::test_weaviate

########################################

"""

import pytest

from langchain.retrievers.self_query import base as B


@pytest.mark.requires("langchain_openai", "langchain_weaviate")
def test_weaviate() -> None:
    """Test SelfQueryRetriever with Weaviate V4"""

    import weaviate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_weaviate.vectorstores import WeaviateVectorStore

    # Initialize the vector store
    weaviate_client = weaviate.connect_to_local(host="weaviate", port=8080)
    vectorstore = WeaviateVectorStore.from_documents(
        [],
        client=weaviate_client,
        index_name="test",
        embedding=OpenAIEmbeddings(),
    )

    metadata_field_info = [{"name": "foo", "type": "string", "description": "test"}]
    llm = ChatOpenAI()
    retriever = B.SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="test",
        metadata_field_info=metadata_field_info,
        verbose=True,
    )

    # Test the retriever
    retriever.get_relevant_documents("test")
    print("Successfully passed test_weaviate")
