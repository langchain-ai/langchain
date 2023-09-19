from langchain.docstore.document import Document
from langchain.vectorstores import Bagel
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
)


def test_similarity_search() -> None:
    """Test similarity search"""
    from bagel.config import Settings

    setting = Settings(
        bagel_api_impl="rest",
        bagel_server_host="api.bageldb.ai",
    )
    bagel = Bagel(client_settings=setting)
    bagel.add_texts(texts=["hello bagel", "hello langchain"])
    result = bagel.similarity_search(query="bagel", k=1)
    assert result == [Document(page_content="hello bagel")]
    bagel.delete_cluster()


def test_bagel() -> None:
    """Test from_texts"""
    texts = ["hello bagel", "hello langchain"]
    txt_search = Bagel.from_texts(cluster_name="testing", texts=texts)
    output = txt_search.similarity_search("hello bagel", k=1)
    assert output == [Document(page_content="hello bagel")]
    txt_search.delete_cluster()


def test_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"metadata": str(i)} for i in range(len(texts))]
    txt_search = Bagel.from_texts(
        cluster_name="testing",
        texts=texts,
        metadatas=metadatas,
    )
    output = txt_search.similarity_search("hello bagel", k=1)
    assert output == [Document(page_content="hello bagel", metadata={"metadata": "0"})]
    txt_search.delete_cluster()


def test_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    txt_search = Bagel.from_texts(
        cluster_name="testing", texts=texts, metadatas=metadatas
    )
    output = txt_search.similarity_search_with_score("hello bagel", k=1)
    assert output == [
        (Document(page_content="hello bagel", metadata={"page": "0"}), 0.0)
    ]
    txt_search.delete_cluster()


def test_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = [[1.1, 2.3, 3.2], [0.3, 0.3, 0.1]]

    vector_search = Bagel.from_texts(
        cluster_name="testing_vector",
        texts=texts,
        metadatas=metadatas,
        text_embeddings=embeddings,
    )

    embedded_query = [1.1, 2.3, 3.2]
    output = vector_search.similarity_search_by_vector_with_relevance_scores(
        query_embeddings=embedded_query, k=1
    )
    assert output == [
        (Document(page_content="hello bagel", metadata={"page": "0"}), 0.0)
    ]
    vector_search.delete_cluster()


def test_with_metadatas_with_scores_using_vector_embe() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embedding_function = FakeEmbeddings()

    vector_search = Bagel.from_texts(
        cluster_name="testing_vector_embedding1",
        texts=texts,
        metadatas=metadatas,
        embedding=embedding_function,
    )
    embedded_query = embedding_function.embed_query("hello bagel")
    output = vector_search.similarity_search_by_vector_with_relevance_scores(
        query_embeddings=embedded_query, k=1
    )
    assert output == [
        (Document(page_content="hello bagel", metadata={"page": "0"}), 0.0)
    ]
    vector_search.delete_cluster()


def test_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"first_letter": text[0]} for text in texts]
    txt_search = Bagel.from_texts(
        cluster_name="testing",
        texts=texts,
        metadatas=metadatas,
    )
    output = txt_search.similarity_search("bagel", k=1, where={"first_letter": "h"})
    assert output == [
        Document(page_content="hello bagel", metadata={"first_letter": "h"})
    ]
    output = txt_search.similarity_search("langchain", k=1, where={"first_letter": "h"})
    assert output == [
        Document(page_content="hello langchain", metadata={"first_letter": "h"})
    ]
    txt_search.delete_cluster()


def test_search_filter_with_scores() -> None:
    texts = ["hello bagel", "this is langchain"]
    metadatas = [{"source": "notion"}, {"source": "google"}]
    txt_search = Bagel.from_texts(
        cluster_name="testing",
        texts=texts,
        metadatas=metadatas,
    )
    output = txt_search.similarity_search_with_score(
        "hello bagel", k=1, where={"source": "notion"}
    )
    assert output == [
        (Document(page_content="hello bagel", metadata={"source": "notion"}), 0.0)
    ]
    txt_search.delete_cluster()


def test_with_include_parameter() -> None:
    """Test end to end construction and include parameter."""
    texts = ["hello bagel", "this is langchain"]
    docsearch = Bagel.from_texts(cluster_name="testing", texts=texts)
    output = docsearch.get(include=["embeddings"])
    assert output["embeddings"] is not None
    output = docsearch.get()
    assert output["embeddings"] is None
    docsearch.delete_cluster()


def test_bagel_update_document() -> None:
    """Test the update_document function in the Bagel class."""
    initial_content = "bagel"
    document_id = "doc1"
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})

    docsearch = Bagel.from_documents(
        cluster_name="testing_docs",
        documents=[original_doc],
        ids=[document_id],
    )

    updated_content = "updated bagel doc"
    updated_doc = Document(page_content=updated_content, metadata={"page": "0"})
    docsearch.update_document(document_id=document_id, document=updated_doc)
    output = docsearch.similarity_search(updated_content, k=1)
    assert output == [Document(page_content=updated_content, metadata={"page": "0"})]
