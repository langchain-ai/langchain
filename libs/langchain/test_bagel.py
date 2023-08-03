from langchain.vectorstores import Bagel
from bagel.config import Settings
from langchain.docstore.document import Document


def test_similarity_search() -> None:
    """Test smiliarity search"""
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
    text_search = Bagel.from_texts(
        cluster_name="testing", texts=texts
    )
    output = text_search.similarity_search("hello bagel", k=1)
    assert output == [Document(page_content="hello bagel")]
    text_search.delete_cluster()


def test_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"metadata": str(i)} for i in range(len(texts))]
    text_search = Bagel.from_texts(
        cluster_name="testing",
        texts=texts,
        metadatas=metadatas,
    )
    output = text_search.similarity_search("hello bagel", k=1)
    assert output == [Document(page_content="hello bagel", metadata={"metadata": "0"})]
    text_search.delete_cluster()


def test_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    text_search = Bagel.from_texts(
        cluster_name="testing",
        texts=texts,
        metadatas=metadatas
    )
    output = text_search.similarity_search_with_score("hello bagel", k=1)
    assert output == [(Document(page_content="hello bagel", metadata={"page": "0"}), 0.0)]
    text_search.delete_cluster()


def test_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["hello bagel", "hello langchain"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = [[1.1, 2.3, 3.2],
                  [0.3, 0.3, 0.1]]

    vector_search = Bagel.from_texts(
        cluster_name="testing_vector",
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    embedded_query = [[1.1, 2.3, 3.2]]
    output = vector_search.similarity_search_by_vector_with_relevance_scores(
        embedding=embedded_query, k=1
    )
    assert output == [(Document(page_content="hello bagel", metadata={"page": "0"}), 0.0)]
    vector_search.delete_cluster()


def main():
    """Bagel intigaration test"""
    test_similarity_search()
    test_bagel()
    test_with_metadatas()
    test_with_metadatas_with_scores()
    test_with_metadatas_with_scores_using_vector()


if __name__ == "__main__":
    main()
