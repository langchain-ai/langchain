"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

import os
from typing import Any, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection

from tests.utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

INDEX_NAME = "langchain-test-index-vectorstores"
NAMESPACE = "langchain_test_db.langchain_test_vectorstores"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
DIMENSIONS = 1536


@pytest.fixture
def example_documents():
    return [
        Document(page_content="Dogs are tough.", metadata={"a": 1}),
        Document(page_content="Cats have fluff.", metadata={"b": 1}),
        Document(page_content="What is a sandwich?", metadata={"c": 1}),
        Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
    ]


def get_collection() -> Collection:
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Collection:
    return get_collection()


class TestMongoDBAtlasVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        collection = get_collection()
        if collection.count_documents({}):
            collection.delete_many({})  # type: ignore[index]

    @classmethod
    def teardown_class(cls) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture
    def embedding_openai(self) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        try:
            return OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model="text-embedding-3-small",
            )
        except Exception:
            return ConsistentFakeEmbeddings(DIMENSIONS)

    def test_from_documents(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_documents_no_embedding_return(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all(["embedding" not in key.metadata for key in output])
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_documents_embedding_return(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1, include_embeddings=True)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all([key.metadata.get("embedding") for key in output])
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_texts(self, embedding_openai: Embeddings, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1

    def test_from_texts_with_metadatas(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        metakeys = ["a", "b", "c", "d", "e"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any([key in output[0].metadata for key in metakeys])

    def test_from_texts_with_metadatas_and_pre_filter(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
        )
        assert output == []

    def test_mmr(self, embedding_openai: Embeddings, collection: Any) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        query = "foo"
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

    def test_retriever(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Demonstrate usage and parity of VectorStore similarity_search with Retriever.invoke."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )

        # We'll use the same query in both cases, although this isn't necessary
        query = "sandwich"

        # Case 1. Default arguments. Performs a vector search
        retriever_default_kwargs = vectorstore.as_retriever()
        result_retriever = retriever_default_kwargs.invoke(query)
        result_vectorstore = vectorstore.similarity_search(query)
        assert all(
            [
                result_retriever[i].page_content == result_vectorstore[i].page_content
                for i in range(len(result_retriever))
            ]
        )

        # Case 2. With kwargs that produce a hybrid search with an additional index
        search_kwargs = dict(
            strategy="hybrid",
            fulltext_index_name="text_index",
            fulltext_search_operator="text",
            fulltext_search_query="heart",
            fulltext_penalty=0,
            vector_penalty=0,
        )

        retriever_nontrivial = vectorstore.as_retriever(search_kwargs=search_kwargs)
        result_retriever = retriever_nontrivial.invoke(query)
        result_vectorstore = vectorstore.similarity_search(query, **search_kwargs)
        assert all(
            [
                result_retriever[i].page_content == result_vectorstore[i].page_content
                for i in range(len(result_retriever))
            ]
        )

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI for chat responses."
    )
    def test_chain(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Demonstrate usage of MongoDBAtlasVectorSearch in a realistic chain

        Follows example in the docs: https://python.langchain.com/v0.2/docs/how_to/hybrid/

        Requires OpenAI_API_KEY for embedding and chat model.
        """

        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_openai,
            vector_index_name=INDEX_NAME,
            fulltext_index_name="text_index",
        )

        texts = [
            "In 2023, I visited Paris",
            "In 2022, I visited New York",
            "In 2021, I visited New Orleans",
        ]
        vectorstore.add_texts(texts)

        query = "What city did I visit last?"
        # If we do a standard similarity search, we get all documents, back, up to limit k:
        out = vectorstore.as_retriever().invoke(query)
        assert len(out) == len(texts)

        # In the docs page, the example,  cities that don't include "new" or "New" are filtered out.
        # One way to do this is to add a post-filter stage.
        # This allows access to MongoDB's powerful query capabilities
        filtered = vectorstore.similarity_search(
            query=query,
            strategy="vector",  # default.
            post_filter_pipeline=[
                {"$match": {"text": {"$regex": ".*new.*", "$options": "i"}}}
            ],
        )
        all(["New" in doc.page_content for doc in filtered])

        # We could do a full-text search index to the Collection/VectorStore
        # In this, we get just 2 results
        # (Note that we're passing a query arg, but it is not used for the full-text search)
        fulltext_search = vectorstore.similarity_search(
            query, strategy="fulltext", fulltext_search_query="new"
        )
        all(["New" in doc.page_content for doc in fulltext_search])

        # But do we expect from a Hybrid Search? Some score attributed to the Vector Search and some to the Text, right?
        # Let's confirm
        hybrid_search = vectorstore.similarity_search(
            query="What city did I visit last?",
            strategy="hybrid",
            fulltext_search_query="new",
        )
        vector_scores = [doc.metadata["vector_score"] for doc in hybrid_search]
        fulltext_scores = [doc.metadata["fulltext_score"] for doc in hybrid_search]
        scores = [doc.metadata["score"] for doc in hybrid_search]
        assert len(fulltext_scores) == len(vector_scores)
        assert all(
            [
                vector_scores[i] + fulltext_scores[i] == scores[i]
                for i in range(len(vector_scores))
            ]
        )

        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI()

        retriever = vectorstore.as_retriever(
            search_kwargs=dict(
                strategy="hybrid", vector_penalty=10, fulltext_search_query="new"
            )
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        answer = chain.invoke("What city did I visit last?")

        assert "Paris" in answer

        # So, even though we've scored our results, we haven't filtered out documents without "New"
        # We need to pass more info to the Chat Model - the scores.
        # First, we create a template that includes the scores.

        scored_template = """Answer the question based only on the following context.
        You are an assistant tasked with analyzing the following documents. 
        Each document is accompanied by a relevant importance.
        Higher scores indicate more relevant documents. 
        Please consider these scores when generating your response.
        Disregard statements with scores below {threshold}

        {statements_and_scores}

        Question: {question}
        """

        # Then we add an additional post-processing step to get the information
        # we need from the metadata of the retriever's Output

        def format_documents_with_scores(documents: List[Document]) -> str:
            """Function composes output of Retriever, a List[Document], into a string highlighting Score information."""
            prompt_parts = []
            for doc in documents:
                score = doc.metadata.get("score", "No score available")
                content = doc.page_content
                prompt_parts.append(f"Score: {score}. Statement: {content}\n")
            return "\n".join(prompt_parts)

        postprocessor = RunnableLambda(format_documents_with_scores)

        chain = (
            {
                "statements_and_scores": retriever | postprocessor,
                "question": RunnablePassthrough(),
                "threshold": RunnableLambda(lambda x: 1.0),
            }
            | ChatPromptTemplate.from_template(scored_template)
            | model
            | StrOutputParser()
        )

        scored_response = chain.invoke("What city did I visit most recently?")

        assert "New York" in scored_response
        assert "Paris" not in scored_response

        # Finally, let's check if our full-text search passes on the query, and gets the same result.
        retriever = vectorstore.as_retriever(
            search_kwargs=dict(
                strategy="fulltext", vector_penalty=10, fulltext_search_query="new"
            )
        )
        chain = (
            {
                "statements_and_scores": retriever | postprocessor,
                "question": RunnablePassthrough(),
                "threshold": RunnableLambda(lambda x: 1.0),
            }
            | ChatPromptTemplate.from_template(scored_template)
            | model
            | StrOutputParser()
        )

        fulltext_response = chain.invoke("What city did I visit most recently?")
        assert "New York" in fulltext_response
        assert "Paris" not in fulltext_response

    def test_strategy_fulltext(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test result of performing fulltext search"""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )

        output = vectorstore.similarity_search(
            "ignored_query", fulltext_search_query="Sandwich", strategy="fulltext", k=10
        )
        assert len(output) == 1
        assert output[0].page_content == "What is a sandwich?"
        # Check for the presence of the metadata key
        assert "score" in output[0].metadata
        assert "c" in output[0].metadata

    def test_strategy_vector(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test explicitly passing vector kwarg matches default."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )

        output_explicit = vectorstore.similarity_search("Sandwich", strategy="vector")
        output_implicit = vectorstore.similarity_search("Sandwich", strategy="vector")
        assert output_explicit == output_implicit

    def test_strategy_hybrid(
        self,
        embedding_openai: Embeddings,
        collection: Any,
    ) -> None:
        """Test scores of Hybrid Search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_openai,
            vector_index_name=INDEX_NAME,
            fulltext_index_name="text_index",
        )

        texts = [
            "In 2023, I visited Paris",
            "In 2022, I visited New York",
            "In 2021, I visited New Orleans",
        ]
        vectorstore.add_texts(texts)

        output_vector = vectorstore.similarity_search_with_score(
            "Visited", strategy="vector"
        )
        n = len(texts)
        assert len(output_vector) == n
        # Check that all the scores roughly match
        assert all(
            [
                abs((output_vector[i][-1] - output_vector[i + 1][-1]) < 0.1)
                for i in range(n - 1)
            ]
        )

        output_hybrid = vectorstore.similarity_search_with_score(
            "Visited",
            strategy="hybrid",
            fulltext_search_query="Visited",
            fulltext_search_operator="text",
        )
        # Check reciprocal ranking scores make sense
        sum_scores = 0

        def scores_expected(n, penalty=0):
            return sum([(1 / (penalty + i)) for i in range(1, n + 1)])

        for i in range(n):
            sum_scores += output_hybrid[i][-1]
        assert abs(sum_scores - 2 * scores_expected(n)) < 0.001

        vector_penalty = 10
        output_penalized = vectorstore.similarity_search_with_score(
            "I visited where?",
            strategy="hybrid",
            fulltext_search_query="New",
            vector_penalty=vector_penalty,
        )
        # Test the weighting of vector scores
        scores = sum(
            [output_penalized[i][0].metadata["vector_score"] for i in range(n)]
        )
        assert abs(scores - scores_expected(n, vector_penalty)) < 0.001
        # Test that score == 0 for the Paris text
        assert 0 in [
            output_penalized[i][0].metadata["fulltext_score"] for i in range(n)
        ]

    def test_include_embeddings(
        self,
        embedding_openai: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test explicitly passing vector kwarg matches default."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )

        output_with = vectorstore.similarity_search(
            "Sandwich", include_embeddings=True, k=1
        )
        assert vectorstore._embedding_key in output_with[0].metadata
        output_without = vectorstore.similarity_search("Sandwich", k=1)
        assert vectorstore._embedding_key not in output_without[0].metadata
