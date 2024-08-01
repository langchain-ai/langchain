# TODO - Rewrite this now that we've refactored

"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

import os
from time import monotonic, sleep
from typing import Any, Dict, List, Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import drop_vector_search_index

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch


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
