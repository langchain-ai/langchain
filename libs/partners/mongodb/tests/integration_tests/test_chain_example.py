"Demonstrates MongoDBAtlasVectorSearch.as_retriever() invoked in a chain" ""

from __future__ import annotations

import os
from time import sleep

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import index

from ..utils import PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_chain_example"
INDEX_NAME = "vector_index"
DIMENSIONS = 1536
TIMEOUT = 60.0
INTERVAL = 0.5


@pytest.fixture
def collection() -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    client: MongoClient = MongoClient(CONNECTION_STRING)
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if all([INDEX_NAME != ix["name"] for ix in clxn.list_search_indexes()]):
        index.create_vector_search_index(
            collection=clxn,
            index_name=INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
            filters=None,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI for chat responses."
)
def test_chain(
    collection: Collection,
) -> None:
    """Demonstrate usage of MongoDBAtlasVectorSearch in a realistic chain

    Follows example in the docs: https://python.langchain.com/docs/how_to/hybrid/

    Requires OpenAI_API_KEY for embedding and chat model.
    Requires INDEX_NAME to have been set up on MONGODB_ATLAS_URI
    """

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    embedding_openai = OpenAIEmbeddings(
        openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
        model="text-embedding-3-small",
    )

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_openai,
        index_name=INDEX_NAME,
        text_key="page_content",
    )

    texts = [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
        "In 2019, I visited San Francisco",
        "In 2020, I visited Vancouver",
    ]
    vectorstore.add_texts(texts)

    # Give the index time to build (For CI)
    sleep(TIMEOUT)

    query = "In the United States, what city did I visit last?"
    # One can do vector search on the vector store, using its various search types.
    k = len(texts)

    store_output = list(vectorstore.similarity_search(query=query, k=k))
    assert len(store_output) == k
    assert isinstance(store_output[0], Document)

    # Unfortunately, the VectorStore output cannot be given to a Chat Model
    # If we wish Chat Model to answer based on our own data,
    # we have to give it the right things to work with.
    # The way that Langchain does this is by piping results along in
    # a Chain: https://python.langchain.com/v0.1/docs/modules/chains/

    # Now, we can turn our VectorStore into something Runnable in a Chain
    # by turning it into a Retriever.
    # For the simple VectorSearch Retriever, we can do this like so.

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=k))

    # This does not do much other than expose our search function
    # as an invoke() method with a a certain API, a Runnable.
    retriever_output = retriever.invoke(query)
    assert len(retriever_output) == len(texts)
    assert retriever_output[0].page_content == store_output[0].page_content

    # To get a natural language response to our question,
    # we need ChatOpenAI, a template to better frame the question as a prompt,
    # and a parser to send the output to a string.
    # Together, these become our Chain!
    # Here goes:

    template = """Answer the question based only on the following context.
     Answer in as few words as possible.
     {context}
     Question: {question}
     """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # type: ignore
        | prompt
        | model
        | StrOutputParser()
    )

    answer = chain.invoke("What city did I visit last?")

    assert "Paris" in answer
