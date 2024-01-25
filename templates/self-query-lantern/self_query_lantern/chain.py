import os
from typing import List, Optional

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document, StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import BaseLLM
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores.lantern import Lantern
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from self_query_lantern import defaults, helper, prompts


class Query(BaseModel):
    __root__: str


def create_chain(
    llm: Optional[BaseLLM] = None,
    embeddings: Optional[Embeddings] = None,
    document_contents: str = defaults.DEFAULT_DOCUMENT_CONTENTS,
    metadata_field_info: List[AttributeInfo] = defaults.DEFAULT_METADATA_FIELD_INFO,
    collection_name: str = defaults.DEFAULT_COLLECTION_NAME,
    connection_string: str = defaults.DEFAULT_CONNECTION_STRING,
):
    """
    Create a chain that can be used to query a Lantern vector store with a self-querying
    capability. By default, this chain will use the OpenAI LLM and OpenAIEmbeddings, and
    work with the default document contents and metadata field info. You can override
    these defaults by passing in your own values.
    :param llm: an LLM to use for generating text
    :param embeddings: an Embeddings to use for generating queries
    :param document_contents: a description of the document set
    :param metadata_field_info: list of metadata attributes
    :param collection_name: name of the Lantern collection to use
    :param connection_string: connection string for the Lantern vector store
    :return:
    """
    llm = llm or OpenAI()
    embeddings = embeddings or OpenAIEmbeddings()

    # Set up a vector store to store your vectors and metadata

    vectorstore = Lantern(
        client=client,
        collection_name=collection_name,
        connection_string=connection_string,
        embeddings=embeddings,
    )

    # Set up a retriever to query your vector store with self-querying capabilities
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_contents, metadata_field_info, verbose=True
    )

    context = RunnableParallel(
        context=retriever | helper.combine_documents,
        query=RunnablePassthrough(),
    )
    pipeline = context | prompts.LLM_CONTEXT_PROMPT | llm | StrOutputParser()
    return pipeline.with_types(input_type=Query)


def initialize(
    embeddings: Optional[Embeddings] = None,
    collection_name: str = defaults.DEFAULT_COLLECTION_NAME,
    documents: List[Document] = defaults.DEFAULT_DOCUMENTS,
):
    """
    Initialize a vector store with a set of documents. By default, the documents will be
    compatible with the default metadata field info. You can override these defaults by
    passing in your own values.
    :param embeddings: an Embeddings to use for generating queries
    :param collection_name: name of the Lantern collection to use
    :param connection_string: connection string for the Lantern vector store
    :param documents: a list of documents to initialize the vector store with
    :return:
    """
    embeddings = embeddings or OpenAIEmbeddings()

    # Set up a vector store to store your vectors and metadata
    Lantern.from_documents(
        documents, embedding=embeddings, collection_name=collection_name, connection_string=connection_string
    )


# Create the default chain
chain = create_chain()
