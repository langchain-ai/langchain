import os
from typing import List, Optional

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.llms import BaseLLM
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from self_query_qdrant import defaults, helper, prompts


class Query(BaseModel):
    __root__: str


def create_chain(
    llm: Optional[BaseLLM] = None,
    embeddings: Optional[Embeddings] = None,
    document_contents: str = defaults.DEFAULT_DOCUMENT_CONTENTS,
    metadata_field_info: List[AttributeInfo] = defaults.DEFAULT_METADATA_FIELD_INFO,
    collection_name: str = defaults.DEFAULT_COLLECTION_NAME,
):
    """
    Create a chain that can be used to query a Qdrant vector store with a self-querying
    capability. By default, this chain will use the OpenAI LLM and OpenAIEmbeddings, and
    work with the default document contents and metadata field info. You can override
    these defaults by passing in your own values.
    :param llm: an LLM to use for generating text
    :param embeddings: an Embeddings to use for generating queries
    :param document_contents: a description of the document set
    :param metadata_field_info: list of metadata attributes
    :param collection_name: name of the Qdrant collection to use
    :return:
    """
    llm = llm or OpenAI()
    embeddings = embeddings or OpenAIEmbeddings()

    # Set up a vector store to store your vectors and metadata
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
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
    :param collection_name: name of the Qdrant collection to use
    :param documents: a list of documents to initialize the vector store with
    :return:
    """
    embeddings = embeddings or OpenAIEmbeddings()

    # Set up a vector store to store your vectors and metadata
    Qdrant.from_documents(
        documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


# Create the default chain
chain = create_chain()
