import uuid
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import DeleteResponse
from langchain_core.indexing.base_index import BaseIndex
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from langchain.pydantic_v1 import BaseModel
from langchain.retrievers import MultiVectorRetriever


class ParentDocumentRetriever(MultiVectorRetriever, BaseIndex[Document]):
    """Retrieve small chunks then retrieve their parent documents.

    When splitting documents for retrieval, there are often conflicting desires:

    1. You may want to have small documents, so that their embeddings can most
        accurately reflect their meaning. If too long, then the embeddings can
        lose meaning.
    2. You want to have long enough documents that the context of each chunk is
        retained.

    The ParentDocumentRetriever strikes that balance by splitting and storing
    small chunks of data. During retrieval, it first fetches the small chunks
    but then looks up the parent ids for those chunks and returns those larger
    documents.

    Note that "parent document" refers to the document that a small chunk
    originated from. This can either be the whole raw document OR a larger
    chunk.

    Examples:

        .. code-block:: python

            from langchain_community.embeddings import OpenAIEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain.storage import InMemoryStore

            # This text splitter is used to create the parent documents
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
            # This text splitter is used to create the child documents
            # It should create documents smaller than the parent
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
            # The storage layer for the parent documents
            store = InMemoryStore()

            # Initialize the retriever
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
    """  # noqa: E501

    child_splitter: TextSplitter
    """The text splitter to use to create child documents."""

    parent_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    child_metadata_fields: Optional[Sequence[str]] = None
    """Metadata fields to leave in child documents. If None, leave all parent document 
        metadata.
    """

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can be provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs, **kwargs)
        if add_to_docstore:
            self.docstore.mset(full_docs)


## V2 Implementation


class ParentRetrieverV2(BaseRetriever):
    underlying_retriever: BaseRetriever
    """The underlying retriever to use to retrieve the parent documents."""
    id_key: str = "doc_id"
    """The key to use to look up the parent documents."""
    store: BaseIndex[Document]
    transformer: BaseDocumentTransformer

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """
        # Config is a problem for composition?
        sub_docs = self.underlying_retriever.invoke(
            query, config={"callbacks": run_manager}
        )
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.store.get_by_ids(ids)
        return docs


# Needs a better name
class FullDocumentIndex(BaseIndex[Document], BaseModel):
    """A specialized index that stores small chunks of data and their embeddings."""

    vectorstore: VectorStore  # <-- Over engineered, a queryable index
    """A specialized index that stores small chunks of data and their embeddings."""
    store: BaseIndex[Document]
    """The storage interface for the parent documents"""
    id_key: str = "doc_id"
    """The key to use to look up the parent documents."""
    chunker: BaseDocumentTransformer
    """Used to chunk the source documents into small chunks that can will be searched."""

    def upsert(
        self,
        items: Sequence[Document],
        /,
        vector_store_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> UpsertResponse:
        """Upsert documents into the index and vectorstore."""
        for item in items:
            if item.id is None:
                raise ValueError("Document must have an ID.")

        child_docs = []

        # This logic is inefficient since we don't have a good way to keep
        # track of the original document
        for doc in items:
            # Can't do this efficiently cuz we have to keep track of the original
            # document
            sub_docs = self.chunker.transform_documents([doc])
            for sub_doc in sub_docs:
                # Select the metadata for the child documents
                for _doc in sub_doc:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
                    # Add the parent id to the child documents
                    _doc.metadata[self.id_key] = doc.id

            child_docs.extend(sub_docs)

        # Needs to clean UP first to keep things synchronized.
        self.vectorstore.delete_by_query(
            {
                "filter": {
                    self.id_key: {
                        "$in": [doc.id for doc in items],
                    }
                }
            }
        )

        self.vectorstore.upsert(child_docs, **(vector_store_kwargs or {}))

        return self.docstore.upsert(items)

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their ids."""
        return self.store.get_by_ids(ids)

    def delete_by_ids(
        self,
        ids: Sequence[str],
        /,
    ) -> DeleteResponse:
        """Delete documents by their ids."""
        # First delete from vectorstore
        self.vectorstore.delete_by_query(
            {
                "filter": {
                    self.id_key: {
                        "$in": ids,
                    }
                }
            }
        )
        delete_response = self.store.delete_by_ids(ids)
        return delete_response

    # This could be an argument for accepting kwargs in get_by_ids!
    def get_by_ids_from_vectorstore(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their ids."""
        return self.vectorstore.get_by_ids(ids)

    def get_retriever(self, **kwargs) -> ParentRetrieverV2:  # <-- Seems over engineered
        """Get documents by their ids."""
        # We do this to maintain the order of the ids that are returned
        return ParentRetrieverV2(
            underlying_retriever=self.vectortore.as_retriever(**kwargs),
            store=self.store,
        )
