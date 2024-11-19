from typing import Any, List, Optional

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


class LindormParentDocumentRetriever(ParentDocumentRetriever):
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

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_splitter: TextSplitter
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

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

            kwargs: Additional keyword arguments passed to the parent document
                routing_field: which field in metadata to use as the document's routing
                              key
                tag: which field in metadata to use as the document's identity
                metadata: global metadata to override parent document's metadata

                {
                    "routing_field": "split_setting",
                    "tag": "source",
                    "metadata":{
                        "split_setting": "10"
                     }
                }

        """
        routing_field = kwargs.pop("routing_field", "split_setting")
        tag = kwargs.pop("tag", "source")
        metadata = kwargs.pop("metadata", {})
        routing = metadata.get(routing_field, "")
        documents = self.parent_splitter.split_documents(documents)
        parent_docs = []
        child_docs = []
        for i, doc in enumerate(documents):
            doc.metadata.update(metadata)
            _id = f"{doc.metadata[tag]}_{routing}_parent_{i}"
            doc.id = _id
            sub_docs = self.child_splitter.split_documents([doc])
            for j, _doc in enumerate(sub_docs):
                _doc.metadata[self.id_key] = _id
                _doc.id = f"{_id}_child_{j}"
            child_docs.extend(sub_docs)
            parent_docs.append((_id, doc))
        child_ids = [d.id for d in child_docs]
        self.vectorstore.add_documents(child_docs, ids=child_ids, **kwargs)
        self.docstore.mset(parent_docs)
