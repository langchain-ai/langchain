import uuid
from typing import Any, Dict, List, Optional, cast

from langchain.callbacks.base import Callbacks
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.schema.storage import BaseStore
from langchain.text_splitter import TextSplitter
from langchain.vectorstores.base import VectorStore


class ParentDocumentRetriever(BaseRetriever):
    """Fetches small chunks, then fetches their parent documents.

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
        ... code-block:: python

        # Imports
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.storage import InMemoryStore

        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        # This text splitter is used to create the child documents
        # It should create documents smaller than the parent
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
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
    """

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    docstore: BaseStore
    """The storage layer for the parent documents"""
    child_splitter: TextSplitter
    """The text splitter to use to create child documents."""
    id_key: str = "doc_id"
    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        sub_docs = self.vectorstore.similarity_search(query)
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return_docs = cast(List[Document], docs)
        return return_docs

    def add_documents(self, documents: List[Document]) -> None:
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)

        docs = []
        full_docs = []
        for doc in documents:
            _id = str(uuid.uuid4())
            sub_docs = self.child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs)
        self.docstore.mset(full_docs)
