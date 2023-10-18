import uuid
from typing import List, Optional, Sequence

from langchain.retrievers import MultiVectorRetriever
from langchain.schema.document import BaseDocumentTransformer, Document


class ParentDocumentRetriever(MultiVectorRetriever):
    """Retrieve small chunks then retrieve their parent documents.

    When transform documents for retrieval, there are often conflicting desires:

    1. You may want to have small documents, so that their embeddings can most
        accurately reflect their meaning. If too long, then the embeddings can
        lose meaning.
    2. You want to have long enough documents that the context of each chunk is
        retained.

    The NewParentDocumentRetriever strikes that balance by transforming and storing
    small chunks of data. During retrieval, it first fetches the small chunks
    but then looks up the parent ids for those chunks and returns those larger
    documents.

    Note that "parent document" refers to the document that a small chunk
    originated from. This can either be the whole raw document OR a larger
    chunk.

    Examples:

        .. code-block:: python

            # Imports
            from langchain.vectorstores import Chroma
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.storage import InMemoryStore

            # This text splitter is used to create the parent documents
            parent_transformer = RecursiveCharacterTextSplitter(chunk_size=2000)
            # This text splitter is used to create the child documents
            # It should create documents smaller than the parent
            child_transformer = RecursiveCharacterTextSplitter(chunk_size=400)
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
            # The storage layer for the parent documents
            store = InMemoryStore()

            # Initialize the retriever
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_transformer=child_transformer,
                parent_transformer=parent_transformer,
            )
    """

    child_transformer: BaseDocumentTransformer
    """The transformer to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def add_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> List[Document]:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        if self.parent_transformer is not None:
            documents = self.parent_transformer.transform_documents(documents)

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

        docs: List[Document] = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_transformer.transform_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs)
        if add_to_docstore:
            self.docstore.mset(full_docs)
        return [doc for _, doc in full_docs]
