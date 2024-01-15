import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.documents import Document

from langchain.indexes import IndexingResult, index
from langchain.retrievers import MultiVectorRetriever
from langchain.text_splitter import TextSplitter


class ParentDocumentRetriever(MultiVectorRetriever):
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

            # Imports
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import OpenAIEmbeddings
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

    child_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create child documents.
    If none, you will have to provide a list of text splitters corresponding
    to each document when calling `add_documents`."""

    id_key: str = "doc_id"
    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""

    parent_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    index_args: Optional[Dict[str, Any]] = None
    """If defined, indexing will be used when storing split documents to the
    vectorstore. This should contain the arguments that you would pass to
    `index`, but without `docs_source` and `vector_store`, both of which will
    be passed automatically."""

    @property
    def _add_documents(
        self
    ) -> Callable[[List[Document]], Union[List[str], IndexingResult]]:
        if self.index_args is None:
            return self.vectorstore.add_documents

        index_args = self.index_args
        index_args.update(
            vector_store=self.vectorstore,
        )

        def _index(docs: List[Document]):
            return index(docs, **index_args)

        return _index

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        child_splitters: Optional[List[TextSplitter]] = None,
        add_to_docstore: bool = True,
    ) -> None:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            child_splitters: Optional list of splitters to use for each document.
                If provided, it should be the same length as the list of documents.
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

        if self.child_splitter is None and child_splitters is None:
            raise ValueError(
                "Either one child splitter must be defined when instantiating "
                "this class, or a list of child splitters must be provided when "
                "calling this method."
            )

        if child_splitters is None:
            child_splitters = [self.child_splitter] * len(documents)
        else:
            if len(documents) != len(child_splitters):
                raise ValueError(
                    "Got uneven list of documents and child splitters. "
                    "If `child_spliiters` is provided, should be same length "
                    "as `documents`."
                )

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            splitter = child_splitters[i]
            sub_docs = splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        self._add_documents(docs)

        if add_to_docstore:
            self.docstore.mset(full_docs)
