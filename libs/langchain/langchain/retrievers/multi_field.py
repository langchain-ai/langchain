from typing import Any, Callable, List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import VectorStore


def _create_content_and_metadata(doc: Document, meta_data_keys: List[str] = []) -> str:
    """
    Concatenates the content of a document with its selected metadata fields.

    This static method takes a document and a list of metadata keys, and
    creates a string that includes the document's content and the specified
    metadata fields, formatted in a way that is suitable for indexing.

    Args:
        doc (Document): The document from which to create the content and
            metadata string.
        meta_data_keys (List[str], optional): A list of metadata keys to be
            included in the string. If empty, all metadata will be included.
            Defaults to an empty list.

    Returns:
        str: A string that contains the document's content and the specified
        metadata fields, formatted for indexing.
    """
    if len(meta_data_keys) == 0:
        meta_data = doc.metadata
    else:
        meta_data = {k: v for k, v in doc.metadata.items() if k in meta_data_keys}
    meta_data_str = "\n".join([f"===\n{k}: {v}" for k, v in meta_data.items()])
    return f"{meta_data_str}\n===\ncontent: {doc.page_content}"


class MultiFieldRetriever(BaseRetriever):
    """
    A retriever that allows for searching across multiple fields of a document.

    This class extends the functionality of a retriever by enabling the
    inclusion of additional metadata fields in the search process. It creates
    a composite document that includes both the content and specified metadata
    fields, which can then be indexed and searched by the underlying retriever
    or vector store.

    Attributes:
        retriever (BaseRetriever): The underlying retriever instance that is
            used to perform the actual retrieval of documents.

    Methods:
        from_documents: Class method to create an instance of MultiFieldRetriever
            from a list of documents and metadata keys.
        _create_content_and_metadata: Static method to concatenate document content
            with selected metadata fields into a single string.
        _get_relevant_documents: Retrieve documents relevant to a given query,
            using the underlying retriever and applying any specified callbacks.
    """

    retriever: BaseRetriever
    documents: List[Document] = []

    @classmethod
    def from_documents(
        cls,
        retriever: Any,
        documents: List[Document],
        meta_data_keys: List[str] = [],
        search_kwargs: dict = {},
        str_concat_func: Callable = _create_content_and_metadata,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Creates an instance of MultiFieldRetriever from a list of documents.

        This method processes the documents by creating a pseudo-document that
        includes both the content and the specified metadata fields. These
        pseudo-documents are then used to initialize the underlying retriever
        or vector store.

        Args:
            retriever (Union[BaseRetriever, VectorStore]): The retriever or
                vector store to be used for document retrieval.
            documents (List[Document]): A list of documents to be indexed.
            meta_data_keys (List[str], optional): A list of metadata keys to be
                included in the search index. Defaults to an empty list.
            search_kwargs (dict, optional): Additional keyword arguments to be
                passed to the search method of the retriever. Defaults to an
                empty dict.
            **kwargs (Any): Additional keyword arguments to be passed to the
                `from_documents` method of the retriever or vector store.

        Returns:
            BaseRetriever: An instance of MultiFieldRetriever with the underlying
            retriever initialized with the processed documents.
        """
        pseudo_doc_list = [
            Document(
                page_content=str_concat_func(doc, meta_data_keys),
                metadata={"uid": i},
            )
            for i, doc in enumerate(documents)
        ]

        retriever = retriever.from_documents(documents=pseudo_doc_list, **kwargs)
        if isinstance(retriever, VectorStore):
            return cls(retriever=retriever.as_retriever(search_kwargs=search_kwargs))
        return cls(retriever=retriever, documents=documents)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieves a list of documents relevant to the specified query.

        This method uses the underlying retriever to find documents that are
        relevant to the given query string. It also applies any callbacks
        provided by the run manager during the retrieval process.

        Args:
            query (str): The query string used to find relevant documents.
            run_manager (CallbackManagerForRetrieverRun): A callback manager
                that handles the execution of callbacks during the retrieval
                process.

        Returns:
            List[Document]: A list of documents that are relevant to the query.
        """
        sub_docs = self.retriever.get_relevant_documents(query)
        ids = [doc.metadata["uid"] for doc in sub_docs]
        return [self.documents[i] for i in ids]
