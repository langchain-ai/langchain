import uuid
from typing import List, Optional, Sequence, Iterable, Any, Type, TypeVar, Dict

from langchain.retrievers import MultiVectorRetriever
from langchain.schema import BaseRetriever, BaseStore
from langchain.schema.document import Document, BaseDocumentTransformer
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

from langchain.pydantic_v1 import BaseModel,Field,Extra

# %%
VST = TypeVar("VST", bound="VectorStore")
class ParentDocumentVectorStore(VectorStore,BaseModel):

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    docstore: BaseStore[str, Document]
    """The storage layer for the parent documents"""
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    child_transformer: BaseDocumentTransformer
    """The transformer to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key = self.id_key,
            search_kwargs=self.search_kwargs,
            **kwargs)

    def add_documents(
            self,
            documents: Sequence[Document],
            *,
            ids: Optional[List[str]] = None,
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
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        # FIXME un bug si on livre des ids. On ne peut pas alors faire un master split
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

        docs = []
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


    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return self.vectorstore.delete(ids, **kwargs)

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k, **kwargs)

    """
    A VectorStoreWrapper to manage multiple associated documents.
    Then, it's possible to use an instance in `index()`
    """
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        self.vectorstore.add_texts(texts, metadatas, **kwargs)

    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VST:
        raise NotImplemented("from_texts not implemented")
