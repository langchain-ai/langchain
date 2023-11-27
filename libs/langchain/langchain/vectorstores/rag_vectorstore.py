import hashlib
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain.callbacks.manager import CallbackManagerForRetrieverRun

# Note: Import directly from langchain_core is not stable and generate some errors
# from langchain_core.pydantic_v1 import BaseModel, Extra, Field
# from langchain_core.stores import BaseStore
# from langchain_core.documents import BaseDocumentTransformer, Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema import BaseDocumentTransformer, BaseStore, Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain.storage import EncoderBackedStore

from .wrapper_vectorstore import WrapperVectorStore

# %%
VST = TypeVar("VST", bound="VectorStore")


class RAGVectorStore(BaseModel, WrapperVectorStore):
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

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    vectorstore: VectorStore
    """The real vectorstore for saving chunks"""
    docstore: BaseStore[str, Union[Document, List[str]]]
    """The storage layer for the parent documents"""

    source_id_key: str = "source"
    """The metadata to identify the id of the parents """

    chunk_id_key: str = "_chunk_id"
    """The metadata to identify the chunk. Add an id if the chunk can not have one """

    child_ids_key: str = "_child_ids"
    """Contain a list with the vectorstore id for all 
    corresponding transformed chunk."""

    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    chunk_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def _get_trunk_from_sub_docs(
        self, sub_docs: List[Document], **kwargs: Any
    ) -> List[Document]:
        if self.chunk_transformer:
            ids = []
            for d in sub_docs:
                if d.metadata[self.chunk_id_key] not in ids:
                    ids.append(d.metadata[self.chunk_id_key])
            docs = cast(List[Document], self.docstore.mget(ids))
            result = [d for d in docs if d is not None]
        else:
            result = sub_docs
        if "k" in kwargs:
            return result[: kwargs["k"]]
        else:
            return result

    async def _aget_trunk_from_sub_docs(
        self, sub_docs: List[Document], **kwargs: Any
    ) -> List[Document]:
        if self.chunk_transformer:
            ids = []
            for d in sub_docs:
                if d.metadata[self.chunk_id_key] not in ids:
                    ids.append(d.metadata[self.chunk_id_key])
            docs = cast(List[Document], self.docstore.mget(ids))
            result = [d for d in docs if d is not None]
        else:
            result = sub_docs
        if "k" in kwargs:
            return result[: kwargs["k"]]
        else:
            return result

    def _update_score_of_chunk(
        self, sub_chunks_and_score: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        ids = []
        scores: Dict[str, float] = {}
        for d, s in sub_chunks_and_score:
            if d.metadata[self.chunk_id_key] not in ids:
                chunk_id = d.metadata[self.chunk_id_key]
                ids.append(chunk_id)
                chunk_s = scores.get(chunk_id, -1.0)
                scores[chunk_id] = max(chunk_s, s)
        docs = cast(Sequence[Document], self.docstore.mget(ids))
        map_ids = {doc.metadata[self.source_id_key]: i for doc, i in zip(docs, ids)}
        return [
            (d, scores[map_ids[d.metadata[self.source_id_key]]])
            for d in docs
            if d is not None
        ]

    def _get_trunk_from_sub_docs_and_score(
        self, sub_docs_and_score: List[Tuple[Document, float]], **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self.chunk_transformer:
            result = self._update_score_of_chunk(sub_docs_and_score)
        else:
            result = sub_docs_and_score
        if "k" in kwargs:
            return result[: cast(int, kwargs["k"])]
        else:
            return result

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        if not self.chunk_transformer:
            return self.vectorstore.as_retriever(**kwargs)

        class ParentVectorRetriever(VectorStoreRetriever):
            """Retrieve from a set of multiple embeddings for the same document."""

            def _get_relevant_documents(
                me, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                """Get documents relevant to a query.
                Args:
                    query: String to find relevant documents for
                    run_manager: The callbacks handler to use
                Returns:
                    List of relevant documents
                """
                vectorstore = cast(RAGVectorStore, self.vectorstore)
                sub_docs = vectorstore.search(
                    query, search_type=self.search_type, **self.search_kwargs
                )
                return self._get_trunk_from_sub_docs(sub_docs)

        return ParentVectorRetriever(
            vectorstore=self,
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )

    def add_documents(
        self,
        documents: List[Document],
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
        """
        chunk_ids = None
        map_doc_ids: Dict[Any, str] = {}
        if self.parent_transformer:
            if ids:  # It's the parent ids
                if len(documents) != len(ids):
                    raise ValueError(
                        "Got uneven list of documents and ids. "
                        "If `ids` is provided, should be same length as `documents`."
                    )

                for id, doc in zip(ids, documents):
                    map_doc_ids[doc.metadata[self.source_id_key]] = id

            else:
                for doc in documents:
                    if self.source_id_key not in doc.metadata:
                        raise ValueError("Each document must have a uniq id.")
                    ids = []
                    for doc in documents:
                        # Some docstore refuse some characters in the id.
                        # We convert the id to hash
                        doc_id = doc.metadata[self.source_id_key]
                        hash_id = hashlib.sha256(
                            str(doc_id).encode("utf-8")
                        ).hexdigest()
                        ids.append(hash_id)
                        map_doc_ids[doc_id] = hash_id
            self.delete(ids=list(map_doc_ids.values()))
        else:
            chunk_ids = ids
            if chunk_ids:
                self.delete(ids=chunk_ids)
            ids = None

        if self.parent_transformer:
            chunk_documents = list(
                self.parent_transformer.transform_documents(documents)
            )
        else:
            chunk_documents = documents

        if chunk_ids is None:
            # Generate an id for each chunk, or use the ids
            # Put the associated chunk id the the transformation.
            # Then, it's possible to retrieve the original chunk with this
            # transformation.
            # for chunk in chunk_documents
            #     if self.chunk_id_key not in chunk.metadata:
            #         chunk.metadata[self.chunk_id_key]=str(uuid.uuid4())
            chunk_ids = [
                chunk.metadata.get(self.chunk_id_key, str(uuid.uuid4()))
                for chunk in chunk_documents
            ]

        chunk_ids_for_doc: Dict[str, List[str]] = {}
        if self.parent_transformer:
            # Associate each chunk with the parent
            for chunk_id, chunk_document in zip(chunk_ids, chunk_documents):
                doc_id = map_doc_ids[chunk_document.metadata[self.source_id_key]]
                list_of_chunk_ids = chunk_ids_for_doc.get(doc_id, [])
                list_of_chunk_ids.append(chunk_id)
                chunk_ids_for_doc[doc_id] = list_of_chunk_ids

        full_chunk_docs = []
        if not self.chunk_transformer:
            self.vectorstore.add_documents(documents=chunk_documents, ids=chunk_ids)
        else:
            for chunk_id, chunk_doc in zip(chunk_ids, chunk_documents):
                all_transformed_chunk: Sequence[
                    Document
                ] = self.chunk_transformer.transform_documents(
                    [chunk_doc]
                )  # TOTRY: use multiple documents
                # If in transformed chunk, add the id of the associated chunk
                for transformed_chunk in all_transformed_chunk:
                    transformed_chunk.metadata[self.chunk_id_key] = chunk_id
                # Save the transformed versions
                transformed_persistance_ids = self.vectorstore.add_documents(
                    list(all_transformed_chunk)
                )
                # Inject id of transformed ids in the chuck document
                chunk_doc.metadata[self.child_ids_key] = ",".join(
                    transformed_persistance_ids
                )
                # Prepare the mset in docstore
                full_chunk_docs.append((chunk_id, chunk_doc))

            # Add the chunks in docstore.
            # In the retriever, it's this instances to return
            # in metadata[child_ids_key], it's possible to find the id of all
            # transformed versions
            self.docstore.mset(full_chunk_docs)

        if self.parent_transformer:
            # With the *parent* mode, for each parent document,
            # we must save the id of all chunk.
            # Then, it's possible to remove/update all chunk when the parent document
            # was updated.
            # Save the parent association with all chunk
            ids = cast(List[str], ids)
            mset_values: List[Tuple[str, List[str]]] = []
            for parent_id, doc in zip(ids, documents):
                mset_values.append((parent_id, chunk_ids_for_doc[parent_id]))
            self.docstore.mset(mset_values)
            return ids
        else:
            return chunk_ids

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        # PPR: implement aadd_documents() if the pull-request is accepted
        raise NotImplementedError("aadd_documents not implemented")

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            raise ValueError("ids must be set")
        if self.parent_transformer:
            if not ids:
                raise ValueError("ids must be set")
            lists_of_chunk_by_doc_ids = cast(List[List[str]], self.docstore.mget(ids))
            chunk_by_doc_ids: List[str] = []
            for list_of_ids in lists_of_chunk_by_doc_ids:
                if list_of_ids:
                    chunk_by_doc_ids.extend([id for id in list_of_ids])
        else:
            chunk_by_doc_ids = ids

        if not any(chunk_by_doc_ids):
            return False

        transformed_ids = set()
        if self.chunk_transformer:
            chunk_docs = cast(List[Document], self.docstore.mget(chunk_by_doc_ids))
            self.docstore.mdelete(chunk_by_doc_ids)
            for chunk_doc in chunk_docs:
                if chunk_doc:
                    transformed_ids.update(
                        chunk_doc.metadata[self.child_ids_key].split(",")
                    )
        if transformed_ids:
            self.vectorstore.delete(ids=list(transformed_ids))
        elif self.parent_transformer:
            return self.vectorstore.delete(ids=chunk_by_doc_ids)
        elif not self.parent_transformer and self.chunk_transformer:
            return len(transformed_ids) != 0
        else:
            return self.vectorstore.delete(ids=ids)
        return False

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        # PPR: implement adelete() if the pull-request is accepted
        raise NotImplementedError("adelete not implemented")

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError("from_texts not implemented")

    # %% searches
    def _trunk_k(
        self, result: List[Document], kwargs: Dict[str, Any]
    ) -> List[Document]:
        if "k" in kwargs:
            return result[: kwargs["k"]]
        else:
            return result

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs = self.vectorstore.search(
            query=query, search_type=search_type, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, **kwargs)

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asearch(
            query=query, search_type=search_type, **kwargs
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.search(query=query, search_type="similarity", k=k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.asimilarity_search(
            query=query, search_type="similarity", k=k, **kwargs
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs_and_score = self.vectorstore.similarity_search_with_score(
            query=query, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs_and_score(subdocs_and_score, k=k)

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Not yet")

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs_and_score = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs_and_score(subdocs_and_score, k=k)

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # _search_kwargs = {**kwargs, **self.search_kwargs}
        # subdocs_and_score = await self.vectorstore.
        # asimilarity_search_with_relevance_scores(
        #     query=query, _search_kwargs
        # )
        # return self._get_trunk_from_sub_docs_and_score(subdocs_and_score, **kwargs)
        raise NotImplementedError("Not yet")

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = self.vectorstore.similarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = await self.vectorstore.asimilarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        subdocs = self.vectorstore.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        subdocs = await self.vectorstore.amax_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(
            subdocs,
            k=k,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        subdocs = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        subdocs = await self.vectorstore.amax_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return await self._aget_trunk_from_sub_docs(subdocs)

    @staticmethod
    def from_vs_in_memory(
        vectorstore: VectorStore,
        *,
        chunk_transformer: Optional[BaseDocumentTransformer] = None,
        parent_transformer: Optional[BaseDocumentTransformer] = None,
        source_id_key: str = "source",
        **kwargs: Any,
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        from langchain.storage import InMemoryStore

        from ..indexes.memory_recordmanager import MemoryRecordManager

        record_manager = MemoryRecordManager(namespace="in-memory")
        docstore = InMemoryStore()
        vectorstore = RAGVectorStore(
            vectorstore=vectorstore,
            docstore=docstore,
            parent_transformer=parent_transformer,
            chunk_transformer=chunk_transformer,
            **kwargs,
        )
        return (
            vectorstore,
            {
                "record_manager": record_manager,
                "vector_store": vectorstore,
                "source_id_key": source_id_key,
            },
        )

    @staticmethod
    def from_vs_in_sql(
        vectorstore: VectorStore,
        db_url: str,
        namespace: str = "rag_vectorstore",
        *,
        chunk_transformer: Optional[BaseDocumentTransformer] = None,
        parent_transformer: Optional[BaseDocumentTransformer] = None,
        source_id_key: str = "source",
        **kwargs: Any,
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        import pickle

        from langchain.indexes import SQLRecordManager

        from ..storage.sql_docstore import SQLStore

        record_manager = SQLRecordManager(namespace=namespace, db_url=db_url)
        record_manager.create_schema()
        sql_docstore = SQLStore(
            namespace=namespace,
            db_url=db_url,
        )
        sql_docstore.create_schema()
        docstore = EncoderBackedStore[str, Union[Document, List[str]]](
            store=sql_docstore,
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )
        vectorstore = RAGVectorStore(
            vectorstore=vectorstore,
            docstore=docstore,
            parent_transformer=parent_transformer,
            chunk_transformer=chunk_transformer,
            **kwargs,
        )
        return (
            vectorstore,
            {
                "record_manager": record_manager,
                "vector_store": vectorstore,
                "source_id_key": source_id_key,
            },
        )
