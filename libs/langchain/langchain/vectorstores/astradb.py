from __future__ import annotations

# import typing
from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import (
    # Any,
    # Callable,
    # Dict,
    Iterable,
    # List,
    # Optional,
    # Tuple,
    # Type,
    TypeVar,
    # Union,
)

import numpy as np

# if typing.TYPE_CHECKING:
#     from cassandra.cluster import Session

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

ADBVST = TypeVar("ADBVST", bound="AstraDB")

DEFAULT_INSERTION_BATCH_SIZE = 20  # 20 is the current maximum for the JSON API

def _unique_list(lst, key: lambda itm: itm):
    visited_keys = set()
    new_lst = []
    for item in lst:
        item_key = key(item)
        if item_key not in visited_keys:
            visited_keys.add(item_key)
            new_lst.append(item)
    return new_lst


def _batch_iterable(iterable: Iterable[T], batch_size: int) -> Iterable[Iterable[T]]:
    this_batch = []
    for entry in iterable:
        this_batch.append(entry)
        if len(this_batch) == batch_size:
            yield this_batch
            this_batch = []
    if this_batch:
        yield this_batch


class AstraDB(VectorStore):
    """Wrapper around DataStax Astra DB for vector-store workloads.

    To use it, you need a recent installation of the `astrapy` library
    and an Astra DB cloud database.

    For quickstart and details, visit:
        docs.datastax.com/en/astra-serverless/docs/vector-search/overview.html

    Example:
        .. code-block:: python

                from langchain.vectorstores import AstraDB
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                TODO change this example
                session = ...             # create your Cassandra session object
                keyspace = 'my_keyspace'  # the keyspace should exist already
                table_name = 'my_vector_store'
                vectorstore = Cassandra(embeddings, session, keyspace, table_name)
    """

    _embedding_dimension: Union[int, None]

    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return {
                f"metadata.{mdk}": mdv
                for mdk, mdv in filter_dict.items()
            }

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension

    def __init__(
        self,
        *,
        embedding: Embeddings,
        collection_name: str,
        token: str,
        api_endpoint: str,
        namespace: Optional[str] = None,
    ) -> None:
        try:
            from astrapy.db import AstraDB, AstraDBCollection
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )
        """Create a vector table."""
        self.embedding = embedding
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.namespace = namespace
        #
        self._embedding_dimension = None
        #
        self.astra_db = AstraDB(
            token = self.token,
            api_endpoint = self.api_endpoint,
            namespace=self.namespace,
        )
        self._provision_collection()
        #
        self.collection = AstraDBCollection(
            collection_name=self.collection_name,
            astra_db=self.astra_db,
        )

    def _drop_collection(self) -> None:
        """
        Drop the collection from storage.

        This is meant as an internal-usage method, no members
        are set other than actual deletion on the backend.
        """
        drop_collection_response = self.astra_db.delete_collection(
            collection_name=self.collection_name,
        )

    def _provision_collection(self) -> None:
        """
        Run the API invocation to create the collection on the backend.

        Internal-usage method, no object members are set,
        other than working on the underlying actual storage.
        """
        provision_collection_response = self.astra_db.create_collection(
            size=self._get_embedding_dimension(),
            collection_name=self.collection_name,
        )
        return None

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @staticmethod
    def _dont_flip_the_cos_score(similarity0to1: float) -> float:
        # the identity
        return similarity0to1

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The underlying API calls already returns a "score proper",
        i.e. one in [0, 1] where higher means more *similar*,
        so here the final score transformation is not reversing the interval:
        """
        return self._dont_flip_the_cos_score

    def clear(self) -> None:
        """Empty the collection of all its stored entries."""
        self._drop_collection()
        self._provision_collection()

    def delete_by_document_id(self, document_id: str) -> bool:
        deletion_response = self.collection.delete(document_id)
        return ((deletion_response or {}).get("status") or {}).get("deletedCount", 0) == 1

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        # TODO a gentle warning if kwargs is something as we swallow them

        if ids is None:
            raise ValueError("No ids provided to delete.")

        with ThreadPoolExecutor(max_workers=20) as tpe:
            deletion_responses = list(tpe.map(
                self.delete_by_document_id,
                ids,
            ))
        return all(deletion_responses)

    def delete_collection(self) -> None:
        """
        Completely, destructively delete the collection from the database.
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        self._drop_collection()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERTION_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
            batch_size (int): Number of documents in each API call

        Returns:
            List[str]: List of IDs of the added texts.
        """

        # TODO a gentle warning if kwargs is something as we swallow them

        _texts = list(texts)  # lest it be a generator or something
        if ids is None:
            ids = [uuid.uuid4().hex for _ in _texts]
        if metadatas is None:
            metadatas = [{} for _ in _texts]
        #
        embedding_vectors = self.embedding.embed_documents(_texts)

        documents_to_insert = [
            {
                "content": b_txt,
                "_id": b_id,
                "$vector": b_emb,
                "metadata": b_md,
            }
            for b_txt, b_emb, b_id, b_md in zip(
                _texts,
                embedding_vectors,
                ids,
                metadatas,
            )
        ]
        # unique by ID, always keep the last
        uniqued_documents_to_insert = _unique_list(
            documents_to_insert[::-1],
            lambda document: document["_id"],
        )[::-1]

        all_ids = []

        def _handle_batch(document_batch):
            im_result = self.collection.insert_many(
                documents=document_batch,
                options={"ordered": False},
                partial_failures_allowed=True,
            )
            if "status" not in im_result:
                raise ValueError(
                    f"API Exception while running bulk insertion: {str(im_result)}"
                )

            batch_inserted = im_result["status"]["insertedIds"]
            # estimation of the preexisting documents that failed
            missed_inserted_ids = {document["_id"] for document in document_batch} - set(batch_inserted)
            errors = im_result.get("errors", [])
            # careful for other sources of error other than "doc already exists"
            num_errors = len(errors)
            unexpected_errors = any(error.get("errorCode") != "DOCUMENT_ALREADY_EXISTS" for error in errors)
            if num_errors != len(missed_inserted_ids) or unexpected_errors:
                raise ValueError(
                    f"API Exception while running bulk insertion: {str(errors)}"
                )

            # deal with the missing insertions as upserts
            missing_from_batch = [
                document
                for document in document_batch
                if document["_id"] in missed_inserted_ids
            ]

            def _handle_missing_document(missing_document):
                replacement_result = self.collection.find_one_and_replace(
                    filter={"_id": missing_document["_id"]},
                    replacement=missing_document,
                )
                return [replacement_result["data"]["document"]["_id"]]

            with ThreadPoolExecutor(max_workers=10) as tpe2:
                batch_replaced = list(tpe2.map(
                    _handle_missing_document,
                    missing_from_batch,
                ))

            upsert_ids = batch_inserted + batch_replaced
            return upsert_ids

        with ThreadPoolExecutor(max_workers=5) as tpe:
            all_ids_nested = tpe.map(
                _handle_batch,
                _batch_iterable(
                    uniqued_documents_to_insert,
                    batch_size,
                ),
            )

        all_ids = [iid for id_list in all_ids_nested for iid in id_list]

        return all_ids

    # id-returning search facilities
    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        metadata_parameter = self._filter_to_metadata(filter)
        #
        hits = list(self.collection.paginated_find(
            filter=metadata_parameter,
            sort={"$vector": embedding},
            options={"limit": k},
            projection={
                "_id": 1,
                "content": 1,
                "metadata": 1,
                "$similarity": 1,
            }
        ))
        #
        return [
            (
                Document(
                    page_content=hit["content"],
                    metadata=hit["metadata"],
                ),
                hit["$similarity"],
                hit["_id"],
            )
            for hit in hits
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
        )

    # id-unaware search facilities
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, docId) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        metadata_parameter = self._filter_to_metadata(filter)

        prefetch_hits = list(self.collection.paginated_find(
            filter=metadata_parameter,
            sort={"$vector": embedding},
            options={"limit": fetch_k},
            projection={
                "_id": 1,
                "content": 1,
                "metadata": 1,
                "$similarity": 1,
                "$vector": 1,
            }
        ))

        # let the mmr utility pick the *indices* in the above array
        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [prefetch_hit["$vector"] for prefetch_hit in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_hits = [
            prefetch_hit
            for prefetch_index, prefetch_hit in enumerate(prefetch_hits)
            if prefetch_index in mmr_chosen_indices
        ]
        return [
            Document(
                page_content=hit["content"],
                metadata=hit["metadata"],
            )
            for hit in mmr_hits
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Optional.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    @classmethod
    def from_texts(
        cls: Type[ADBVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> ADBVST:
        """Create an Astra DB vectorstore from raw texts.

        No support for specifying text IDs

        Returns:
            an `AstraDb` vectorstore.
        """
        # TODO: these params, optional, etc - see init
        database_id = kwargs["database_id"]
        token = kwargs["token"]
        collection_name = kwargs["collection_name"]
        namespace = kwargs.get("namespace")
        #
        astra_db_store = cls(
            embedding=embedding,
            database_id=database_id,
            token=token,
            collection_name=collection_name,
            namespace=namespace,
        )
        astra_db_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            batch_size=batch_size,
        )
        return astra_db_store

    @classmethod
    def from_documents(
        cls: Type[ADBVST],
        documents: List[Document],
        embedding: Embeddings,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> ADBVST:
        """Create an Astra Db vectorstore from a document list.

        No support for specifying text IDs

        Returns:
            an `AstraDb` vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            batch_size=batch_size,
            **kwargs,
        )
