from __future__ import annotations

import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.utils.iter import batch_iterate
from langchain.vectorstores.utils import maximal_marginal_relevance

ADBVST = TypeVar("ADBVST", bound="AstraDB")
T = TypeVar("T")
U = TypeVar("U")
DocDict = Dict[str, Any]  # dicts expressing entries to insert

# Batch/concurrency default values (if parameters not provided):
# Size of batches for bulk insertions:
#   (20 is the max batch size for the HTTP API at the time of writing)
DEFAULT_BATCH_SIZE = 20
# Number of threads to insert batches concurrently:
DEFAULT_BULK_INSERT_BATCH_CONCURRENCY = 16
# Number of threads in a batch to insert pre-existing entries:
DEFAULT_BULK_INSERT_OVERWRITE_CONCURRENCY = 10
# Number of threads (for deleting multiple rows concurrently):
DEFAULT_BULK_DELETE_CONCURRENCY = 20


def _unique_list(lst: List[T], key: Callable[[T], U]) -> List[T]:
    visited_keys: Set[U] = set()
    new_lst = []
    for item in lst:
        item_key = key(item)
        if item_key not in visited_keys:
            visited_keys.add(item_key)
            new_lst.append(item)
    return new_lst


class AstraDB(VectorStore):
    """Wrapper around DataStax Astra DB for vector-store workloads.

    To use it, you need a recent installation of the `astrapy` library
    and an Astra DB cloud database.

    For quickstart and details, visit:
        docs.datastax.com/en/astra/home/astra.html

    Example:
        .. code-block:: python

                from langchain.vectorstores import AstraDB
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = AstraDB(
                  embedding=embeddings,
                  collection_name="my_store",
                  token="AstraCS:...",
                  api_endpoint="https://<DB-ID>-us-east1.apps.astra.datastax.com"
                )

                vectorstore.add_texts(["Giraffes", "All good here"])
                results = vectorstore.similarity_search("Everything's ok", k=1)
    """

    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return {f"metadata.{mdk}": mdv for mdk, mdv in filter_dict.items()}

    def __init__(
        self,
        *,
        embedding: Embeddings,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[Any] = None,  # 'astrapy.db.AstraDB' if passed
        namespace: Optional[str] = None,
        metric: Optional[str] = None,
        batch_size: Optional[int] = None,
        bulk_insert_batch_concurrency: Optional[int] = None,
        bulk_insert_overwrite_concurrency: Optional[int] = None,
        bulk_delete_concurrency: Optional[int] = None,
    ) -> None:
        try:
            from astrapy.db import (
                AstraDB as LibAstraDB,
            )
            from astrapy.db import (
                AstraDBCollection as LibAstraDBCollection,
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )
        """
        Create an AstraDB vector store object.

        Args (only keyword-arguments accepted):
            embedding (Embeddings): embedding function to use.
            collection_name (str): name of the Astra DB collection to create/use.
            token (Optional[str]): API token for Astra DB usage.
            api_endpoint (Optional[str]): full URL to the API endpoint,
                such as "https://<DB-ID>-us-east1.apps.astra.datastax.com".
            astra_db_client (Optional[Any]): *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            namespace (Optional[str]): namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            metric (Optional[str]): similarity function to use out of those
                available in Astra DB. If left out, it will use Astra DB API's
                defaults (i.e. "cosine" - but, for performance reasons,
                "dot_product" is suggested if embeddings are normalized to one).

        Advanced arguments (coming with sensible defaults):
            batch_size (Optional[int]): Size of batches for bulk insertions.
            bulk_insert_batch_concurrency (Optional[int]): Number of threads
                to insert batches concurrently.
            bulk_insert_overwrite_concurrency (Optional[int]): Number of
                threads in a batch to insert pre-existing entries.
            bulk_delete_concurrency (Optional[int]): Number of threads
                (for deleting multiple rows concurrently).

        A note on concurrency: as a rule of thumb, on a typical client machine
        it is suggested to keep the quantity
            bulk_insert_batch_concurrency * bulk_insert_overwrite_concurrency
        much below 1000 to avoid exhausting the client multithreading/networking
        resources. The hardcoded defaults are somewhat conservative to meet
        most machines' specs, but a sensible choice to test may be:
            bulk_insert_batch_concurrency = 80
            bulk_insert_overwrite_concurrency = 10
        A bit of experimentation is required to nail the best results here,
        depending on both the machine/network specs and the expected workload
        (specifically, how often a write is an update of an existing id).
        Remember you can pass concurrency settings to individual calls to
        add_texts and add_documents as well.
        """

        # Conflicting-arg checks:
        if astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' to AstraDB if passing "
                    "'token' and 'api_endpoint'."
                )

        self.embedding = embedding
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.namespace = namespace
        # Concurrency settings
        self.batch_size: int = batch_size or DEFAULT_BATCH_SIZE
        self.bulk_insert_batch_concurrency: int = (
            bulk_insert_batch_concurrency or DEFAULT_BULK_INSERT_BATCH_CONCURRENCY
        )
        self.bulk_insert_overwrite_concurrency: int = (
            bulk_insert_overwrite_concurrency
            or DEFAULT_BULK_INSERT_OVERWRITE_CONCURRENCY
        )
        self.bulk_delete_concurrency: int = (
            bulk_delete_concurrency or DEFAULT_BULK_DELETE_CONCURRENCY
        )
        # "vector-related" settings
        self._embedding_dimension: Optional[int] = None
        self.metric = metric

        if astra_db_client is not None:
            self.astra_db = astra_db_client
        else:
            self.astra_db = LibAstraDB(
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.namespace,
            )
        self._provision_collection()

        self.collection = LibAstraDBCollection(
            collection_name=self.collection_name,
            astra_db=self.astra_db,
        )

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension

    def _drop_collection(self) -> None:
        """
        Drop the collection from storage.

        This is meant as an internal-usage method, no members
        are set other than actual deletion on the backend.
        """
        _ = self.astra_db.delete_collection(
            collection_name=self.collection_name,
        )
        return None

    def _provision_collection(self) -> None:
        """
        Run the API invocation to create the collection on the backend.

        Internal-usage method, no object members are set,
        other than working on the underlying actual storage.
        """
        _ = self.astra_db.create_collection(
            dimension=self._get_embedding_dimension(),
            collection_name=self.collection_name,
            metric=self.metric,
        )
        return None

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @staticmethod
    def _dont_flip_the_cos_score(similarity0to1: float) -> float:
        """Keep similarity from client unchanged ad it's in [0:1] already."""
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
        return None

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Remove a single document from the store, given its document_id (str).
        Return True if a document has indeed been deleted, False if ID not found.
        """
        deletion_response = self.collection.delete(document_id)
        return ((deletion_response or {}).get("status") or {}).get(
            "deletedCount", 0
        ) == 1

    def delete(
        self,
        ids: Optional[List[str]] = None,
        concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ids.

        Args:
            ids (Optional[List[str]]): List of ids to delete.
            concurrency (Optional[int]): max number of threads issuing
                single-doc delete requests. Defaults to instance-level setting.

        Returns:
            Optional[bool]: True if deletion is successful,
                False otherwise, None if not implemented.
        """

        if kwargs:
            warnings.warn(
                "Method 'delete' of AstraDB vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        if ids is None:
            raise ValueError("No ids provided to delete.")

        _max_workers = concurrency or self.bulk_delete_concurrency
        with ThreadPoolExecutor(max_workers=_max_workers) as tpe:
            _ = list(
                tpe.map(
                    self.delete_by_document_id,
                    ids,
                )
            )
        return True

    def delete_collection(self) -> None:
        """
        Completely delete the collection from the database (as opposed
        to 'clear()', which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        self._drop_collection()
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: Optional[int] = None,
        batch_concurrency: Optional[int] = None,
        overwrite_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through the embeddings and add them to the vectorstore.

        If passing explicit ids, those entries whose id is in the store already
        will be replaced.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of ids.
            batch_size (Optional[int]): Number of documents in each API call.
                Check the underlying Astra DB HTTP API specs for the max value
                (20 at the time of writing this). If not provided, defaults
                to the instance-level setting.
            batch_concurrency (Optional[int]): number of threads to process
                insertion batches concurrently. Defaults to instance-level
                setting if not provided.
            overwrite_concurrency (Optional[int]):  number of threads to process
                pre-existing documents in each batch (which require individual
                API calls). Defaults to instance-level setting if not provided.

        A note on metadata: there are constraints on the allowed field names
        in this dictionary, coming from the underlying Astra DB API.
        For instance, the `$` (dollar sign) cannot be used in the dict keys.
        See this document for details:
            docs.datastax.com/en/astra-serverless/docs/develop/dev-with-json.html

        Returns:
            List[str]: List of ids of the added texts.
        """

        if kwargs:
            warnings.warn(
                "Method 'add_texts' of AstraDB vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        _texts = list(texts)
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
        # make unique by id, keeping the last
        uniqued_documents_to_insert = _unique_list(
            documents_to_insert[::-1],
            lambda document: document["_id"],
        )[::-1]

        all_ids = []

        def _handle_batch(document_batch: List[DocDict]) -> List[str]:
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
            missed_inserted_ids = {
                document["_id"] for document in document_batch
            } - set(batch_inserted)
            errors = im_result.get("errors", [])
            # careful for other sources of error other than "doc already exists"
            num_errors = len(errors)
            unexpected_errors = any(
                error.get("errorCode") != "DOCUMENT_ALREADY_EXISTS" for error in errors
            )
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

            def _handle_missing_document(missing_document: DocDict) -> str:
                replacement_result = self.collection.find_one_and_replace(
                    filter={"_id": missing_document["_id"]},
                    replacement=missing_document,
                )
                return replacement_result["data"]["document"]["_id"]

            _u_max_workers = (
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency
            )
            with ThreadPoolExecutor(max_workers=_u_max_workers) as tpe2:
                batch_replaced = list(
                    tpe2.map(
                        _handle_missing_document,
                        missing_from_batch,
                    )
                )

            upsert_ids = batch_inserted + batch_replaced
            return upsert_ids

        _b_max_workers = batch_concurrency or self.bulk_insert_batch_concurrency
        with ThreadPoolExecutor(max_workers=_b_max_workers) as tpe:
            all_ids_nested = tpe.map(
                _handle_batch,
                batch_iterate(
                    batch_size or self.batch_size,
                    uniqued_documents_to_insert,
                ),
            )

        all_ids = [iid for id_list in all_ids_nested for iid in id_list]

        return all_ids

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
        hits = list(
            self.collection.paginated_find(
                filter=metadata_parameter,
                sort={"$vector": embedding},
                options={"limit": k, "includeSimilarity": True},
                projection={
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                },
            )
        )
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
            for (doc, score, doc_id) in self.similarity_search_with_score_id_by_vector(
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

        prefetch_hits = list(
            self.collection.paginated_find(
                filter=metadata_parameter,
                sort={"$vector": embedding},
                options={"limit": fetch_k, "includeSimilarity": True},
                projection={
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "$vector": 1,
                },
            )
        )

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
            query (str): Text to look up documents similar to.
            k (int = 4): Number of Documents to return.
            fetch_k (int = 20): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float = 0.5): Number between 0 and 1 that determines the degree
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
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ADBVST:
        """Create an Astra DB vectorstore from raw texts.

        Args:
            texts (List[str]): the texts to insert.
            embedding (Embeddings): the embedding function to use in the store.
            metadatas (Optional[List[dict]]): metadata dicts for the texts.
            ids (Optional[List[str]]): ids to associate to the texts.
            *Additional arguments*: you can pass any argument that you would
                to 'add_texts' and/or to the 'AstraDB' class constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AstraDb` vectorstore.
        """

        known_kwargs = {
            "collection_name",
            "token",
            "api_endpoint",
            "astra_db_client",
            "namespace",
            "metric",
            "batch_size",
            "bulk_insert_batch_concurrency",
            "bulk_insert_overwrite_concurrency",
            "bulk_delete_concurrency",
            "batch_concurrency",
            "overwrite_concurrency",
        }
        if kwargs:
            unknown_kwargs = set(kwargs.keys()) - known_kwargs
            if unknown_kwargs:
                warnings.warn(
                    "Method 'from_texts' of AstraDB vector store invoked with "
                    f"unsupported arguments ({', '.join(sorted(unknown_kwargs))}), "
                    "which will be ignored."
                )

        collection_name: str = kwargs["collection_name"]
        token = kwargs.get("token")
        api_endpoint = kwargs.get("api_endpoint")
        astra_db_client = kwargs.get("astra_db_client")
        namespace = kwargs.get("namespace")
        metric = kwargs.get("metric")

        astra_db_store = cls(
            embedding=embedding,
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            namespace=namespace,
            metric=metric,
            batch_size=kwargs.get("batch_size"),
            bulk_insert_batch_concurrency=kwargs.get("bulk_insert_batch_concurrency"),
            bulk_insert_overwrite_concurrency=kwargs.get(
                "bulk_insert_overwrite_concurrency"
            ),
            bulk_delete_concurrency=kwargs.get("bulk_delete_concurrency"),
        )
        astra_db_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=kwargs.get("batch_size"),
            batch_concurrency=kwargs.get("batch_concurrency"),
            overwrite_concurrency=kwargs.get("overwrite_concurrency"),
        )
        return astra_db_store

    @classmethod
    def from_documents(
        cls: Type[ADBVST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> ADBVST:
        """Create an Astra DB vectorstore from a document list.

        Utility method that defers to 'from_texts' (see that one).

        Args: see 'from_texts', except here you have to supply 'documents'
            in place of 'texts' and 'metadatas'.

        Returns:
            an `AstraDB` vectorstore.
        """
        return super().from_documents(documents, embedding, **kwargs)
