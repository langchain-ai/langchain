from __future__ import annotations

import asyncio
import uuid
import warnings
from asyncio import Task
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from astrapy.db import AstraDB as LibAstraDB
    from astrapy.db import AsyncAstraDB

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

                from langchain_community.vectorstores import AstraDB
                from langchain_community.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = AstraDB(
                  embedding=embeddings,
                  collection_name="my_store",
                  token="AstraCS:...",
                  api_endpoint="https://<DB-ID>-us-east1.apps.astra.datastax.com"
                )

                vectorstore.add_texts(["Giraffes", "All good here"])
                results = vectorstore.similarity_search("Everything's ok", k=1)

      Constructor Args (only keyword-arguments accepted):
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
          pre_delete_collection (Optional[bool]): whether to delete the collection
              before creating it. If False and the collection already exists,
              the collection will be used as is.

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

    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            metadata_filter = {}
            for k, v in filter_dict.items():
                if k and k[0] == "$":
                    if isinstance(v, list):
                        metadata_filter[k] = [AstraDB._filter_to_metadata(f) for f in v]
                    else:
                        metadata_filter[k] = AstraDB._filter_to_metadata(v)  # type: ignore[assignment]
                else:
                    metadata_filter[f"metadata.{k}"] = v

            return metadata_filter

    def __init__(
        self,
        *,
        embedding: Embeddings,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[LibAstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        metric: Optional[str] = None,
        batch_size: Optional[int] = None,
        bulk_insert_batch_concurrency: Optional[int] = None,
        bulk_insert_overwrite_concurrency: Optional[int] = None,
        bulk_delete_concurrency: Optional[int] = None,
        pre_delete_collection: bool = False,
    ) -> None:
        """
        Create an AstraDB vector store object. See class docstring for help.
        """
        try:
            from astrapy.db import AstraDB as LibAstraDB
            from astrapy.db import AstraDBCollection
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )

        # Conflicting-arg checks:
        if astra_db_client is not None or async_astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' or 'async_astra_db_client' to "
                    "AstraDB if passing 'token' and 'api_endpoint'."
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

        self.astra_db = astra_db_client
        self.async_astra_db = async_astra_db_client
        self.collection = None
        self.async_collection = None

        if token and api_endpoint:
            self.astra_db = LibAstraDB(
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.namespace,
            )
            try:
                from astrapy.db import AsyncAstraDB

                self.async_astra_db = AsyncAstraDB(
                    token=self.token,
                    api_endpoint=self.api_endpoint,
                    namespace=self.namespace,
                )
            except (ImportError, ModuleNotFoundError):
                pass

        if self.astra_db is not None:
            self.collection = AstraDBCollection(
                collection_name=self.collection_name,
                astra_db=self.astra_db,
            )

        self.async_setup_db_task: Optional[Task] = None
        if self.async_astra_db is not None:
            from astrapy.db import AsyncAstraDBCollection

            self.async_collection = AsyncAstraDBCollection(
                collection_name=self.collection_name,
                astra_db=self.async_astra_db,
            )
            try:
                self.async_setup_db_task = asyncio.create_task(
                    self._setup_db(pre_delete_collection)
                )
            except RuntimeError:
                pass

        if self.async_setup_db_task is None:
            if not pre_delete_collection:
                self._provision_collection()
            else:
                self.clear()

    def _ensure_astra_db_client(self):  # type: ignore[no-untyped-def]
        if not self.astra_db:
            raise ValueError("Missing AstraDB client")

    async def _setup_db(self, pre_delete_collection: bool) -> None:
        if pre_delete_collection:
            await self.async_astra_db.delete_collection(  # type: ignore[union-attr]
                collection_name=self.collection_name,
            )
        await self._aprovision_collection()

    async def _ensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            await self.async_setup_db_task

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension

    def _provision_collection(self) -> None:
        """
        Run the API invocation to create the collection on the backend.

        Internal-usage method, no object members are set,
        other than working on the underlying actual storage.
        """
        self.astra_db.create_collection(  # type: ignore[union-attr]
            dimension=self._get_embedding_dimension(),
            collection_name=self.collection_name,
            metric=self.metric,
        )

    async def _aprovision_collection(self) -> None:
        """
        Run the API invocation to create the collection on the backend.

        Internal-usage method, no object members are set,
        other than working on the underlying actual storage.
        """
        await self.async_astra_db.create_collection(  # type: ignore[union-attr]
            dimension=self._get_embedding_dimension(),
            collection_name=self.collection_name,
            metric=self.metric,
        )

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
        self.delete_collection()
        self._provision_collection()

    async def aclear(self) -> None:
        """Empty the collection of all its stored entries."""
        await self._ensure_db_setup()
        if not self.async_astra_db:
            await run_in_executor(None, self.clear)
        await self.async_collection.delete_many({})  # type: ignore[union-attr]

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Remove a single document from the store, given its document_id (str).
        Return True if a document has indeed been deleted, False if ID not found.
        """
        self._ensure_astra_db_client()
        deletion_response = self.collection.delete_one(document_id)  # type: ignore[union-attr]
        return ((deletion_response or {}).get("status") or {}).get(
            "deletedCount", 0
        ) == 1

    async def adelete_by_document_id(self, document_id: str) -> bool:
        """
        Remove a single document from the store, given its document_id (str).
        Return True if a document has indeed been deleted, False if ID not found.
        """
        await self._ensure_db_setup()
        if not self.async_collection:
            return await run_in_executor(None, self.delete_by_document_id, document_id)
        deletion_response = await self.async_collection.delete_one(document_id)
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

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            concurrency (Optional[int]): max number of concurrent delete queries.
                Defaults to instance-level setting.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if kwargs:
            warnings.warn(
                "Method 'adelete' of AstraDB vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        if ids is None:
            raise ValueError("No ids provided to delete.")

        return all(
            await gather_with_concurrency(
                concurrency, *[self.adelete_by_document_id(doc_id) for doc_id in ids]
            )
        )

    def delete_collection(self) -> None:
        """
        Completely delete the collection from the database (as opposed
        to 'clear()', which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        self._ensure_astra_db_client()
        self.astra_db.delete_collection(  # type: ignore[union-attr]
            collection_name=self.collection_name,
        )

    async def adelete_collection(self) -> None:
        """
        Completely delete the collection from the database (as opposed
        to 'clear()', which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        await self._ensure_db_setup()
        if not self.async_astra_db:
            await run_in_executor(None, self.delete_collection)
        await self.async_astra_db.delete_collection(  # type: ignore[union-attr]
            collection_name=self.collection_name,
        )

    @staticmethod
    def _get_documents_to_insert(
        texts: Iterable[str],
        embedding_vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[DocDict]:
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        #
        documents_to_insert = [
            {
                "content": b_txt,
                "_id": b_id,
                "$vector": b_emb,
                "metadata": b_md,
            }
            for b_txt, b_emb, b_id, b_md in zip(
                texts,
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
        return uniqued_documents_to_insert

    @staticmethod
    def _get_missing_from_batch(
        document_batch: List[DocDict], insert_result: Dict[str, Any]
    ) -> Tuple[List[str], List[DocDict]]:
        if "status" not in insert_result:
            raise ValueError(
                f"API Exception while running bulk insertion: {str(insert_result)}"
            )
        batch_inserted = insert_result["status"]["insertedIds"]
        # estimation of the preexisting documents that failed
        missed_inserted_ids = {document["_id"] for document in document_batch} - set(
            batch_inserted
        )
        errors = insert_result.get("errors", [])
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
        return batch_inserted, missing_from_batch

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
        self._ensure_astra_db_client()

        embedding_vectors = self.embedding.embed_documents(list(texts))
        documents_to_insert = self._get_documents_to_insert(
            texts, embedding_vectors, metadatas, ids
        )

        def _handle_batch(document_batch: List[DocDict]) -> List[str]:
            im_result = self.collection.insert_many(  # type: ignore[union-attr]
                documents=document_batch,
                options={"ordered": False},
                partial_failures_allowed=True,
            )
            batch_inserted, missing_from_batch = self._get_missing_from_batch(
                document_batch, im_result
            )

            def _handle_missing_document(missing_document: DocDict) -> str:
                replacement_result = self.collection.find_one_and_replace(  # type: ignore[union-attr]
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
            return batch_inserted + batch_replaced

        _b_max_workers = batch_concurrency or self.bulk_insert_batch_concurrency
        with ThreadPoolExecutor(max_workers=_b_max_workers) as tpe:
            all_ids_nested = tpe.map(
                _handle_batch,
                batch_iterate(
                    batch_size or self.batch_size,
                    documents_to_insert,
                ),
            )
        return [iid for id_list in all_ids_nested for iid in id_list]

    async def aadd_texts(
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
            batch_concurrency (Optional[int]): number of concurrent batch insertions.
                Defaults to instance-level setting if not provided.
            overwrite_concurrency (Optional[int]): number of concurrent API calls to
                process pre-existing documents in each batch.
                Defaults to instance-level setting if not provided.

        A note on metadata: there are constraints on the allowed field names
        in this dictionary, coming from the underlying Astra DB API.
        For instance, the `$` (dollar sign) cannot be used in the dict keys.
        See this document for details:
            docs.datastax.com/en/astra-serverless/docs/develop/dev-with-json.html

        Returns:
            List[str]: List of ids of the added texts.
        """
        await self._ensure_db_setup()
        if not self.async_collection:
            await super().aadd_texts(
                texts,
                metadatas,
                ids=ids,
                batch_size=batch_size,
                batch_concurrency=batch_concurrency,
                overwrite_concurrency=overwrite_concurrency,
            )
        if kwargs:
            warnings.warn(
                "Method 'aadd_texts' of AstraDB vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        embedding_vectors = await self.embedding.aembed_documents(list(texts))
        documents_to_insert = self._get_documents_to_insert(
            texts, embedding_vectors, metadatas, ids
        )

        async def _handle_batch(document_batch: List[DocDict]) -> List[str]:
            im_result = await self.async_collection.insert_many(  # type: ignore[union-attr]
                documents=document_batch,
                options={"ordered": False},
                partial_failures_allowed=True,
            )
            batch_inserted, missing_from_batch = self._get_missing_from_batch(
                document_batch, im_result
            )

            async def _handle_missing_document(missing_document: DocDict) -> str:
                replacement_result = await self.async_collection.find_one_and_replace(  # type: ignore[union-attr]
                    filter={"_id": missing_document["_id"]},
                    replacement=missing_document,
                )
                return replacement_result["data"]["document"]["_id"]

            _u_max_workers = (
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency
            )
            batch_replaced = await gather_with_concurrency(
                _u_max_workers,
                *[_handle_missing_document(doc) for doc in missing_from_batch],
            )
            return batch_inserted + batch_replaced

        _b_max_workers = batch_concurrency or self.bulk_insert_batch_concurrency
        all_ids_nested = await gather_with_concurrency(
            _b_max_workers,
            *[
                _handle_batch(batch)
                for batch in batch_iterate(
                    batch_size or self.batch_size,
                    documents_to_insert,
                )
            ],
        )

        return [iid for id_list in all_ids_nested for iid in id_list]

    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        self._ensure_astra_db_client()
        metadata_parameter = self._filter_to_metadata(filter)
        #
        hits = list(
            self.collection.paginated_find(  # type: ignore[union-attr]
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

    async def asimilarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        await self._ensure_db_setup()
        if not self.async_collection:
            return await run_in_executor(
                None,
                self.asimilarity_search_with_score_id_by_vector,  # type: ignore[arg-type]
                embedding,
                k,
                filter,
            )
        metadata_parameter = self._filter_to_metadata(filter)
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
            async for hit in self.async_collection.paginated_find(
                filter=metadata_parameter,
                sort={"$vector": embedding},
                options={"limit": k, "includeSimilarity": True},
                projection={
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                },
            )
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
        )

    async def asimilarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = await self.embedding.aembed_query(query)
        return await self.asimilarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
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

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
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
            for (
                doc,
                score,
                doc_id,
            ) in await self.asimilarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = await self.embedding.aembed_query(query)
        return await self.asimilarity_search_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
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

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in await self.asimilarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = await self.embedding.aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding_vector,
            k,
            filter=filter,
        )

    @staticmethod
    def _get_mmr_hits(embedding, k, lambda_mult, prefetch_hits):  # type: ignore[no-untyped-def]
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

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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
        self._ensure_astra_db_client()
        metadata_parameter = self._filter_to_metadata(filter)

        prefetch_hits = list(
            self.collection.paginated_find(  # type: ignore[union-attr]
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

        return self._get_mmr_hits(embedding, k, lambda_mult, prefetch_hits)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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
        await self._ensure_db_setup()
        if not self.async_collection:
            return await run_in_executor(
                None,
                self.max_marginal_relevance_search_by_vector,
                embedding,
                k,
                fetch_k,
                lambda_mult,
                filter,
                **kwargs,
            )
        metadata_parameter = self._filter_to_metadata(filter)

        prefetch_hits = [
            hit
            async for hit in self.async_collection.paginated_find(
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
        ]

        return self._get_mmr_hits(embedding, k, lambda_mult, prefetch_hits)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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
        embedding_vector = await self.embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    @classmethod
    def _from_kwargs(
        cls: Type[ADBVST],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> ADBVST:
        known_kwargs = {
            "collection_name",
            "token",
            "api_endpoint",
            "astra_db_client",
            "async_astra_db_client",
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
        async_astra_db_client = kwargs.get("async_astra_db_client")
        namespace = kwargs.get("namespace")
        metric = kwargs.get("metric")

        return cls(
            embedding=embedding,
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            metric=metric,
            batch_size=kwargs.get("batch_size"),
            bulk_insert_batch_concurrency=kwargs.get("bulk_insert_batch_concurrency"),
            bulk_insert_overwrite_concurrency=kwargs.get(
                "bulk_insert_overwrite_concurrency"
            ),
            bulk_delete_concurrency=kwargs.get("bulk_delete_concurrency"),
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
        astra_db_store = AstraDB._from_kwargs(embedding, **kwargs)
        astra_db_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=kwargs.get("batch_size"),
            batch_concurrency=kwargs.get("batch_concurrency"),
            overwrite_concurrency=kwargs.get("overwrite_concurrency"),
        )
        return astra_db_store  # type: ignore[return-value]

    @classmethod
    async def afrom_texts(
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
        astra_db_store = AstraDB._from_kwargs(embedding, **kwargs)
        await astra_db_store.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=kwargs.get("batch_size"),
            batch_concurrency=kwargs.get("batch_concurrency"),
            overwrite_concurrency=kwargs.get("overwrite_concurrency"),
        )
        return astra_db_store  # type: ignore[return-value]

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
