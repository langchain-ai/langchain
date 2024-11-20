from __future__ import annotations

import json
import logging
import queue
import time
import uuid
from threading import Event, Thread
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from tenacity import retry, stop_after_attempt, wait_fixed

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
logger = logging.getLogger(__name__)


def _import_opensearch() -> Any:
    """Import OpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return OpenSearch


def _import_bulk() -> Any:
    """Import bulk if available, otherwise raise error."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return bulk


def _import_not_found_error() -> Any:
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError


def _get_lindorm_search_client(lindorm_search_url: str, **kwargs: Any) -> Any:
    """
    Get lindorm search client through `opensearchpy` base on the lindorm_search_url,
    otherwise raise error.
    """
    try:
        opensearch = _import_opensearch()
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 600
        if kwargs.get("retry_on_timeout") is None:
            kwargs["retry_on_timeout"] = True
        if kwargs.get("max_retries") is None:
            kwargs["max_retries"] = 3
        if kwargs.get("pool_maxsize") is None:
            kwargs["pool_maxsize"] = 20
        client = opensearch(lindorm_search_url, **kwargs)
    except ValueError as e:
        raise ImportError(
            f"Lindorm Search client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client


def _bulk_ingest_embeddings(
    client: Any,
    index_name: str,
    embeddings: List[List[float]],
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    vector_field: str = "vector_field",
    text_field: str = "text",
    max_chunk_bytes: Optional[int] = 10 * 1024 * 1024,
    routing_field: Optional[str] = None,
) -> List[str]:
    """Bulk Ingest Embeddings into given index."""
    bulk = _import_bulk()
    requests = []
    return_ids = []

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            "_id": _id,
            vector_field: embeddings[i],
            text_field: text,
            "metadata": metadata,
        }
        if routing_field:
            # Get routing from metadata if it exists
            routing = metadata.get(routing_field, None)
            if not routing:
                raise RuntimeError(
                    f"routing field [{routing_field}] no found in metadata [{metadata}]"
                )
            else:
                request["routing"] = routing
        requests.append(request)
        return_ids.append(_id)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
    def bulk_with_retry(
        client: Any, requests: List[dict], max_chunk_bytes: Optional[int]
    ) -> Any:
        bulk(client, requests, max_chunk_bytes=max_chunk_bytes)

    try:
        bulk_with_retry(client, requests, max_chunk_bytes)
    except Exception as e:
        logger.error(f"RetryError in bulking:{e}")
    return return_ids


def _default_text_mapping(dimension: int, method_name: str, **kwargs: Any) -> Dict:
    routing_field = kwargs.get("routing_field", None)
    excludes_from_source = kwargs.get("excludes_from_source", None)
    analyzer = kwargs.get("analyzer", "ik_max_word")
    text_field = kwargs.get("text_field", "text")
    engine = kwargs["engine"]
    shard = kwargs["shards"]
    space_type = kwargs["space_type"]
    data_type = kwargs["data_type"]
    vector_field = kwargs["vector_field"]

    if method_name == "ivfpq":
        ivfpq_m = kwargs["ivfpq_m"]
        nlist = kwargs["nlist"]
        centroids_use_hnsw = True if nlist > 10000 else False
        centroids_hnsw_m = 24
        centroids_hnsw_ef_construct = 500
        centroids_hnsw_ef_search = 100
        parameters = {
            "m": ivfpq_m,
            "nlist": nlist,
            "centroids_use_hnsw": centroids_use_hnsw,
            "centroids_hnsw_m": centroids_hnsw_m,
            "centroids_hnsw_ef_construct": centroids_hnsw_ef_construct,
            "centroids_hnsw_ef_search": centroids_hnsw_ef_search,
        }
    elif method_name == "hnsw":
        neighbor = kwargs["hnsw_m"]
        ef_construction = kwargs["hnsw_ef_construction"]
        parameters = {"m": neighbor, "ef_construction": ef_construction}
    elif method_name == "flat":
        parameters = {}
    else:
        raise RuntimeError(f"unexpected method_name: {method_name}")

    mapping = {
        "settings": {"index": {"number_of_shards": shard, "knn": True}},
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "data_type": data_type,
                    "method": {
                        "engine": engine,
                        "name": method_name,
                        "space_type": space_type,
                        "parameters": parameters,
                    },
                },
                text_field: {"type": "text", "analyzer": analyzer},
            }
        },
    }

    if excludes_from_source:
        mapping["mappings"]["_source"] = {
            "excludes": excludes_from_source
        }  # e.g. {"excludes": ["vector_field"]}

    if method_name == "ivfpq" and routing_field is not None:
        mapping["settings"]["index"]["knn_routing"] = True
        mapping["settings"]["index"]["knn.offline.construction"] = True

    if method_name == "flat" and routing_field is not None:
        mapping["settings"]["index"]["knn_routing"] = True

    return mapping


def embedding_thread(
    embedding: Embeddings,
    data_bulk_queue: queue.Queue,
    embeddings_queue: queue.Queue,
    keyboard_stop_event: Event,
    expect_dimension: int,
) -> Any:
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def embedding_and_write_queue(data_bulk: Any) -> Any:
        try:
            texts_bulk, ids_bulk, metadatas_bulk = data_bulk
            embeddings_bulk = embedding.embed_documents(texts_bulk)
            if not isinstance(embeddings_bulk, list):
                raise RuntimeError("Embedding failed!")
            if expect_dimension != len(embeddings_bulk[0]):
                logger.error(
                    f"Embedding dimension dismatch, "
                    f"expect [{expect_dimension}], "
                    f"get [{len(embeddings_bulk[0])}], "
                    f"try set kwargs 'dimension' "
                    f"when initializing lindorm search store"
                )
                raise RuntimeError("Embedding dimension dismatch!")
            embeddings_queue.put(
                [texts_bulk, embeddings_bulk, ids_bulk, metadatas_bulk], timeout=1
            )
        except queue.Full:
            raise RuntimeError(
                f"Embedding queue is full with maxsize:{embeddings_queue.maxsize}"
            )

    logger.info("embedding_thread start")
    while not keyboard_stop_event.is_set():
        try:
            data_bulk = data_bulk_queue.get(timeout=1)
            # retry once exception throwed
            embedding_and_write_queue(data_bulk)
        except queue.Empty:
            logger.debug("texts queue is empty, wait to next batch")
            # set event to make bulk_ingest_embedding_thread end as expected
            time.sleep(1)
        except Exception as e:
            logger.error(f"RetryError in embedding:{e}")
            keyboard_stop_event.set()
    logger.info("embedding_thread stop")


def bulk_ingest_embedding_thread(
    embeddings_queue: queue.Queue,
    return_ids_queue: queue.Queue,
    keyboard_stop_event: Event,
    client: Any,
    **kwargs: Any,
) -> None:
    index_name = kwargs.get("index_name")
    if index_name is None:
        logger.error("index name can't be empty")
        return
    vector_field = kwargs.get("vector_field", "vector_field")
    text_field = kwargs.get("text_field", "text")
    max_chunk_bytes = kwargs.get("max_chunk_bytes", 10 * 1024 * 1024)
    routing_field = kwargs.get("routing_field", None)

    logger.info("bulk_ingest_embedding_thread start")
    while not keyboard_stop_event.is_set():
        try:
            data_bulk = embeddings_queue.get(timeout=1)
            texts_bulk, embeddings_bulk, ids_bulk, metadatas_bulk = data_bulk
            return_ids = _bulk_ingest_embeddings(
                client,
                index_name,
                embeddings_bulk,
                texts_bulk,
                ids=ids_bulk,
                metadatas=metadatas_bulk,
                vector_field=vector_field,
                text_field=text_field,
                max_chunk_bytes=max_chunk_bytes,
                routing_field=routing_field,
            )
            return_ids_queue.put(return_ids)
        except queue.Empty:
            logger.debug("embedding query is empty, wait to next batch")
            time.sleep(1)
        except Exception as e:
            logger.error(e)
            keyboard_stop_event.set()
    logger.info("bulk_ingest_embedding_thread stop")


class LindormVectorStore(VectorStore):
    """
    Implement of ElasticSearch/OpenSearch-Compatible Lindorm Search Engine vector store.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import LindormSearchStore
            lindorm_vector_search = LindormSearchStore(
                "http://{instanceid}-proxy-search-vpc.lindorm.aliyuncs.com:30070",
                "embeddings",
                embedding
            )

    """

    texts_queue: queue.Queue[Any]
    embeddings_queue: queue.Queue[Any]
    return_ids_queue: queue.Queue[Any]

    def __init__(
        self,
        lindorm_search_url: str,
        index_name: str,
        embedding: Embeddings,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.embedding = embedding
        self.index_name = index_name
        self.dimension = kwargs.pop("dimension", 1024)

        # init client
        self.client = _get_lindorm_search_client(lindorm_search_url, **kwargs)

        # init queue, event and working thread
        self.texts_queue = queue.Queue()
        self.embeddings_queue = queue.Queue()
        self.return_ids_queue = queue.Queue()
        self.keyboard_stop_event = Event()

        self.embed_jobs: List[Any] = []
        self.write_jobs: List[Any] = []

        # self.progress_bar = tqdm()
        self.total_done = 0

        # init route info
        self.ivfpq_trained = True
        if kwargs.get("routing_field") is not None:
            self.is_route_index = True
        else:
            self.is_route_index = False

        # init index & threads
        kwargs.pop("bulk_size", None)  # avoid keyword repeat error
        self.kwargs = kwargs
        self.init_index(**kwargs)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def init_index(self, **kwargs: Any) -> None:
        not_found_error = _import_not_found_error()
        try:
            search_index = self.client.indices.get(index=self.index_name)
            logger.info("search_index: ", search_index)
        except not_found_error:
            vector_field = kwargs.pop("vector_field", "vector_field")
            shards = kwargs.pop("shards", 2)

            engine = kwargs.pop("engine", "lvector")
            method_name = kwargs.pop("method_name", "hnsw")
            data_type = kwargs.pop("data_type", "float")
            space_type = kwargs.pop("space_type", "cosinesimil")

            hnsw_m = kwargs.pop("hnsw_m", 24)
            hnsw_ef_construction = kwargs.pop("hnsw_ef_construction", 500)
            ivfpq_m = kwargs.pop("ivfpq_m", self.dimension)
            nlist = kwargs.pop("nlist", 1000)
            centroids_use_hnsw = kwargs.pop(
                "centroids_use_hnsw", True if nlist >= 5000 else False
            )
            centroids_hnsw_m = kwargs.pop("centroids_hnsw_m", 24)
            centroids_hnsw_ef_construct = kwargs.pop("centroids_hnsw_ef_construct", 500)
            centroids_hnsw_ef_search = kwargs.pop("centroids_hnsw_ef_search", 100)

            mapping = _default_text_mapping(
                self.dimension,
                method_name,
                shards=shards,
                engine=engine,
                data_type=data_type,
                space_type=space_type,
                vector_field=vector_field,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                nlist=nlist,
                ivfpq_m=ivfpq_m,
                centroids_use_hnsw=centroids_use_hnsw,
                centroids_hnsw_m=centroids_hnsw_m,
                centroids_hnsw_ef_construct=centroids_hnsw_ef_construct,
                centroids_hnsw_ef_search=centroids_hnsw_ef_search,
                **kwargs,
            )

            parent_index = kwargs.pop("parent_index", None)
            if parent_index is not None:
                if parent_index == self.index_name:
                    raise RuntimeError(
                        f"not allow index inherit the same index: "
                        f"{parent_index} == {self.index_name}"
                    )

                self.check_allow_inherit(parent_index, mapping)
                mapping["settings"]["index"]["knn.vector_codebook_inherit_from"] = (
                    parent_index
                )
                mapping["settings"]["index"]["knn.offline.construction"] = True

            self.client.indices.create(index=self.index_name, body=mapping)

            if parent_index is not None:
                # trigger new vector table
                embedding = self.embedding.embed_query("demo")
                if embedding is None:
                    raise RuntimeError(
                        "embedding server return None while embedding query: demo"
                    )
                routing_field = kwargs.get("routing_field", None)
                routing = "0"
                return_ids = _bulk_ingest_embeddings(
                    self.client,
                    self.index_name,
                    [embedding],
                    ["demo"],
                    metadatas=(
                        None if routing_field is None else [{routing_field: routing}]
                    ),
                    ids=["demo_id__"],
                    routing_field=routing_field,
                )
                self.refresh()
                del_rsp = self.delete(ids=return_ids, **kwargs)
                logger.info(f"id {return_ids}, del {del_rsp}")
            elif self.is_route_index and method_name == "ivfpq":
                self.ivfpq_trained = False

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        response = self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"terms": {"_id": ids}}},
            refresh=True,
            conflicts="proceed",
        )
        return response

    async def adelete_by_query(
        self,
        body: Any,
        params: Any = None,
        headers: Any = None,
    ) -> Any:
        return await run_in_executor(None, self.delete_by_query, body, params, headers)

    def delete_by_query(
        self,
        body: Any,
        params: Any = None,
        headers: Any = None,
    ) -> Any:
        def delete_route_by_query(route: str, body: Any, params: Any) -> Any:
            routing_field = self.kwargs.get("routing_field", "routing_field")
            term = {f"metadata.{routing_field}.keyword": f"{route}"}
            must_clauses = [{"term": term}, body["query"]]
            body = {"query": {"bool": {"must": must_clauses}}}

            logger.info("body: {}".format(body))
            return self.client.delete_by_query(
                index=self.index_name, body=body, params=params
            )

        if "routing" in params:
            return delete_route_by_query(params["routing"], body, params)
        logger.info("body: {}".format(body))
        return self.client.delete_by_query(
            index=self.index_name, body=body, params=params, headers=headers
        )

    def delete_index(self, index_name: Optional[str] = None) -> Optional[bool]:
        """Deletes a given index from vectorstore."""
        if index_name is None:
            if self.index_name is None:
                raise ValueError("index_name must be provided.")
            index_name = self.index_name
        try:
            self.client.indices.delete(index=index_name)
            return True
        except Exception as e:
            raise e

    def index_exists(self, index_name: Optional[str] = None) -> Optional[bool]:
        """If given index present in vectorstore, returns True else False."""
        if index_name is None:
            if self.index_name is None:
                raise ValueError("index_name must be provided.")
            index_name = self.index_name

        return self.client.indices.exists(index=index_name)

    def start_working_threads(
        self, embed_thread_num: int = 2, write_thread_num: int = 4, **kwargs: Any
    ) -> None:
        if len(self.embed_jobs) == 0 and len(self.write_jobs) == 0:
            self.keyboard_stop_event.clear()
            for _ in range(embed_thread_num):
                embed_t = Thread(
                    target=embedding_thread,
                    args=(
                        self.embedding,
                        self.texts_queue,
                        self.embeddings_queue,
                        self.keyboard_stop_event,
                        self.dimension,
                    ),
                    daemon=True,
                )
                embed_t.start()
                self.embed_jobs.append(embed_t)

            if kwargs.get("index_name") is None:
                kwargs["index_name"] = self.index_name
            self.write_jobs = []
            for _ in range(write_thread_num):
                write_t = Thread(
                    target=bulk_ingest_embedding_thread,
                    args=(
                        self.embeddings_queue,
                        self.return_ids_queue,
                        self.keyboard_stop_event,
                        self.client,
                    ),
                    kwargs=kwargs,
                    daemon=True,
                )
                write_t.start()
                self.write_jobs.append(write_t)

    def stop_working_threads(self) -> None:
        self.keyboard_stop_event.set()
        if self.embed_jobs:
            for t in self.embed_jobs:
                t.join()
        if self.write_jobs:
            for t in self.write_jobs:
                t.join()
        self.embed_jobs = []
        self.write_jobs = []

    def check_alive_working_thread(self) -> None:
        embed_thread_alive_num = 0
        for t in self.embed_jobs:
            if t.is_alive():
                embed_thread_alive_num += 1
        logger.info(
            f"embed thread: alive {embed_thread_alive_num},"
            f" total {len(self.embed_jobs)}"
        )

        write_thread_alive_num = 0
        for t in self.write_jobs:
            if t.is_alive():
                write_thread_alive_num += 1
        logger.info(
            f"write thread: alive {write_thread_alive_num}, "
            f"total {len(self.write_jobs)}"
        )

    def train_ivfpq_index_with_routing(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> None:
        logger.info("start train ivfpq ...")
        kwargs = kwargs or self.kwargs

        # insert train data
        kwargs["method_name"] = "ivfpq"
        if not kwargs.get("routing_field"):
            raise RuntimeError("Using ivfpq, but routing field is not specified!")
        logger.info(
            f"Init ivfpq index with routing field [{kwargs.get('routing_field')}]"
        )

        least_data_num = self.kwargs.get("nlist", 1000) * 30

        if len(texts) <= least_data_num:
            self.keyboard_stop_event.set()
            self.delete_index(self.index_name)
            raise RuntimeError(
                f"train data [{len(texts)}] is too little, at least [{least_data_num}]"
            )
        logger.info(f"ivfpq train data num: {least_data_num}")

        self.__add_texts(
            texts=texts[:least_data_num],
            metadatas=metadatas[:least_data_num] if metadatas else None,
            ids=ids[:least_data_num] if ids else None,
            bulk_size=bulk_size,
        )

        def build_ivfpq_index(index_name: str, field_name: str) -> Any:
            body = {
                "indexName": index_name,
                "fieldName": field_name,
                "removeOldIndex": "true",
                "ivf_train_only": "true",
            }

            response = self.client.transport.perform_request(
                method="POST",
                url="/_plugins/_vector/index/build",
                body=json.dumps(body),
            )
            # response = {'payload': ['default_vector_test1_my_vector']}
            return response

        def check_ivfpq_task(index_name: str, field_name: str) -> Any:
            logger.info("start check ivfpq task ...")
            body = {
                "indexName": index_name,
                "fieldName": field_name,
                "taskIds": '["default_' + index_name + "_" + field_name + '"]',
            }

            max_retries = kwargs.get("train_ivfpq_timeout", 600) / 10
            while True:
                max_retries -= 1
                if max_retries < 0:
                    self.delete_index(self.index_name)
                    raise RuntimeError(
                        "check ivfpq task terminated because of timeout!"
                    )

                response = self.client.transport.perform_request(
                    method="GET",
                    url="/_plugins/_vector/index/tasks",
                    body=json.dumps(body),
                )

                logger.info(response)
                if "finish building" in response.get("payload", [""])[0]:
                    break
                time.sleep(10)

            logger.info("finish check ivfpq task ...")
            return response

        def reserve_ivfpq_codebook(index_name: str) -> Any:
            logger.info("start reserve ivfpq codebook ...")
            response = self.client.transport.perform_request(
                method="POST",
                url="/_truncate/" + index_name,
                params={"reserve_codebook": "true"},
            )
            logger.info("finish reserve ivfpq codebook ...")
            return response

        # train
        index_name = kwargs.get("index_name") or self.index_name
        field_name = kwargs.get("vector_field", "vector_field")
        build_ivfpq_index(index_name, field_name)

        # check
        check_ivfpq_task(index_name, field_name)

        # reserve
        reserve_ivfpq_codebook(index_name)

        self.total_done = 0
        self.ivfpq_trained = True
        logger.info("finish train ivfpq ...")

    def refresh(self) -> None:
        self.client.transport.perform_request(
            method="POST",
            url=f"/{self.index_name}/_refresh",
        )

    def __filter_existed_ids(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 1024,
    ) -> Tuple[List[str], Optional[List[dict]], Optional[List[str]]]:
        """
        filter text when id existed
        :param texts:
        :param metadatas:
        :param ids:
        :return:
        """

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
        def __fetch_existing_ids(batch_ids: List[str]) -> Set[str]:
            try:
                existing_docs = self.client.mget(
                    index=self.index_name, body={"ids": batch_ids}, _source=False
                )
                return {doc["_id"] for doc in existing_docs["docs"] if doc["found"]}
            except Exception as e:
                logger.error(f"Error fetching batch {batch_ids}: {e}")
                return set()

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
        def __fetch_existing_routing_ids(
            batch_ids: List[str], route_ids: List[str]
        ) -> Set[str]:
            try:
                existing_docs = self.client.mget(
                    body={
                        "docs": [
                            {"_index": self.index_name, "_id": id, "routing": routing}
                            for id, routing in zip(batch_ids, route_ids)
                        ]
                    },
                    _source=False,
                )
                return {doc["_id"] for doc in existing_docs["docs"] if doc["found"]}
            except Exception as e:
                logger.error(f"Error fetching batch {batch_ids}: {e}")
                return set()

        if ids is None:
            return texts, metadatas, ids

        if len(texts) != len(ids):
            raise RuntimeError(f"texts {len(texts)} != {ids}")

        if self.is_route_index and metadatas is None:
            raise RuntimeError(
                "route_index need metadatas's routing field, but metadatas is None"
            )

        filtered_texts = []
        filtered_metadatas = []
        filtered_ids = []

        def batch(iterable: Any, n: int) -> Any:
            length = len(iterable)
            for idx in range(0, length, n):
                yield iterable[idx : min(idx + n, length)]

        for ids_batch, texts_batch, metadatas_batch in zip(
            batch(ids, bulk_size),
            batch(texts, bulk_size),
            (
                batch(metadatas, bulk_size)
                if metadatas is not None
                else batch([None] * len(ids), bulk_size)
            ),
        ):
            if self.is_route_index:
                routing = self.kwargs.get("routing_field")
                existing_ids_set = __fetch_existing_routing_ids(
                    ids_batch, [meta[routing] for meta in metadatas_batch]
                )
            else:
                existing_ids_set = __fetch_existing_ids(ids_batch)

            for text, metadata, doc_id in zip(texts_batch, metadatas_batch, ids_batch):
                if doc_id not in existing_ids_set:
                    filtered_texts.append(text)
                    filtered_ids.append(doc_id)
                    if metadatas is not None:
                        filtered_metadatas.append(metadata)

        return (
            filtered_texts,
            metadatas if metadatas is None else filtered_metadatas,
            filtered_ids,
        )

    def __add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
    ) -> Any:
        texts = list(texts)
        total_items = len(texts)
        if total_items == 0:
            logger.warning("Texts size is zero!")
            return []

        if not self.kwargs.get("overwrite", False):
            texts, metadatas, ids = self.__filter_existed_ids(texts, metadatas, ids)
            logger.info(
                f"after _id filter, texts num change from {total_items} => {len(texts)}"
            )

            total_items = len(texts)
            if len(texts) == 0:
                logger.info("All texts existed, Finish")
                return ids

        for offset in range(0, total_items, bulk_size):
            texts_bulk = texts[offset : offset + bulk_size]
            metadatas_bulk = (
                metadatas[offset : offset + bulk_size] if metadatas else None
            )
            ids_bulk = ids[offset : offset + bulk_size] if ids else None
            self.texts_queue.put([texts_bulk, ids_bulk, metadatas_bulk])

        return_ids = []
        while not self.keyboard_stop_event.is_set():
            try:
                # main thread will block util all task done or receive KeyboardInterrupt
                return_ids_bulk = self.return_ids_queue.get(timeout=1)
                return_ids += return_ids_bulk
            except queue.Empty:
                # when call add_texts in a loop, and use KeyboardInterrupt
                # to stop a iteration, some residual data from last iteration
                # may not read in time, so here use '>='
                if len(return_ids) >= total_items:
                    logger.info("All task done.")
                    break
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt, interrupt current task...")
                # clean existing task. However, data in return_ids_queue is successfully
                # written into lsearch, it is not necessary to clear it.
                self.keyboard_stop_event.set()
                with self.texts_queue.mutex:
                    self.texts_queue.queue.clear()
                with self.embeddings_queue.mutex:
                    self.embeddings_queue.queue.clear()

                # close current progress bar to avoid being triggerred twice or more
                # time in notebook.
                raise KeyboardInterrupt(
                    "Received KeyboardInterrupt, interrupt current task..."
                )
        self.refresh()
        return return_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        """
        self.init_index(**self.kwargs)
        self.start_working_threads(**self.kwargs)
        # If using ivfpq, train data when first called
        texts_: Sequence[str] = (
            texts if isinstance(texts, (list, tuple)) else list(texts)
        )
        bulk_size = kwargs.get("bulk_size", 500)
        if not self.ivfpq_trained:
            self.train_ivfpq_index_with_routing(
                texts=texts_,
                metadatas=metadatas,
                ids=ids,
                bulk_size=bulk_size,
                **self.kwargs,
            )

        return self.__add_texts(
            texts=texts_, metadatas=metadatas, ids=ids, bulk_size=bulk_size
        )

    def similarity_search(
        self, query: str = "", k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

            metadata_field: Document field that metadata is stored in. Defaults to
            "metadata".
            Can be set to a special value "*" to include the entire document.

        Optional Args for Approximate Search:
            search_type: "approximate_search"; Defaults to "approximate_search".

            hybrid: Whether to use hybrid retrieval, in which rrf and multi-filter
            can be used. Defaults to False.

            rrf_rank_constant: Coefficient to control rank in rrf. Defaults to "60".

            filter: Filter data according to conditions, the execution order depends
            on the filter_type. When passing a List[Dict], each element in the list
            corresponds to a filter condition, and currently it should only be used
            when hybrid = True. Defaults to None.

            filter_type: "pre_filter", "post_filter". Default to "post_filter" once
            filter is set. Otherwise, default to None.

            min_score: Similarity threshold, requiring the returned vector score to
            be greater than this value. Defaults to "0.0".

        """
        docs_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc[0] for doc in docs_with_scores]

    async def asimilarity_search(
        self, query: str = "", k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await run_in_executor(None, self.similarity_search, query, k=k, **kwargs)

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance.

        Returns:
            List of Tuples of (doc, similarity_score)
        """

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None, self.similarity_search_with_score, *args, **kwargs
        )

    def similarity_search_with_score(
        self, query: str = "", k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to query.

        Only supports Approximate Search now.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        if "multi_query_rrf" in kwargs:
            return self._multi_query_similarity_with_score(**kwargs)

        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        index_name = kwargs.get("index_name", self.index_name)
        params = {}
        routing = kwargs.get("routing", None)
        if routing is not None:
            params["routing"] = str(routing)

        search_query = self._get_search_query(query, k, **kwargs)

        logger.debug(f"search query:{search_query}")
        response = self.client.search(
            index=index_name, body=search_query, params=params
        )
        return self._process_response(response, text_field, metadata_field)

    def _get_search_query(self, query: str, k: int = 4, **kwargs: Any) -> Dict:
        search_type = kwargs.get("search_type", "approximate_search")
        text_field = kwargs.get("text_field", "text")
        hybrid = kwargs.get("hybrid", False)
        filter = kwargs.get("filter", None)

        routing = kwargs.get("routing", None)

        if search_type == "approximate_search":
            embedding = self.embedding.embed_query(query)
            if embedding is None:
                raise RuntimeError(
                    f"embedding server return None while embedding query: {query}"
                )

            match_text = kwargs.get("match_text", query)
            vector_field = kwargs.get("vector_field", "vector_field")
            rrf_rank_constant = kwargs.get("rrf_rank_constant", "60")
            min_score = kwargs.get("min_score", "0.0")
            filter_type = kwargs.get("filter_type", None)
            ef_search = kwargs.get("ef_search", None)
            nprobe = kwargs.get("nprobe", None)
            reorder_factor = kwargs.get("reorder_factor", None)
            client_refactor = kwargs.get("client_refactor", None)
            rrf_window_size = kwargs.get("rrf_window_size", None)

            search_query = self._default_approximate_search_query(
                query_vector=embedding,
                k=k,
                vector_field=vector_field,
                text_field=text_field,
                hybrid=hybrid,
                rrf_rank_constant=rrf_rank_constant,
                match_text=match_text,
                filter=filter,
                filter_type=filter_type,
                min_score=min_score,
                ef_search=ef_search,
                nprobe=nprobe,
                reorder_factor=reorder_factor,
                client_refactor=client_refactor,
                rrf_window_size=rrf_window_size,
                routing=routing,
            )

        elif search_type == "text_search":
            must = kwargs.get("must", None)
            must_not = kwargs.get("must_not", None)
            should = kwargs.get("should", None)
            minimum_should_match = kwargs.get("minimum_should_match", 0)

            search_query = self._default_text_search_query(
                query_text=query,
                k=k,
                text_field=text_field,
                must=must,
                must_not=must_not,
                should=should,
                minimum_should_match=minimum_should_match,
                filter=filter,
                routing=routing,
            )
        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        return search_query

    def _multi_query_similarity_with_score(
        self, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        multi_kwargs = kwargs.get("multi_query_rrf", [])
        search_queries = ""
        for single_kwargs in multi_kwargs:
            query = single_kwargs.pop("query", None)
            k = single_kwargs.pop("k", 4)
            routing = single_kwargs.get("routing", None)

            single_index_args = {"index": self.index_name}
            if routing is not None:
                single_index_args = {"index": self.index_name, "routing": routing}

            single_query_args = self._get_search_query(query, k, **single_kwargs)
            if "ext" in single_query_args and "lvector" in single_query_args["ext"]:
                single_query_args["ext"]["lvector"]["hybrid_search_type"] = (
                    "multi_filter_rrf"
                )

            search_queries += json.dumps(single_index_args) + "\n"
            search_queries += json.dumps(single_query_args) + "\n"

        if len(search_queries) == 0:
            raise RuntimeError("at lease need one query")

        response = self.client.transport.perform_request(
            method="POST", url="/_msearch_rrf", body=search_queries
        )

        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        return self._process_response(response, text_field, metadata_field)

    @staticmethod
    def _process_response(
        response: dict, text_field: str, metadata_field: str
    ) -> List[Any]:
        hits = [hit for hit in response["hits"]["hits"]]
        documents_with_scores = [
            (
                (
                    Document(
                        id=hit["_id"],
                        page_content=hit["_source"][text_field],
                        metadata=(
                            hit["_source"]
                            if metadata_field == "*"
                            or metadata_field not in hit["_source"]
                            else hit["_source"][metadata_field]
                        ),
                    ),
                    hit["_score"],
                )
                if "_rank_from" not in hit
                else (
                    Document(
                        id=hit["_id"],
                        page_content=hit["_source"][text_field],
                        metadata=(
                            hit["_source"]
                            if metadata_field == "*"
                            or metadata_field not in hit["_source"]
                            else hit["_source"][metadata_field]
                        ),
                    ),
                    hit["_score"],
                    hit["_rank_from"],
                )
            )
            for hit in hits
        ]
        return documents_with_scores

    def _default_approximate_search_query(
        self,
        query_vector: List[float],
        k: int = 4,
        vector_field: str = "vector_field",
        text_field: str = "text",
        hybrid: bool = False,
        rrf_rank_constant: str = "60",
        match_text: str = "",
        filter: Optional[List[Dict]] = None,
        filter_type: Optional[str] = None,
        min_score: str = "0.0",
        ef_search: Optional[str] = None,  # only for hnsw
        nprobe: Optional[str] = None,  # "2000"
        reorder_factor: Optional[str] = None,  # "20"
        client_refactor: Optional[str] = None,  # "true"
        rrf_window_size: Optional[str] = None,
        routing: Optional[str] = None,
    ) -> Dict:
        """For Approximate k-NN Search, this is the default query."""

        if filter is not None:
            filter_type = "post_filter" if filter_type is None else filter_type
            if not isinstance(filter, list):
                raise RuntimeError(f"unexpected filter with {type(filter)}")

        if not hybrid:
            # Doing search with vector
            final_ext: Any = {"lvector": {}}
            if min_score != "0.0":
                final_ext["lvector"]["min_score"] = min_score
            if ef_search:
                final_ext["lvector"]["ef_search"] = ef_search
            if nprobe:
                final_ext["lvector"]["nprobe"] = nprobe
            if reorder_factor:
                final_ext["lvector"]["reorder_factor"] = reorder_factor
            if client_refactor:
                final_ext["lvector"]["client_refactor"] = client_refactor

            search_query: Dict = {
                "size": k,
                "_source": True,  # force return '_source'
                "query": {"knn": {vector_field: {"vector": query_vector, "k": k}}},
            }

            if filter is not None:
                # when using filter, transform filter from List[Dict] to Dict
                afilter = {"bool": {"must": filter}} if len(filter) > 1 else filter[0]
                # filter should be Dict
                search_query["query"]["knn"][vector_field]["filter"] = afilter
                if filter_type:
                    final_ext["lvector"]["filter_type"] = filter_type

            if final_ext != {"lvector": {}}:
                search_query["ext"] = final_ext

        else:
            must_clauses = [{"match": {text_field: match_text}}]
            if routing is not None:
                routing_field = self.kwargs.get("routing_field", "routing_field")
                must_clauses.append(
                    {"term": {f"metadata.{routing_field}.keyword": routing}}
                )
            final_filter: Any = None
            if filter is not None:
                # Doing rrf search with full text, vector and filter.
                # use two bool expression to do rrf and filter respectively
                final_filter = {
                    "bool": {
                        "must": [
                            {"bool": {"must": must_clauses}},
                            {"bool": {"filter": filter}},  # filter should be List[Dict]
                        ]
                    }
                }
                final_ext = {
                    "lvector": {
                        "filter_type": filter_type,
                        "hybrid_search_type": "filter_rrf",
                        "rrf_rank_constant": rrf_rank_constant,
                    }
                }
            else:
                # Doing rrf search with full text and vector.
                final_filter = {"bool": {"must": must_clauses}}
                final_ext = {
                    "lvector": {
                        "hybrid_search_type": "filter_rrf",
                        "rrf_rank_constant": rrf_rank_constant,
                    }
                }
            if rrf_window_size:
                final_ext["lvector"]["rrf_window_size"] = rrf_window_size
            if min_score != "0.0":
                final_ext["lvector"]["min_score"] = min_score
            if ef_search:
                final_ext["lvector"]["ef_search"] = ef_search
            if nprobe:
                final_ext["lvector"]["nprobe"] = nprobe
            if reorder_factor:
                final_ext["lvector"]["reorder_factor"] = reorder_factor
            if client_refactor:
                final_ext["lvector"]["client_refactor"] = client_refactor

            search_query = {
                "size": k,
                "_source": True,
                "query": {
                    "knn": {
                        vector_field: {
                            "vector": query_vector,
                            "filter": final_filter,
                            "k": k,
                        }
                    },
                },
                "ext": final_ext,
            }

        return search_query

    def _default_text_search_query(
        self,
        query_text: str,
        k: int = 4,
        text_field: str = "text",
        must: Optional[List[Dict]] = None,
        must_not: Optional[List[Dict]] = None,
        should: Optional[List[Dict]] = None,
        minimum_should_match: int = 0,
        filter: Optional[List[Dict]] = None,
        routing: Optional[str] = None,
    ) -> Dict:
        query_clause: Any = None
        if routing is not None:
            routing_field = self.kwargs.get("routing_field", "routing_field")
            query_clause = {
                "bool": {
                    "must": [
                        {"match": {text_field: query_text}},
                        {"term": {f"metadata.{routing_field}.keyword": routing}},
                    ]
                }
            }
        else:
            query_clause = {"match": {text_field: query_text}}
        # build the simplest search_query when only query_text is specified
        if not must and not must_not and not should and not filter:
            search_query = {"size": k, "query": query_clause}
            return search_query

        # build complex search_query when either of must/must_not/should/filter
        if must:
            if not isinstance(must, list):
                raise RuntimeError(f"unexpected [must] clause with {type(filter)}")
            if query_clause not in must:
                must.append(query_clause)
        else:
            must = [query_clause]

        boolean_query: Dict = {"must": must}

        if must_not:
            if not isinstance(must_not, list):
                raise RuntimeError(f"unexpected [must_not] clause with {type(filter)}")
            boolean_query["must_not"] = must_not

        if should:
            if not isinstance(should, list):
                raise RuntimeError(f"unexpected [should] clause with {type(filter)}")
            boolean_query["should"] = should
            if minimum_should_match != 0:
                boolean_query["minimum_should_match"] = minimum_should_match

        if filter:
            if not isinstance(filter, list):
                raise RuntimeError(f"unexpected [filter] clause with {type(filter)}")
            boolean_query["filter"] = filter

        search_query = {"size": k, "query": {"bool": boolean_query}}

        return search_query

    def get_by_ids(self, ids: Sequence[str]) -> List[Any]:
        query_body = {
            "query": {"terms": {"_id": ids}},
            "_source": True,
            "size": len(ids),
        }

        docs = self.client.search(index=self.index_name, body=query_body, pretty=True)

        text_field = self.kwargs.get("text_field", "text")
        metadata_field = self.kwargs.get("metadata_field", "metadata")
        documents_with_scores = [
            (
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=(
                        hit["_source"]
                        if metadata_field == "*" or metadata_field not in hit["_source"]
                        else hit["_source"][metadata_field]
                    ),
                ),
                hit["_score"],
            )
            for hit in docs["hits"]["hits"]
        ]
        return documents_with_scores

    def check_allow_inherit(self, parent_index: str, mapping: Dict) -> Any:
        response = self.client.transport.perform_request(
            method="GET", url=f"/{parent_index}/_mapping?pretty"
        )
        new_vector_field = mapping["mappings"]["properties"]["vector_field"]
        parent_vector_field = response[parent_index]["mappings"]["properties"][
            "vector_field"
        ]
        if new_vector_field["type"] != parent_vector_field["type"]:
            raise RuntimeError(
                f"vector type {new_vector_field['type']} != "
                f"{parent_vector_field['type']}"
            )
        elif new_vector_field["dimension"] != parent_vector_field["dimension"]:
            raise RuntimeError(
                f"dimension {new_vector_field['dimension']} != "
                f"{parent_vector_field['dimension']}"
            )
        elif (
            new_vector_field["method"]["name"] != parent_vector_field["method"]["name"]
        ):
            raise RuntimeError(
                f"method_name {new_vector_field['method']['name']} != "
                f"{parent_vector_field['method']['name']}"
            )
        elif (
            new_vector_field["method"]["space_type"]
            != parent_vector_field["method"]["space_type"]
        ):
            raise RuntimeError(
                f"space_type {new_vector_field['method']['space_type']} != "
                f"{parent_vector_field['method']['space_type']}"
            )

        new_vector_parameters = new_vector_field["method"]["parameters"]
        parent_vector_parameters = parent_vector_field["method"]["parameters"]
        if new_vector_parameters["nlist"] != parent_vector_parameters["nlist"]:
            raise RuntimeError(
                f"nlist {new_vector_parameters['nlist']} != "
                f"{parent_vector_parameters['nlist']}"
            )
        elif (
            new_vector_parameters["centroids_use_hnsw"]
            != parent_vector_parameters["centroids_use_hnsw"]
        ):
            raise RuntimeError(
                f"nlist {new_vector_parameters['centroids_use_hnsw']} != "
                f"{parent_vector_parameters['centroids_use_hnsw']}"
            )
        elif new_vector_parameters["m"] != parent_vector_parameters["m"]:
            raise RuntimeError(
                f"nlist {new_vector_parameters['m']} != "
                f"{parent_vector_parameters['m']}"
            )
        elif new_vector_parameters["centroids_use_hnsw"]:
            if (
                new_vector_parameters["centroids_hnsw_ef_construct"]
                != parent_vector_parameters["centroids_hnsw_ef_construct"]
            ):
                raise RuntimeError(
                    f"nlist {new_vector_parameters['centroids_hnsw_ef_construct']} != "
                    f"{parent_vector_parameters['centroids_hnsw_ef_construct']}"
                )
            elif (
                new_vector_parameters["centroids_hnsw_m"]
                != parent_vector_parameters["centroids_hnsw_m"]
            ):
                raise RuntimeError(
                    f"nlist {new_vector_parameters['centroids_hnsw_m']} != "
                    f"{parent_vector_parameters['centroids_hnsw_m']}"
                )

        response = self.client.transport.perform_request(
            method="GET", url=f"/{parent_index}/_settings?pretty"
        )
        new_index_field = mapping["settings"]["index"]
        parent_index_field = response[parent_index]["settings"]["index"]
        if "knn_routing" in parent_index_field:
            parent_index_knn_routing = (
                True
                if parent_index_field.get("knn_routing", "false") == "true"
                else False
            )
            if parent_index_knn_routing != new_index_field.get("knn_routing", False):
                raise RuntimeError(
                    f"knn_routing {new_index_field.get('knn_routing')} != "
                    f"{parent_index_knn_routing}"
                )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: int = 500,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LindormVectorStore:
        """Construct LindormSearchStore wrapper from raw texts.

            Example:
            .. code-block:: python

            from langchain_community.vectorstores import LindormSearchStore
            from langchain_community.embeddings import LindormAIEmbeddings
            embeddings = LindormAIEmbeddings(
                endpoint='http://{instanceid}-proxy-ml.lindorm.rds.aliyuncs.com:9002',
                 username='root',
                 password='xxx',
                 model_name='emb_model'
                 )
            lindorm_vector_search = LindormSearchStore.from_texts(
                texts,
                embeddings,
                lindorm_search_url="http://{instanceid}-proxy-search-vpc.lindorm.aliyuncs.com:30070"
            )

        LindormSearch currently by default supports Approximate Search powered by
        lvector engines.

            Optional Args:
                vector_field: Document field embeddings are stored in.
                Defaults to "vector_field".

                text_field: Document field the text of the document is stored in.
                Defaults to "text".

            Optional Keyword Args for Approximate Search:
                method_name: "flat", "hnsw", "ivfpq"; default: "hnsw"

                engine: "lvector"; default: "lvector"

                space_type: "l2", "cosinesimil", "innerproduct"; default: "cosinesimil"

                m: Between 2 and 32768; default: 16

                nlist:
                    Number of cluster centersdefault. Between 2 and 1000000;
                    default:1000

                centroids_use_hnsw:
                    Whether to use the HNSW algorithm when searching for cluster centers
                    default: False

                centroids_hnsw_m:
                    Between 1 and 100; default: 16

                centroids_hnsw_ef_search:
                    Size of the dynamic list used during KNN-search
                    Higher values lead to more accurate but slower searches;
                    default: 100

                centroids_hnsw_ef_construct:
                     Size of the dynamic list used during k-NN graph
                     creation.Higher values lead to more accurate graph but slower
                     indexing speed;
                     default: 100
        """
        lindorm_search_url = kwargs.pop("lindorm_search_url")
        index_name = kwargs.pop("index_name")

        lindorm_search_store = cls(lindorm_search_url, index_name, embedding, **kwargs)

        if len(texts) == 0:
            logger.warning("Texts size is zero!")
            return lindorm_search_store

        lindorm_search_store.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, bulk_size=bulk_size
        )
        return lindorm_search_store
