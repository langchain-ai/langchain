from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.utils import maximal_marginal_relevance

try:
    import vdms
except ImportError:
    raise ImportError(
        "Could not import vdms python package. "
        "Please install it with `pip install vdms`."
    )


class DistanceMetric(str, Enum):
    """
    Enumerator of the Distance strategies for calculating distances
    between vectors.
    """

    L2 = "L2"  # Euclidean Distance
    IP = "IP"  # Inner Product


class IndexEngine(str, Enum):
    """
    Enumerator of the underlying implementation for indexing
    """

    TileDBDense = "TileDBDense"
    TileDBSparse = "TileDBSparse"
    FaissFlat = "FaissFlat"
    FaissIVFFlat = "FaissIVFFlat"
    Flinng = "Flinng"


DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_DISTANCE_METRIC = DistanceMetric.L2
DEFAULT_INSERT_BATCH_SIZE = 100
DEFAULT_K = 3  # Number of Documents to return.
INVALID_DOC_METADATA_KEYS = ["_distance", "content", "blob"]
INVALID_METADATA_VALUE = ["Missing property", None, {}]
DEFAULT_PROPERTIES = ["_distance", "id", "content"]
DEFAULT_SEARCH_ENGINE = IndexEngine.FaissFlat
DEFAULT_VDMS_CONNECTION = {"host": "localhost", "port": "55555"}
logger = logging.getLogger(__name__)


class VDMS(VectorStore):
    """Wrapper around Intel Lab's VDMS for vector-store workloads.

    To use, you should have both:
    - the ``vdms`` python package installed
    - a host (str) and port (int) associated with a deployed VDMS Server

    Visit https://github.com/IntelLabs/vdms/wiki more information.

    Args:
        collection_name: Name of data collection [Default: langchain]
        distance_strategy: Method used to calculate distances. VDMS supports
            "L2" (euclidean distance) or "IP" (inner product) [Default: L2]
        engine: Underlying implementation for indexing and computing distances.
            VDMS supports TileDBDense, TileDBSparse, FaissFlat, FaissIVFFlat,
            and Flinng [Default: FaissFlat]
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        connection_args: Dictionary defining "host" and "port" for VDMS server
        relevance_score_fn: Function for obtaining relevance score

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import VDMS
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
            import vdms

            vectorstore = VDMS(
                collection_name="langchain_store",
                distance_strategy="L2",
                engine="FaissFlat"
                embedding_function=HuggingFaceEmbeddings(),
                connection_args={"host": "localhost", "port": 55555},
            )
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,  # DescriptorSet name
        distance_strategy: Union[DistanceMetric, str] = DEFAULT_DISTANCE_METRIC,
        engine: Union[IndexEngine, str] = DEFAULT_SEARCH_ENGINE,
        connection_args: dict[str, Any] = DEFAULT_VDMS_CONNECTION,
        embedding_function: Embeddings = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        # Connect to VDMS Server
        self._create_connection_alias(connection_args)

        # Check required parameters
        self.similarity_search_engine = engine
        self.distance_strategy = distance_strategy
        self._embedding_function = embedding_function
        self.check_required_inputs()

        # Update other parameters
        self.override_relevance_score_fn = relevance_score_fn

        # Initialize collection
        self._collection_name = self.add_collection_set(
            collection_name,
            engine=self.similarity_search_engine,
            metric=self.distance_strategy,
        )

    def _create_connection_alias(self, connection_args: dict) -> Any:
        if connection_args is None:
            connection_args = DEFAULT_VDMS_CONNECTION
        elif "host" not in connection_args:
            connection_args["host"] = DEFAULT_VDMS_CONNECTION["host"]
        elif "port" not in connection_args:
            connection_args["port"] = DEFAULT_VDMS_CONNECTION["port"]

        self._client = vdms.vdms()
        self._client.connect(connection_args["host"], int(connection_args["port"]))

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy.lower() == "ip":
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy.lower() == "l2":
            # Default behavior is to use euclidean distance relevancy
            return self._euclidean_relevance_score_fn
        # elif self.distance_strategy.lower() == "cosine":
        #     return self._cosine_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product,"
                " or euclidean"
            )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        # Pop score threshold so that only relevancy scores, not raw scores, are
        # filtered.
        # relevance_score_fn = self._select_relevance_score_fn()
        if self.override_relevance_score_fn is None:
            # raise ValueError(
            #     "normalize_score_fn must be provided to"
            #     " VDMS constructor to normalize scores"
            # )
            kwargs["normalize_distance"] = True
        docs_and_scores = self.similarity_search_with_score(
            query,
            k,
            filter,
            # where_document,
            **kwargs,
        )

        docs_and_rel_scores = []
        for doc, score in docs_and_scores:
            if self.override_relevance_score_fn is None:
                docs_and_rel_scores.append((doc, score))
            else:
                docs_and_rel_scores.append(
                    (doc, self.override_relevance_score_fn(score))
                )
        return docs_and_rel_scores

    def add_collection(
        self,
        collection_name: str,
        metadatas: Optional[List] = None,
        embeddings: Optional[List] = None,
        documents: Optional[List] = None,
        ids: Optional[List] = None,
    ) -> None:
        # all_queries = []
        # all_blobs = []

        lengths = [
            len(x)
            for x in [metadatas, embeddings, documents, ids]
            if x is not None and len(x) > 0
        ]
        min_len = min(lengths)

        if metadatas is None:
            metadatas = [None] * min_len
        if embeddings is None:
            embeddings = [None] * min_len
        if documents is None:
            documents = [None] * min_len
        if ids is None:
            # ids = [None] * min_len
            ids = [str(uuid.uuid1()) for _ in range(min_len)]

        for meta, emb, doc, id in zip(metadatas, embeddings, documents, ids):
            all_queries = []
            all_blobs = []
            query, blob = self.get_add_collection_query(
                collection_name, metadata=meta, embedding=emb, document=doc, id=id
            )

            if blob is not None:
                all_queries.append(query)
                all_blobs.append(blob)

                response, response_array = self._client.query(all_queries, all_blobs)
            # self.print_last_response()

    def add_collection_set(
        self,
        collection_name: str,
        engine: Union[IndexEngine, str] = IndexEngine.FaissFlat,
        metric: Union[DistanceMetric, str] = DistanceMetric.L2,
    ) -> str:
        all_queries = []
        query = add_descriptorset(
            "AddDescriptorSet",
            collection_name,
            self._embedding_dimension,
            engine=engine,
            metric=metric,
        )

        all_queries.append(query)
        response, _ = self._client.query(all_queries)

        if "FailedCommand" in response[0]:
            raise ValueError(f"Failed to add collection {collection_name}")

        return collection_name

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more images through the embeddings and add to the vectorstore.

        Args:
            uris List[str]: File path to the image.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added images.
        """
        # Map from uris to blobs
        blobs = [self.encode_image(uri=uri) for uri in uris]

        # Populate IDs
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in uris]

        # Set embeddings
        embeddings = None
        if self._embedding_function is not None and hasattr(
            self._embedding_function, "embed_image"
        ):
            embeddings = self._embedding_function.embed_image(uris=uris)

        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all images
            length_diff = len(uris) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                images_with_metadatas = [uris[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=images_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                images_without_metadatas = [uris[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=images_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=b64_texts,
                ids=ids,
            )
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (List[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        ids = kwargs.get("ids", None)
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids]
                    if embeddings is not None
                    else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self.add_collection(
                        self._collection_name,
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self.add_collection(
                    self._collection_name,
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self.add_collection(
                self._collection_name,
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

    def check_required_inputs(self):
        # Check Distance Metric
        if self.distance_strategy not in sorted(DistanceMetric):
            raise ValueError("distance_strategy must be either 'L2' or 'IP'")

        # Check Engines
        if self.similarity_search_engine not in sorted(IndexEngine):
            raise ValueError(
                "engine must be either 'TileDBDense', 'TileDBSparse', "
                + "'FaissFlat', 'FaissIVFFlat', or 'Flinng'"
            )

        # Check Embedding Func is provided and store dimension size
        if self._embedding_function is None:
            raise ValueError("Must provide embedding function")

        self._embedding_dimension = len(
            self._embedding_function.embed_query("This is a sample sentence.")
        )

    def count(self, collection_name: str) -> int:
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        collection_properties = self.get_collection_properties(collection_name)
        results = {"count": "", "list": collection_properties}
        query = add_descriptor(
            "FindDescriptor",
            collection_name,
            label=None,
            ref=None,
            props=None,
            link=None,
            k_neighbors=None,
            constraints=None,
            results=results,
        )

        all_queries.append(query)

        response, response_array = self._client.query(all_queries, all_blobs)
        # self.print_last_response()
        return response[0]["FindDescriptor"]["returned"]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        constraints: Optional[Dict] = None,
        **kwargs: Any,
    ) -> bool:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        name = collection_name if collection_name is not None else self._collection_name
        return self.delete_collection(name, ids=ids, constraints=constraints)

    def delete_collection(
        self,
        collection_name: str,
        ids: Union[None, List[str]] = None,
        constraints: Union[None, Dict[str, Any]] = None,
    ) -> bool:
        """
        Deletes entire collection if id is not provided
        """
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        collection_properties = self.get_collection_properties(collection_name)
        results = {"list": collection_properties}

        if constraints is None:
            constraints = {"_deletion": ["==", 1]}
        else:
            constraints["_deletion"] = ["==", 1]

        if ids is not None:
            constraints["id"] = ["==", ids if len(ids) > 1 else ids[0]]

        query = add_descriptor(
            "FindDescriptor",
            collection_name,
            label=None,
            ref=None,
            props=None,
            link=None,
            k_neighbors=None,
            constraints=constraints,
            results=results,
        )

        all_queries.append(query)

        response, response_array = self._client.query(all_queries, all_blobs)
        # self.print_last_response()
        return "FindDescriptor" in response[0]

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return f.read()

    @classmethod
    def from_documents(
        cls: Type[VDMS],
        documents: List[Document],
        embedding_function: Embeddings,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,  # Add this line
        connection_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VDMS:
        """Create a VDMS vectorstore from a list of documents.

        Args:
            collection_name (str): Name of the collection to create.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to vectorstore.
            embedding_function (Embeddings): Embedding function. Defaults to None.

        Returns:
            VDMS: VDMS vectorstore.
        """

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_function=embedding_function,
            ids=ids,
            collection_name=collection_name,
            connection_args=connection_args,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[VDMS],
        texts: List[str],
        embedding_function: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        connection_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VDMS:
        """Create a VDMS vectorstore from a raw documents.

        Args:
            texts (List[str]): List of texts to add to the collection.
            embedding_function (Embeddings): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            collection_name (str): Name of the collection to create.

        Returns:
            VDMS: VDMS vectorstore.
        """
        vdms_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
            connection_args=connection_args,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        vdms_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vdms_collection

    def get(
        self,
        collection_name: str,
        constraints: Optional[Dict] = None,
        limit: Optional[int] = None,
        include: List[str] = ["metadata"],
    ) -> Tuple[Any, Any]:
        """Gets the collection.
        Get embeddings and their associate data from the data store.
        If no ids or where filter is provided returns all embeddings up to limit.

        Args:
            ids: The ids of the embeddings to get. Optional.
            where: A Where type dict used to filter results by.
                   E.g. `{"color" : "red", "price": 4.20}`. Optional.
            limit: The number of documents to return. Optional.
            offset: The offset to start returning results from.
                    Useful for paging results with limit. Optional.
            where_document: A WhereDocument type dict used to filter by the documents.
                            E.g. `{$contains: "hello"}`. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.
        """
        # kwargs: Dict[str, Any] = {
        #     "constraints": constraints,
        #     "include": include,
        # }

        # if limit is not None:
        #     kwargs["limit"] = limit

        # return self.get_entries(collection_name, **kwargs)
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        results: Dict[str, Any] = {"count": ""}

        if limit is not None:
            results["limit"] = limit

        # Include metadata
        if "metadata" in include:
            collection_properties = self.get_collection_properties(collection_name)
            results["list"] = collection_properties

        # Include embedding
        if "embeddings" in include:
            results["blob"] = True

        query = add_descriptor(
            "FindDescriptor",
            collection_name,
            label=None,
            ref=None,
            props=None,
            link=None,
            k_neighbors=None,
            constraints=constraints,
            results=results,
        )

        all_queries.append(query)

        response, response_array = self._client.query(all_queries, all_blobs)
        # self.print_last_response()
        return response, response_array

    def get_add_collection_query(
        self,
        collection_name: str,
        metadata: Optional[Any] = None,
        embedding: Optional[Any] = None,
        document: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Union[bytes, None]]:
        current_props = self.get_collection_properties(collection_name)

        props = {} if id is None else {"id": id}
        if metadata:
            # Validate properties (no arrays accepted)
            metadata = validate_vdms_properties(metadata)
            props.update(metadata)
            # for mkey, mval in metadata.items():
            #     props[mkey] = mval
            #     # if mkey not in self.collection_properties:
            #     #     self.collection_properties.append(mkey)
            self.update_collection_properties(
                collection_name, current_props, list(metadata.keys())
            )
        if document:
            props["content"] = document

        # query = add_descriptor("AddDescriptor", collection_name, label=None,
        #                            ref=None, props=props, link=None, k_neighbors=None,
        #                            constraints=None, results=None)
        query = check_vdms_then_adddescriptor(
            self._client,
            collection_name,
            label=None,
            ref=None,
            props=props,
            link=None,
            k_neighbors=None,
        )

        if "FindDescriptor" in list(query.keys()):
            skipped_value = {
                prop_key: prop_val[-1]
                for prop_key, prop_val in query["FindDescriptor"]["constraints"].items()
            }
            print(
                f"[!] Embedding with id ({id}) exists in DB;",
                "Therefore, skipped and not inserted",
            )
            print(f"\tSkipped values are: {skipped_value}")
            return query, None

        # emb = np.array(embedding, dtype="float32")
        # blob = emb.tobytes()

        blob = embedding2bytes(embedding)

        return query, blob

    def get_collection_properties(
        self,
        collection_name: str,
        unique_entity: Optional[bool] = False,
        deletion: Optional[bool] = False,
    ) -> List:
        find_query = find_property_entity(
            collection_name, unique_entity=unique_entity, deletion=deletion
        )
        response, response_blob = self._client.query([find_query])
        if len(response_blob) > 0:
            collection_properties = bytes2str(response_blob[0]).split(",")
        else:
            collection_properties = deepcopy(DEFAULT_PROPERTIES)
        return collection_properties

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding_vector = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            # where_document=where_document,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # if isinstance(embedding[0], float):
        #     embedding = [embedding]
        results = self.query_collection_embeddings(
            query_embeddings=embedding,
            n_results=fetch_k,
            filter=filter,
            # where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        embedding_list = [list(bytes2embedding(result)) for result in results[0][1]]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            # where_document=where_document,
        )
        return docs

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # if isinstance(embedding[0], float):
        #     embedding = [embedding]
        results = self.query_collection_embeddings(
            query_embeddings=embedding,
            n_results=fetch_k,
            filter=filter,
            # where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        embedding_list = [list(bytes2embedding(result)) for result in results[0][1]]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs_and_scores(results)

        selected_results = [
            (r, s) for i, (r, s) in enumerate(candidates) if i in mmr_selected
        ]
        return selected_results

    def query_collection_embeddings(
        self,
        collection_name: Optional[str] = None,
        query_embeddings: Optional[List[float]] = None,
        n_results: int = DEFAULT_K,
        filter: Union[None, Dict[str, Any]] = None,
        results: Union[None, Dict[str, Any]] = None,
        normalize_distance: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        all_responses: List[Any] = []
        # all_queries = []
        # all_blobs = []
        if collection_name is None:
            collection_name = self._collection_name

        if query_embeddings is None:
            return all_responses

        if isinstance(query_embeddings[0], (float, int)):
            query_embeddings = [query_embeddings]

        include = kwargs.get("include", ["metadatas"])
        if results is None and "metadatas" in include:
            collection_properties = self.get_collection_properties(collection_name)
            results = {"list": collection_properties, "blob": "embeddings" in include}

        for qemb in query_embeddings:
            all_queries = []
            all_blobs = []

            query = add_descriptor(
                "FindDescriptor",
                collection_name,
                label=None,
                ref=None,
                props=None,
                link=None,
                k_neighbors=n_results,
                constraints=filter,
                results=results,
            )

            all_queries.append(query)

            # if qemb:
            # emb = np.array(qemb, dtype="float32")
            # blob2 = emb.tobytes()
            blob = embedding2bytes(qemb)
            all_blobs.append(blob)

            if normalize_distance:
                # results["sort"] = "_distance"
                query = add_descriptor(
                    "FindDescriptor",
                    collection_name,
                    label=None,
                    ref=None,
                    props=None,
                    link=None,
                    k_neighbors=1000,
                    constraints=filter,
                    results=results,
                )
                all_queries.append(query)
                all_blobs.append(blob)

            response, response_array = self._client.query(all_queries, all_blobs)
            # self.print_last_response()

            if normalize_distance and len(response) == 2:
                # normalize distances and remove query
                max_dist = max(
                    [
                        ent["_distance"]
                        for ent in response[1]["FindDescriptor"]["entities"]
                    ]
                )
                del response[1]
                if len(response_array) > 0:
                    # TODO implement-> remove blobs
                    pass
                for ent_idx, ent in enumerate(
                    response[0]["FindDescriptor"]["entities"]
                ):
                    ent["_distance"] = ent["_distance"] / max_dist
                    response[0]["FindDescriptor"]["entities"][ent_idx] = ent
            all_responses.append([response, response_array])

        return all_responses

    def run_update_documents(
        self,
        collection_name: str,
        ids: List[str],
        metadatas: Union[None, List[Any]] = None,
        embeddings: Union[None, List[Any]] = None,
        documents: Union[None, List[Any]] = None,
    ) -> None:
        # return ValueError("Cannot update Descriptor; Capability not in VDMS")
        """
        Updates (find, delete, add) a collection based on id.
        If more than one collection returned with id, error occuers
        """
        lengths = [
            len(x)
            for x in [metadatas, embeddings, documents, ids]
            if x is not None and len(x) > 0
        ]
        min_len = min(lengths)

        if metadatas is None:
            metadatas = [None] * min_len
        if embeddings is None:
            embeddings = [None] * min_len
        if documents is None:
            documents = [None] * min_len

        collection_properties = self.get_collection_properties(collection_name)
        for meta, emb, doc, id in zip(metadatas, embeddings, documents, ids):
            results = {"list": collection_properties}

            constraints = {"_deletion": ["==", 1]}

            if id is not None:
                constraints["id"] = ["==", id]

            query = add_descriptor(
                "FindDescriptor",
                collection_name,
                label=None,
                ref=None,
                props=None,
                link=None,
                k_neighbors=None,
                constraints=constraints,
                results=results,
            )

            response, response_array = self._client.query([query])

            query, blob = self.get_add_collection_query(
                collection_name, metadata=meta, embedding=emb, document=doc, id=id
            )
            if blob is not None:
                response, response_array = self._client.query([query], [blob])
                # self.print_last_response()

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with VDMS.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        # normalize_distance=False,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents most similar to the query vector.
        """
        # if isinstance(embedding[0], float):
        #     embedding = [embedding]

        # if normalize_distance:
        #     kwargs["normalize_distance"] = normalize_distance
        results = self.query_collection_embeddings(
            query_embeddings=embedding,
            n_results=k,
            filter=filter,
            # where_document=where_document,
            **kwargs,
        )

        return _results_to_docs(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        # normalize_distance=False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with VDMS with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        # if normalize_distance:
        #     kwargs["normalize_distance"] = normalize_distance

        if self._embedding_function is None:
            raise ValueError("Must provide embedding function")
            # results = self.query_collection_embeddings(
            #     query_texts=[query],
            #     n_results=k,
            #     filter=filter,
            #     where_document=where_document,
            #     **kwargs
            # )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.query_collection_embeddings(
                query_embeddings=query_embedding,
                n_results=k,
                filter=filter,
                # where_document=where_document,
                **kwargs,
            )

        return _results_to_docs_and_scores(results)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        # where_document: Optional[Dict[str, str]] = None,
        # normalize_distance=False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs most similar to embedding vector and similarity score.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        # if normalize_distance:
        kwargs["normalize_distance"] = True

        results = self.query_collection_embeddings(
            query_embeddings=embedding,
            n_results=k,
            filter=filter,
            # where_document=where_document,
            **kwargs,
        )
        return _results_to_docs_and_scores(results)

    def update_collection_properties(
        self,
        collection_name: str,
        current_collection_properties: List,
        new_collection_properties: Optional[List],
    ) -> None:
        if new_collection_properties is not None:
            old_collection_properties = deepcopy(current_collection_properties)
            for prop in new_collection_properties:
                if prop not in current_collection_properties:
                    current_collection_properties.append(prop)

            if current_collection_properties != old_collection_properties:
                all_queries, blob_arr = build_property_query(
                    collection_name,
                    command_type="update",
                    all_properties=current_collection_properties,
                )
                response, _ = self._client.query(all_queries, [blob_arr])
                # self._client.print_last_response()

    def update_document(
        self, collection_name: str, document_id: str, document: Document
    ) -> None:
        """Update a document in the collection.

        Args:
            document_id (str): ID of the document to update.
            document (Document): Document to update.
        """
        return self.update_documents(collection_name, [document_id], [document])

    def update_documents(
        self, collection_name: str, ids: List[str], documents: List[Document]
    ) -> None:
        """Update a document in the collection.

        Args:
            ids (List[str]): List of ids of the document to update.
            documents (List[Document]): List of documents to update.
        """
        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        if self._embedding_function is None:
            raise ValueError(
                "For update, you must specify an embedding function on creation."
            )
        embeddings = self._embedding_function.embed_documents(text)

        # if hasattr(
        #     self._collection._client, "max_batch_size"
        # ):
        #     # from vdmsdb.utils.batch_utils import create_batches

        #     for batch in create_batches(
        #         api=self._collection._client,
        #         ids=ids,
        #         metadatas=metadata,
        #         documents=text,
        #         embeddings=embeddings,
        #     ):
        #         self._collection.update(
        #             ids=batch[0],
        #             embeddings=batch[1],
        #             documents=batch[3],
        #             metadatas=batch[2],
        #         )
        # else:
        self.run_update_documents(
            collection_name,
            ids,
            metadatas=metadata,
            embeddings=embeddings,
            documents=text,
        )


"""
VDMS UTILITY
"""


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    final_res = []
    # for qres in results:
    responses, blobs = results[0]
    if (
        "FindDescriptor" in responses[0]
        and "entities" in responses[0]["FindDescriptor"]
    ):
        result_entities = responses[0]["FindDescriptor"]["entities"]
        # result_blobs = blobs
        for ent in result_entities:
            distance = ent["_distance"]
            txt_contents = ent["content"]
            for p in INVALID_DOC_METADATA_KEYS:
                if p in ent:
                    del ent[p]
            props = {
                mkey: mval
                for mkey, mval in ent.items()
                if mval not in INVALID_METADATA_VALUE
            }

            final_res.append(
                (Document(page_content=txt_contents, metadata=props), distance)
            )
    return final_res


def add_descriptor(
    command_str: str,
    setname: str,
    label: Optional[str] = None,
    ref: Optional[int] = None,
    props: Optional[dict] = None,
    link: Optional[dict] = None,
    k_neighbors: Optional[int] = None,
    constraints: Optional[dict] = None,
    results: Optional[dict] = None,
) -> Dict[str, Dict[str, Any]]:
    entity: Dict[str, Any] = {"set": setname}

    if "Add" in command_str and label:
        entity["label"] = label

    if ref is not None:
        entity["_ref"] = ref

    if props not in INVALID_METADATA_VALUE:
        entity["properties"] = props

    if "Add" in command_str and link is not None:
        entity["link"] = link

    if "Find" in command_str and k_neighbors is not None:
        entity["k_neighbors"] = int(k_neighbors)

    if "Find" in command_str and constraints not in INVALID_METADATA_VALUE:
        entity["constraints"] = constraints

    if "Find" in command_str and results not in INVALID_METADATA_VALUE:
        entity["results"] = results

    query = {command_str: entity}
    return query


def add_descriptorset(
    command_str,
    name,
    num_dims=None,
    engine=None,
    metric=None,
    ref=None,
    props=None,
    link=None,
    storeIndex=False,
    constraints=None,
    results=None,
):
    # command_str = "AddDescriptorSet"
    if command_str == "AddDescriptorSet" and all(
        var is not None for var in [name, num_dims]
    ):
        entity = {
            "name": name,
            "dimensions": num_dims,
        }

        if engine is not None:
            entity["engine"] = engine

        if metric is not None:
            entity["metric"] = metric

        if ref is not None:
            entity["_ref"] = ref

        if props not in [None, {}]:
            entity["properties"] = props

        if link is not None:
            entity["link"] = link

    elif command_str == "FindDescriptorSet":
        entity = {"set": name}

        if storeIndex:
            entity["storeIndex"] = storeIndex

        if constraints not in [None, {}]:
            entity["constraints"] = constraints

        if results is not None:
            entity["results"] = results

    else:
        raise ValueError(f"Unknown command: {command_str}")

    query = {command_str: entity}
    return query


def add_entity_with_blob(
    collection_name: str, all_properties: List
):  # -> Tuple[Dict[str, Dict[str, Any]],List[bytes]]:
    all_properties_str = (
        ",".join(all_properties) if len(all_properties) > 0 else ""
    )  #','.join([''])

    querytype = "AddEntity"
    entity = {}
    # entity["class"] = "collection properties"
    entity["class"] = "properties"
    entity["blob"] = True  # New

    props = {"name": collection_name}
    props["type"] = "queryable properties"
    props["content"] = all_properties_str
    entity["properties"] = props

    byte_data = str2bytes(all_properties_str)
    # if len(all_properties) > 0:
    #     # byte_data = bytes(','.join(all_properties), 'utf-8')
    #     byte_data = str2bytes(','.join(all_properties))
    # else:
    #     # byte_data = bytes(','.join(['']), 'utf-8')
    #     byte_data = str2bytes(','.join(['']))

    query = {}
    query[querytype] = entity
    return query, byte_data


def build_property_query(
    collection_name: str,
    command_type: str = "find",
    all_properties: List = [],
    ref: Optional[int] = None,
):
    all_queries = []
    blob_arr = []

    choices = ["find", "add", "update"]
    if command_type.lower() not in choices:
        raise ValueError("[!] Invalid type. Choices are : {}".format(",".join(choices)))

    if command_type.lower() == "find":
        query = find_property_entity(collection_name, unique_entity=True)
        all_queries.append(query)

    elif command_type.lower() == "add":
        query, byte_data = add_entity_with_blob(collection_name, all_properties)
        all_queries.append(query)
        blob_arr.append(byte_data)

    elif command_type.lower() == "update":
        # Find & Delete
        query = find_property_entity(collection_name, deletion=True)
        all_queries.append(query)

        # Add
        query, byte_data = add_entity_with_blob(collection_name, all_properties)
        all_queries.append(query)
        blob_arr.append(byte_data)

    return all_queries, blob_arr


def bytes2embedding(blob: bytes) -> Any:
    emb = np.frombuffer(blob, dtype="float32")
    return emb


def bytes2str(in_bytes: bytes) -> str:
    return in_bytes.decode()


def check_valid_response(response: Any, cmd_list: List[str]) -> bool:
    return isinstance(response, list) and any(
        cmd in response[0]
        and "returned" in response[0][cmd]
        and response[0][cmd]["returned"] > 0
        for cmd in cmd_list
    )


def check_vdms_then_adddescriptor(
    db: vdms.vdms,
    setname: str,
    label: Optional[str] = None,
    ref: Optional[int] = None,
    props: Optional[dict] = None,
    link: Optional[dict] = None,
    k_neighbors: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    addDescriptor = add_descriptor(
        "AddDescriptor", setname, ref=ref, label=label, props=props, link=link
    )

    props_as_constraints = (
        None if props is None else {k: ["==", v] for k, v in props.items()}
    )
    results = (
        None
        if props_as_constraints is None
        else {"list": list(props_as_constraints.keys()), "limit": 1}
    )

    findDescriptor = add_descriptor(
        "FindDescriptor",
        setname,
        k_neighbors=k_neighbors,
        constraints=props_as_constraints,
        results=results,
    )

    res, _ = db.query([findDescriptor])
    # print(findDescriptor, res)

    if check_valid_response(res, ["AddDescriptor", "FindDescriptor"]):
        if ref is not None:
            findDescriptor["FindDescriptor"]["_ref"] = ref
        # if link is not None:
        #     findDescriptor["FindDescriptor"]["link"] = link
        return findDescriptor
    return addDescriptor


def embedding2bytes(embedding: Any) -> bytes:
    emb = np.array(embedding, dtype="float32")
    blob = emb.tobytes()
    return blob


def find_property_entity(
    collection_name: str,
    unique_entity: Optional[bool] = False,
    deletion: Optional[bool] = False,
) -> Dict[str, Dict[str, Any]]:
    querytype = "FindEntity"
    entity = {}
    # entity["class"] = "collection properties"
    entity["class"] = "properties"
    if unique_entity:
        entity["unique"] = unique_entity

    results = {}
    results["blob"] = True
    results["count"] = ""
    results["list"] = ["content"]
    entity["results"] = results

    constraints = {}
    if deletion:
        constraints["_deletion"] = ["==", 1]
    constraints["name"] = ["==", collection_name]
    entity["constraints"] = constraints

    query = {}
    query[querytype] = entity
    return query


def str2bytes(in_str: str) -> bytes:
    return str.encode(in_str)


def validate_vdms_properties(metadata: Dict) -> Dict:
    new_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list) and len(value) == 1:
            new_metadata[key] = value[0]
        elif isinstance(value, list) and len(value) >= 1:
            print("VDMS doesn't support arrays as properties.")
            if isinstance(value[0], str):
                print("Converting str of arrays to comma delimited string")
                new_metadata[key] = ",".join(value)
            else:
                print(f"Skipping array of {value[0].__class__.__name__}")
        else:
            new_metadata[key] = value
    return new_metadata


#
