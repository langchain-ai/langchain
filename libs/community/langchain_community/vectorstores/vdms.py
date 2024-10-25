from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    Union,
    get_args,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing.base import UpsertResponse
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import vdms

TEXT_PROPERTY = "content"  # Property name for the text
DEFAULT_COLLECTION_NAME = "langchain"
LANGCHAIN_ID_PROPERTY = "langchain_id"  # Property name for the unique id
DEFAULT_INSERT_BATCH_SIZE = 1000  # 32
DEFAULT_K = 3  # Number of Documents to return.
DEFAULT_FETCH_K = (
    DEFAULT_K * 5
)  # Number of Documents to fetch to pass to knn when filters applied.
INVALID_METADATA_VALUE = ["Missing property", None, {}]  # type: List
DEFAULT_PROPERTIES = ["_distance", LANGCHAIN_ID_PROPERTY, TEXT_PROPERTY]
INVALID_DOC_METADATA_KEYS = ["_distance", TEXT_PROPERTY, "blob"]

DISTANCE_METRICS = Literal[
    "L2",  # Euclidean Distance
    "IP",  # Inner Product
]
AVAILABLE_DISTANCE_METRICS: List[DISTANCE_METRICS] = list(get_args(DISTANCE_METRICS))
ENGINES = Literal[
    "FaissFlat",  # FAISS IndexFlat
    "FaissHNSWFlat",  # FAISS IndexHNSWFlat
    "FaissIVFFlat",  # FAISS IndexIVFFlat
    "Flinng",  # FLINNG
    "TileDBDense",  # TileDB Dense
    "TileDBSparse",  # TileDB Sparse
]
AVAILABLE_ENGINES: List[ENGINES] = list(get_args(ENGINES))


def VDMS_Client(
    host: str = "localhost",
    port: int = 55555,
) -> vdms.vdms:
    """VDMS client for the VDMS server.

    Args:
        host: IP or hostname of VDMS server
        port: Port to connect to VDMS server
    """
    try:
        import vdms
    except ImportError:
        raise ImportError(
            "Could not import vdms python package. "
            "Please install it with `pip install vdms."
        )

    client = vdms.vdms()
    client.connect(host, port)
    return client


class VDMS(VectorStore):
    """Intel Lab's VDMS for vectorstore workloads.

    To use, you should have both:
    - the ``vdms`` python package installed
    - a host (str) and port (int) associated with a deployed VDMS Server

    A single VDMS instance can support numerous vector stores and use
    different distance metrics and engines. The vectorstores must have a
    unique collection_name (DescriptorSet).

    Visit https://github.com/IntelLabs/vdms/wiki for more information.

    IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        client: VDMS Client used to connect to VDMS server
        embedding: Any embedding function implementing
            `langchain_core.embeddings.Embeddings` interface.
        embedding_dimensions: Dimensions of embedding function
        collection_name: Name of data collection [Default: langchain]
        engine: Underlying implementation for indexing and computing distances.
            VDMS supports TileDBDense, TileDBSparse, FaissFlat, FaissIVFFlat,
            FaissHNSWFlat, and Flinng [Default: FaissFlat]
        distance_strategy: Method used to calculate distances. VDMS supports
            "L2" (euclidean distance) or "IP" (inner product) [Default: L2]
        log_level (int, optional): Logging level. Defaults to logging.WARNING.

    Example:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

            model_name = "sentence-transformers/all-mpnet-base-v2"
            vectorstore = VDMS(
                client=VDMS_Client("localhost", 55555),
                embedding=HuggingFaceEmbeddings(model_name=model_name),
                collection_name="langchain-demo",
                distance_strategy="L2",
                engine="FaissFlat",
            )
    """

    @override
    def __init__(
        self,
        client: vdms.vdms,
        *,
        embedding: Embeddings,
        embedding_dimensions: Optional[int] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        engine: ENGINES = "FaissFlat",
        distance_strategy: DISTANCE_METRICS = "L2",
        log_level: int = logging.WARNING,
        **kwargs: Any,
    ) -> None:
        # super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.collection_name = collection_name

        # Check required parameters
        self._client = client
        self.similarity_search_engine = engine
        self.distance_strategy = distance_strategy
        self.embedding = embedding
        self.utils = VDMS_Utils(client, logger=self.logger)
        self._check_required_inputs(collection_name, embedding_dimensions)
        self.updated_properties_flag = False
        self._add_set()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def _add_set(self) -> None:
        collection_name = self.collection_name
        embedding_dimension = self.embedding_dimension
        engine = self.similarity_search_engine
        metric = self.distance_strategy

        query = self.utils.add_descriptor_set(
            "AddDescriptorSet",
            collection_name,
            embedding_dimension,
            engine=getattr(engine, "value", engine),
            metric=getattr(metric, "value", metric),
        )

        response, _ = self.utils.run_vdms_query([query])

        if "FailedCommand" in response[0]:
            raise ValueError(f"Failed to add collection {collection_name}")

        self.logger.info(f"Descriptor set {collection_name} created")

    def _check_required_inputs(
        self, collection_name: str, embedding_dimensions: Union[int, None]
    ) -> None:
        # Check Distance Metric
        if self.distance_strategy not in AVAILABLE_DISTANCE_METRICS:
            raise ValueError("distance_strategy must be either 'L2' or 'IP'")

        # Check Engines
        if self.similarity_search_engine not in AVAILABLE_ENGINES:
            raise ValueError(
                "engine must be either 'TileDBDense', 'TileDBSparse', "
                + "'FaissFlat', 'FaissIVFFlat', 'FaissHNSWFlat', or 'Flinng'"
            )

        # Check Embedding Func is provided and store dimension size
        if self.embedding is None:
            raise ValueError("Must provide embedding function")

        if embedding_dimensions is not None:
            self.embedding_dimension = embedding_dimensions
        elif self.embedding is not None and hasattr(self.embedding, "embed_query"):
            self.embedding_dimension = len(
                # self._embed_query("This is a sample sentence.")
                self.embedding.embed_query("This is a sample sentence.")
            )
        elif self.embedding is not None and (
            hasattr(self.embedding, "embed_image")
            or hasattr(self.embedding, "embed_video")
        ):
            if hasattr(self.embedding, "model"):
                try:
                    self.embedding_dimension = (
                        self.embedding.model.token_embedding.embedding_dim
                    )
                except ValueError:
                    raise ValueError(
                        "Embedding dimension needed. Please define embedding_dimensions"
                    )
            else:
                raise ValueError(
                    "Embedding dimension needed. Please define embedding_dimensions"
                )

        # Check for properties
        current_props = self.utils.get_properties(collection_name)
        if hasattr(self, "collection_properties"):
            self.collection_properties.extend(current_props)
        else:
            self.collection_properties: List[str] = current_props

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding, Embeddings):
            return self.embedding.embed_documents(texts)
        else:
            p_str = "Must provide `embedding` which is expected"
            p_str += " to be an Embeddings object"
            raise ValueError(p_str)

    def _embed_image(self, uris: List[str]) -> List[List[float]]:
        if self.embedding is not None and hasattr(self.embedding, "embed_image"):
            return self.embedding.embed_image(uris=uris)
        else:
            raise ValueError(
                "Must provide `embedding` which has attribute `embed_image`"
            )

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding, Embeddings):
            return self.embedding.embed_query(text)
        else:
            raise ValueError(
                "Must provide `embedding` which is expected"
                " to be an Embeddings object"
            )

    def _embed_video(self, paths: List[str], **kwargs: Any) -> List[List[float]]:
        if self.embedding is not None and hasattr(self.embedding, "embed_video"):
            return self.embedding.embed_video(paths=paths, **kwargs)
        else:
            raise ValueError(
                "Must provide `embedding` which has attribute `embed_video`"
            )

    def _len_check_if_sized(self, x: Any, y: Any, x_name: str, y_name: str) -> None:
        """
        Check that sizes of two variables are the same

        Args:
            x: Variable to compare
            y: Variable to compare
            x_name: Name for variable x
            y_name: Name for variable y
        """
        if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
            raise ValueError(
                f"{x_name} and {y_name} expected to be equal length but "
                f"len({x_name})={len(x)} and len({y_name})={len(y)}"
            )
        return

    def __update(
        self,
        collection_name: str,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[Sequence[Union[dict, None]]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        """
        Updates (find, delete, add) a collection based on id.
        If more than one collection returned with id, error occurs
        """

        metadatas = metadatas if metadatas is not None else [None for _ in ids]
        self._len_check_if_sized(ids, texts, "ids", "texts")
        self._len_check_if_sized(ids, embeddings, "ids", "embeddings")
        self._len_check_if_sized(ids, metadatas, "ids", "metadatas")

        # Find and delete by ID
        _ = self.delete(ids, collection_name=collection_name)

        # Add as batch
        _ = self.add_from(
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            batch_size=batch_size,
            **kwargs,
        )

    def update_properties(
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
                all_queries, blob_arr = self.utils.build_property_query(
                    collection_name,
                    command_type="update",
                    all_properties=current_collection_properties,
                )
                response, _ = self.utils.run_vdms_query(all_queries, [blob_arr])
                self.updated_properties_flag = True

    def add_batch(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[Sequence[Union[dict, None]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List:
        self._len_check_if_sized(texts, embeddings, "texts", "embeddings")

        metadatas = metadatas if metadatas is not None else [None for _ in texts]
        self._len_check_if_sized(texts, metadatas, "texts", "metadatas")

        ids = ids if ids is not None else [str(uuid.uuid4()) for _ in texts]
        self._len_check_if_sized(texts, ids, "texts", "ids")

        extended_emb: List[Any] = []
        batch_properties: List[Dict] = []
        for meta, emb, doc, id in zip(metadatas, embeddings, texts, ids):
            extended_emb.extend(emb)
            batch_properties.append(
                self.get_props_from_metadata(collection_name, doc, meta, id)
            )
        all_blobs = [self.utils.embedding2bytes(extended_emb)]
        all_queries = [
            self.utils.add_descriptor(
                "AddDescriptor",
                collection_name,
                label=None,
                ref=None,
                props=batch_properties,
                link=None,
                k_neighbors=None,
                constraints=None,
                results=None,
            )
        ]
        # if isinstance(all_queries, dict):
        #     all_queries = [all_queries]
        response, _ = self.utils.run_vdms_query(all_queries, all_blobs)

        return ids if response[0]["AddDescriptor"]["status"] == 0 else []

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        ids = None
        if "ids" in kwargs:
            ids = kwargs.pop("ids")
            if ids and len(ids) != len(documents):
                raise ValueError(
                    "The number of ids must match the number of documents. "
                    "Got {len(ids)} ids and {len(documents)} documents."
                )

            documents_ = []

            for id_, document in zip(ids, documents):
                doc_with_id = Document(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    id=id_,
                )
                documents_.append(doc_with_id)
        else:
            documents_ = documents

        if ids is None:
            ids = []
            for doc in documents_:
                if hasattr(doc, "id") and doc.id is not None:
                    ids.append(str(doc.id))
                elif "id" in doc.metadata:
                    ids.append(str(doc.metadata["id"]))
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs["ids"] = ids

        # If upsert has been implemented, we can use it to add documents
        return self.upsert(documents_, **kwargs)["succeeded"]

    def add_from(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[Sequence[Union[dict, None]]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        # Get initial properties
        orig_props = self.utils.get_properties(self.collection_name)
        inserted_ids: List[str] = []
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_embedding_vectors = embeddings[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            if metadatas:
                batch_metadatas = metadatas[start_idx:end_idx]

            result_ids = self.add_batch(
                self.collection_name,
                embeddings=batch_embedding_vectors,
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

            inserted_ids.extend(result_ids)

        # Update Properties
        self.update_properties(
            self.collection_name, orig_props, self.collection_properties
        )
        return inserted_ids

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        add_path: Optional[bool] = True,
        **kwargs: Any,
    ) -> List[str]:
        """Run images through the embeddings and add to the vectorstore.

        Images are added as embeddings (AddDescriptor) instead of separate
        entity (AddImage) within VDMS to leverage similarity search capability

        Args:
            uris: List of paths to the images to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the images.
            ids: Optional list of unique IDs.
            batch_size (int): Number of concurrent requests to send to the server.
            add_path: Bool to add image path as metadata

        Returns:
            List of ids from adding images into the vectorstore.
        """
        # Map from uris to blobs to base64
        b64_texts = [self.encode_image(image_path=uri) for uri in uris]

        if add_path and metadatas:
            for midx, uri in enumerate(uris):
                metadatas[midx]["image_path"] = uri
        elif add_path:
            metadatas = []
            for uri in uris:
                metadatas.append({"image_path": uri})

        # Populate IDs
        ids = ids if ids is not None else [str(uuid.uuid4()) for _ in uris]

        # Set embeddings
        embeddings = self._embed_image(uris=uris)

        if metadatas is None:
            metadatas = [{} for _ in uris]
        else:
            metadatas = [self.utils.validate_vdms_properties(m) for m in metadatas]

        inserted_ids = self.add_from(
            texts=b64_texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            batch_size=batch_size,
            **kwargs,
        )
        return inserted_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            batch_size (int): Number of concurrent requests to send to the server.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        texts = list(texts)

        embeddings = self._embed_documents(texts)

        if metadatas is None:
            metadatas = [{} for _ in texts]
        else:
            metadatas = [self.utils.validate_vdms_properties(m) for m in metadatas]

        if ids is None:
            ids = []
            for meta in metadatas:
                if LANGCHAIN_ID_PROPERTY in meta:
                    ids.append(meta[LANGCHAIN_ID_PROPERTY])
                else:
                    ids.append(str(uuid.uuid4()))

        inserted_ids = self.add_from(
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            batch_size=batch_size,
            **kwargs,
        )
        return inserted_ids

    def add_videos(
        self,
        paths: List[str],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 1,
        add_path: Optional[bool] = True,
        **kwargs: Any,
    ) -> List[str]:
        """Run videos through the embeddings and add to the vectorstore.

        Videos are added as embeddings (AddDescriptor) instead of separate
        entity (AddVideo) within VDMS to leverage similarity search capability

        Args:
            paths: List of paths to the videos to add to the vectorstore.
            text: Optional list of text associated with the videos.
            metadatas: Optional list of metadatas associated with the videos.
            ids: Optional list of unique IDs.
            batch_size (int): Number of concurrent requests to send to the server.
            add_path: Bool to add video path as metadata

        Returns:
            List of ids from adding videos into the vectorstore.
        """
        if texts is None:
            texts = ["" for _ in paths]

        if add_path and metadatas:
            for midx, path in enumerate(paths):
                metadatas[midx]["video_path"] = path
        elif add_path:
            metadatas = []
            for path in paths:
                metadatas.append({"video_path": path})

        # Populate IDs
        ids = ids if ids is not None else [str(uuid.uuid4()) for _ in paths]

        # Set embeddings
        embeddings = self._embed_video(paths=paths, **kwargs)

        if metadatas is None:
            metadatas = [{} for _ in paths]
        else:
            metadatas = [self.utils.validate_vdms_properties(m) for m in metadatas]

        inserted_ids = self.add_from(
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            batch_size=batch_size,
            **kwargs,
        )
        return inserted_ids

    def check_and_update_properties(self) -> None:
        if self.updated_properties_flag:
            updated_props = self.utils.get_properties(self.collection_name)
            if self.collection_properties != updated_props:
                self.collection_properties = updated_props
            self.updated_properties_flag = False

    def count(self, collection_name: str) -> int:
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        results = {
            "count": "",
            "list": [LANGCHAIN_ID_PROPERTY],
        }  # collection_properties}
        query = self.utils.add_descriptor(
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

        response, response_array = self.utils.run_vdms_query(all_queries, all_blobs)
        return response[0]["FindDescriptor"]["returned"]

    def decode_image(self, base64_image: str) -> bytes:
        return base64.b64decode(base64_image)

    @override
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful, False otherwise
        """

        if "collection_name" in kwargs:
            collection_name = kwargs.pop("collection_name")
        else:
            collection_name = self.collection_name

        if "constraints" in kwargs and isinstance(kwargs["constraints"], dict):
            constraints = kwargs.pop("constraints")
            constraints["_deletion"] = ["==", 1]
        else:
            constraints = {"_deletion": ["==", 1]}

        if ids is not None:
            all_constraints = []
            for id in ids:
                tmp = deepcopy(constraints)
                tmp[LANGCHAIN_ID_PROPERTY] = ["==", id]
                all_constraints.append(tmp)
        else:
            all_constraints = [constraints]

        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        collection_properties = self.utils.get_properties(collection_name)
        results = {"list": collection_properties}

        for constraint in all_constraints:
            query = self.utils.add_descriptor(
                "FindDescriptor",
                collection_name,
                label=None,
                ref=None,
                props=None,
                link=None,
                k_neighbors=None,
                constraints=constraint,
                results=results,
            )

            all_queries.append(query)
        response, _ = self.utils.run_vdms_query(all_queries, all_blobs)

        # Update/store indices after deletion
        query = self.utils.add_descriptor_set(
            "FindDescriptorSet", collection_name, storeIndex=True
        )
        _, _ = self.utils.run_vdms_query([query])

        return "FindDescriptor" in response[0]

    def descriptor2document(self, d: dict) -> Document:
        metadata = {}
        d_id = None
        txt_contents = None
        for k, v in d.items():
            if k not in INVALID_DOC_METADATA_KEYS:
                metadata[k] = v
        if LANGCHAIN_ID_PROPERTY in metadata:
            d_id = metadata.pop(LANGCHAIN_ID_PROPERTY)
        # if TEXT_PROPERTY in d:
        txt_contents = d[TEXT_PROPERTY]
        doc = Document(page_content=txt_contents, metadata=metadata, id=d_id)
        return doc

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            blob = f.read()
            return base64.b64encode(blob).decode("utf-8")

    @classmethod
    @override
    def from_documents(
        cls: Type[VDMS],
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        collection_name: str = DEFAULT_COLLECTION_NAME,  # Add this line
        **kwargs: Any,
    ) -> VDMS:
        """Creates a new vectorstore from a list of documents

        Args:
            documents: List of documents
            embedding: Embedding function to use.
            ids: Optional list of IDs associated with the documents.
            batch_size (int): Number of concurrent requests to send to the server.
            collection_name (str): Name of the collection to create.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        client: vdms.vdms = kwargs.pop("client")
        vectorstore = cls(
            client=client,
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        vectorstore.add_documents(documents, ids=ids, batch_size=batch_size, **kwargs)
        return vectorstore

    @classmethod
    @override
    def from_texts(
        cls: Type[VDMS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        **kwargs: Any,
    ) -> VDMS:
        """Creates a new vectorstore from a list of texts

        Args:
            texts: List of text strings
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            ids: Optional list of IDs associated with the texts.
            batch_size (int): Number of concurrent requests to send to the server.
            collection_name (str): Name of the collection to create.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        client: vdms.vdms = kwargs.pop("client")
        vdms_store = cls(
            client=client,
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        # ids = kwargs.pop("ids", None)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        vdms_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )
        return vdms_store

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            documents: List of Document objects found in the vectorstore.
        """

        collection_name = self.collection_name
        all_constraints = []
        for id in ids:
            constraints = {
                LANGCHAIN_ID_PROPERTY: ["==", id],
            }
            all_constraints.append(constraints)

        collection_properties = self.utils.get_properties(collection_name)
        results = {"list": collection_properties}

        docs = []
        for constraint in all_constraints:
            query = self.utils.add_descriptor(
                "FindDescriptor",
                collection_name,
                label=None,
                ref=None,
                props=None,
                link=None,
                k_neighbors=None,
                constraints=constraint,
                results=results,
            )

            response, _ = self.utils.run_vdms_query([query])

            this_docs = [
                self.descriptor2document(doc)
                for doc in response[0]["FindDescriptor"].get("entities", [])
            ]
            docs.extend(this_docs)
        return docs

    def get_by_constraints(
        self,
        collection_name: str,
        constraints: Optional[Dict] = None,
        limit: Optional[int] = None,
        include: List[str] = ["metadata"],
    ) -> Tuple[Any, Any]:
        """Gets the collection.
        Get embeddings and their associated data from the data store.
        If no constraints provided returns all embeddings up to limit.

        Args:
            constraints: A dict used to filter results by.
                   E.g. `{"color" : ["==", "red"], "price": [">", 4.00]}`. Optional.
            limit: The number of documents to return. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.
        """
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        results: Dict[str, Any] = {"count": ""}

        if limit is not None:
            results["limit"] = limit

        # Include metadata
        if "metadata" in include:
            collection_properties = self.utils.get_properties(collection_name)
            results["list"] = collection_properties

        # Include embedding
        if "embeddings" in include:
            results["blob"] = True

        query = self.utils.add_descriptor(
            "FindDescriptor",
            collection_name,
            k_neighbors=None,
            constraints=constraints,
            results=results,
        )

        all_queries.append(query)

        response, response_array = self.utils.run_vdms_query(all_queries, all_blobs)
        return response, response_array

    def get_embedding_from_query(self, query: str) -> List[float]:
        if not os.path.isfile(query) and hasattr(self.embedding, "embed_query"):
            query_embedding: List[float] = self._embed_query(query)
        elif os.path.isfile(query) and hasattr(self.embedding, "embed_image"):
            query_embedding = self._embed_image(uris=[query])[0]
        elif os.path.isfile(query) and hasattr(self.embedding, "embed_video"):
            query_embedding = self._embed_video(paths=[query])[0]
        else:
            error_msg = f"Could not generate embedding for query '{query}'."
            error_msg += "If using path for image or video, verify embedding model "
            error_msg += "has callable functions 'embed_image' or 'embed_video'."
            raise ValueError(error_msg)
        return query_embedding

    def get_props_from_metadata(
        self,
        collection_name: str,
        document: str,
        metadata: Optional[Dict] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if id is None:
            props = {}
        else:
            props = {LANGCHAIN_ID_PROPERTY: id}
            # id_exists, query = self.utils.check_descriptor_exists_by_id(
            #     self._client, collection_name, id
            # )
            # if id_exists:
            #     skipped_value = {
            #         prop_key: prop_val[-1]
            #         for prop_key, prop_val in query["FindDescriptor"][
            #             "constraints"
            #         ].items()
            #     }
            #     pstr = f"[!] Embedding with id ({id}) exists in DB;"
            #     pstr += "Therefore, skipped and not inserted"
            #     self.logger.warning(pstr)
            #     self.logger.warning(f"\tSkipped values are: {skipped_value}")
            #     return {}

        if metadata:
            for k, v in metadata.items():
                if k not in props:
                    props[k] = v
            # props.update(metadata)
            if LANGCHAIN_ID_PROPERTY not in props and "id" in metadata:
                metadata[LANGCHAIN_ID_PROPERTY] = metadata["id"]
        if document not in [None, ""]:
            props["content"] = document

        for k in props.keys():
            if k not in self.collection_properties:
                self.collection_properties.append(k)
        return props

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns similar documents to the query that also have diversity

        This algorithm balances relevance and diversity in the search results.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Document objects ordered by decreasing similarity/diversty.
        """
        self.logger.info(f"Max Marginal Relevance search for query: {query}")
        query_embedding = self.get_embedding_from_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k, fetch_k, lambda_mult, filter, **kwargs
        )

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to search for.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        start_time = time.time()
        results = self.query_by_embeddings(
            query_embeddings=[embedding],
            k=k,
            fetch_k=fetch_k,
            filter=filter,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        if len(results[0][1]) == 0:
            # No results returned
            return []
        else:
            embedding_list = [
                list(self.utils.bytes2embedding(result)) for result in results[0][1]
            ]

            mmr_selected = maximal_marginal_relevance(
                np.array(embedding, dtype=np.float32),
                embedding_list,
                k=k,
                lambda_mult=lambda_mult,
            )

            self.logger.info(
                f"VDMS similarity search mmr took {time.time() - start_time} seconds"
            )
            candidates = self.results2docs(results)
            return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to search for.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        start_time = time.time()
        results = self.query_by_embeddings(
            query_embeddings=[embedding],
            k=k,
            fetch_k=fetch_k,
            filter=filter,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        if len(results[0][1]) == 0:
            # No results returned
            return []
        else:
            embedding_list = [
                list(self.utils.bytes2embedding(result)) for result in results[0][1]
            ]

            mmr_selected = maximal_marginal_relevance(
                np.array(embedding, dtype=np.float32),
                embedding_list,
                k=k,
                lambda_mult=lambda_mult,
            )

            self.logger.info(
                f"VDMS similarity search mmr took {time.time() - start_time} seconds"
            )
            candidates = self.results2docs_and_scores(results)
            return [(r, s) for i, (r, s) in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Returns similar documents to the query that also have diversity

        This algorithm balances relevance and diversity in the search results.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Document objects ordered by decreasing similarity/diversty.
        """
        self.logger.info(f"Max Marginal Relevance search for query: {query}")
        query_embedding = self.get_embedding_from_query(query)
        return self.max_marginal_relevance_search_by_vector_with_score(
            query_embedding, k, fetch_k, lambda_mult, filter, **kwargs
        )

    def query_by_embeddings(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        collection_name: Optional[str] = None,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        filter: Union[None, Dict[str, Any]] = None,
        results: Union[None, Dict[str, Any]] = None,
        normalize_distance: bool = False,
        **kwargs: Any,
    ) -> List:
        self.check_and_update_properties()

        all_responses: List[Any] = []

        if collection_name is None:
            collection_name = self.collection_name

        if query_embeddings is None:
            return all_responses

        include = kwargs.get("include", ["metadatas"])
        if results is None and "metadatas" in include:
            results = {
                "list": self.collection_properties,
                "blob": "embeddings" in include,
            }

        for qemb in query_embeddings:
            response, response_array = self.utils.get_descriptor_response(
                "FindDescriptor",
                collection_name,
                k_neighbors=k,
                fetch_k=fetch_k,
                constraints=filter,
                results=results,
                normalize_distance=normalize_distance,
                query_embedding=qemb,
            )
            try:
                result_entities = response[0]["FindDescriptor"].get("entities", [])
            except ValueError:
                result_entities = []
            for eidx, ent in enumerate(result_entities):
                response[0]["FindDescriptor"]["entities"][eidx] = {
                    k: v for k, v in ent.items() if v not in INVALID_METADATA_VALUE
                }

            all_responses.append([response, response_array])

        return all_responses

    def results2docs(self, results: Any) -> List[Document]:
        return [doc for doc, _ in self.results2docs_and_scores(results)]

    def results2docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        final_res: List[Any] = []
        try:
            responses, blobs = results[0]
            result_entities = responses[0]["FindDescriptor"].get("entities", [])
            if len(result_entities) > 0:
                for ent in result_entities:
                    distance = round(ent["_distance"], 10)
                    txt_contents = ent[TEXT_PROPERTY]
                    props = {
                        mkey: mval
                        for mkey, mval in ent.items()
                        if (
                            mval not in INVALID_METADATA_VALUE
                            and mkey not in INVALID_DOC_METADATA_KEYS
                        )
                    }

                    final_res.append(
                        (Document(page_content=txt_contents, metadata=props), distance)
                    )
        except Exception as e:
            self.logger.warning(
                f"No results returned. Error while parsing results: {e}"
            )
        return final_res

    @override
    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Query string to search for.
            k: Number of Documents to return.
            fetch_k: Number of candidates to fetch for knn (>= k).
            filter: Filter by metadata. Defaults to None.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        assert self.embedding is not None, "Embedding function is not set"
        query_embedding = self.get_embedding_from_query(query)
        return self.similarity_search_by_vector(
            query_embedding, k, fetch_k=fetch_k, filter=filter, **kwargs
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of candidates to fetch for knn (>= k).
            filter: Filter by metadata. Defaults to None.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        start_time = time.time()
        results = self.query_by_embeddings(
            query_embeddings=[embedding],
            k=k,
            fetch_k=fetch_k,
            filter=filter,
            **kwargs,
        )
        self.logger.info(
            f"VDMS similarity search took {time.time() - start_time} seconds"
        )

        final_docs = []
        for this_result in results:
            resp, resp_arr = this_result
            try:
                descriptor = resp[0]["FindDescriptor"].get("entities", [])
            except ValueError:
                descriptor = []
            if isinstance(descriptor, dict):
                final_docs.append(self.descriptor2document(descriptor))
            elif isinstance(descriptor, list):
                for desc in descriptor:
                    final_docs.append(self.descriptor2document(desc))
            else:
                pass
        return final_docs

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        filter: Optional[Dict[str, List]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_embedding = self.get_embedding_from_query(query)
        results = self.query_by_embeddings(
            query_embeddings=[query_embedding],
            k=k,
            fetch_k=fetch_k,
            filter=filter,
            **kwargs,
        )

        return self.results2docs_and_scores(results)

    def upsert(self, documents: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Insert or update items

        Updating documents is dependent on the documents' `id` attribute.

        Args:
            items: List of Document objects to upsert

        Returns:
            UpsertResponse object with succeeded and failed
        """
        # For now, simply delete and add
        # We could do something more efficient to update metadata,
        # but we don't support changing the embedding of a descriptor.
        ids: List[str]
        if "ids" in kwargs:
            ids = kwargs.pop("ids")
            if ids and len(ids) != len(documents):
                raise ValueError(
                    "The number of ids must match the number of documents. "
                    "Got {len(ids)} ids and {len(documents)} documents."
                )
        else:
            ids = [
                item.id
                for item in documents
                if hasattr(item, "id") and item.id is not None
            ]

        text = [document.page_content for document in documents]
        metadatas = [
            self.utils.validate_vdms_properties(document.metadata)
            if getattr(document, "metadata", None) is not None
            else {}
            for document in documents
        ]
        embeddings = self._embed_documents(text)

        self.__update(
            self.collection_name,
            ids,
            texts=text,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )
        return UpsertResponse(succeeded=ids, failed=[])


class VDMS_Utils:
    def __init__(self, client: vdms.vdms, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger

        # Check connection to client
        if not self.client.is_connected():
            raise ValueError(
                "VDMS client must be connected to a VDMS server."
                + "Please use VDMS_Client to establish a connection"
            )

    def add_descriptor(
        self,
        command_str: str,
        setname: str,
        label: Optional[str] = None,
        ref: Optional[int] = None,
        props: Optional[Union[dict, List]] = None,
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

        use_batch = isinstance(props, list) and len(props) > 1
        convert_batch = isinstance(props, list) and len(props) == 1

        if use_batch:
            entity["batch_properties"] = props
        elif convert_batch and props[0] not in INVALID_METADATA_VALUE:
            entity["properties"] = props[0]
        elif isinstance(props, dict) and props not in INVALID_METADATA_VALUE:
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

    def add_descriptor_set(
        self,
        command_str: str,
        name: str,
        num_dims: Optional[int] = None,
        engine: Optional[str] = None,
        metric: Optional[str] = None,
        ref: Optional[int] = None,
        props: Optional[Dict] = None,
        link: Optional[Dict] = None,
        storeIndex: bool = False,
        constraints: Optional[Dict] = None,
        results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if command_str == "AddDescriptorSet" and all(
            var is not None for var in [name, num_dims]
        ):
            entity: Dict[str, Any] = {
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
        self, collection_name: str, all_properties: List
    ) -> Tuple[Dict[str, Any], bytes]:
        all_properties_str = ",".join(all_properties) if len(all_properties) > 0 else ""

        querytype = "AddEntity"
        entity: Dict[str, Any] = {}
        entity["class"] = "properties"
        entity["blob"] = True  # New

        props: Dict[str, Any] = {"name": collection_name}
        props["type"] = "queryable properties"
        props[TEXT_PROPERTY] = all_properties_str
        entity["properties"] = props

        byte_data = self.str2bytes(all_properties_str)

        query: Dict[str, Any] = {}
        query[querytype] = entity
        return query, byte_data

    def build_property_query(
        self,
        collection_name: str,
        command_type: str = "find",
        all_properties: List = [],
        ref: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        all_queries: List[Any] = []
        blob_arr: List[Any] = []

        choices = ["find", "add", "update"]
        if command_type.lower() not in choices:
            raise ValueError(
                "[!] Invalid type. Choices are : {}".format(",".join(choices))
            )

        if command_type.lower() == "find":
            query = self.find_property_entity(collection_name, unique_entity=True)
            all_queries.append(query)

        elif command_type.lower() == "add":
            query, byte_data = self.add_entity_with_blob(
                collection_name, all_properties
            )
            all_queries.append(query)
            blob_arr.append(byte_data)

        elif command_type.lower() == "update":
            # Find & Delete
            query = self.find_property_entity(collection_name, deletion=True)
            all_queries.append(query)

            # Add
            query, byte_data = self.add_entity_with_blob(
                collection_name, all_properties
            )
            all_queries.append(query)
            blob_arr.append(byte_data)

        return all_queries, blob_arr

    def bytes2embedding(self, blob: bytes) -> Any:
        emb = np.frombuffer(blob, dtype="float32")
        return emb

    def bytes2str(self, in_bytes: bytes) -> str:
        return in_bytes.decode()

    def check_valid_response(
        self, all_queries: List[dict], response: Any
    ) -> tuple[Any, bool]:
        cmd_list = self.get_cmds_from_query(all_queries)
        valid_res = isinstance(response, list) and any(
            cmd in response[0]
            and "returned" in response[0][cmd]
            and response[0][cmd]["returned"] > 0
            for cmd in cmd_list
        )

        # ID required
        for ridx, res in enumerate(response):
            if "FindDescriptor" in res:
                ent = res["FindDescriptor"].get("entities", [])
                response[ridx]["FindDescriptor"]["entities"] = [
                    e for e in ent if LANGCHAIN_ID_PROPERTY in e
                ]
                response[ridx]["FindDescriptor"]["returned"] = len(
                    response[ridx]["FindDescriptor"]["entities"]
                )

        return response, valid_res

    # def check_descriptor_exists_by_id(
    #     self,
    #     client: vdms.vdms,
    #     setname: str,
    #     id: str,
    # ) -> Tuple[bool, Any]:
    #     constraints = {LANGCHAIN_ID_PROPERTY: ["==", id]}
    #     findDescriptor = self.add_descriptor(
    #         "FindDescriptor",
    #         setname,
    #         constraints=constraints,
    #         results={"list": [LANGCHAIN_ID_PROPERTY], "count": ""},
    #     )
    #     all_queries = [findDescriptor]
    #     res, _ = client.query(all_queries)

    #     res, valid_res = self.check_valid_response(all_queries, res)
    #     return valid_res, findDescriptor

    def embedding2bytes(
        self, embedding: Union[List[float], None]
    ) -> Union[bytes, None]:
        """Convert embedding to bytes."""

        blob = None
        if embedding is not None:
            emb = np.array(embedding, dtype="float32")
            blob = emb.tobytes()
        return blob

    def find_property_entity(
        self,
        collection_name: str,
        unique_entity: Optional[bool] = False,
        deletion: Optional[bool] = False,
    ) -> Dict[str, Dict[str, Any]]:
        querytype = "FindEntity"
        entity: Dict[str, Any] = {}
        entity["class"] = "properties"
        if unique_entity:
            entity["unique"] = unique_entity

        results: Dict[str, Any] = {}
        results["blob"] = True
        results["count"] = ""
        results["list"] = [TEXT_PROPERTY]
        entity["results"] = results

        constraints: Dict[str, Any] = {}
        if deletion:
            constraints["_deletion"] = ["==", 1]
        constraints["name"] = ["==", collection_name]
        entity["constraints"] = constraints

        query: Dict[str, Any] = {}
        query[querytype] = entity
        return query

    def get_cmds_from_query(self, all_queries: list) -> List[str]:
        return list(set([k for q in all_queries for k in q.keys()]))

    def get_descriptor_response(
        self,
        command_str: str,
        setname: str,
        k_neighbors: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        constraints: Optional[dict] = None,
        results: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
        normalize_distance: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List]:
        all_blobs: List[Any] = []
        blob = self.embedding2bytes(query_embedding)
        if blob is not None:
            all_blobs.append(blob)

        if constraints is None:
            # K results returned
            response, response_array, max_dist = self.get_k_candidates(
                setname,
                fetch_k,
                results,
                all_blobs,
                normalize=normalize_distance,
                k=k_neighbors,
            )
        else:
            if results is None:
                results = {"list": [LANGCHAIN_ID_PROPERTY]}
            elif "list" not in results:
                results["list"] = [LANGCHAIN_ID_PROPERTY]
            elif LANGCHAIN_ID_PROPERTY not in results["list"]:
                results["list"].append(LANGCHAIN_ID_PROPERTY)

            # (1) Find docs satisfy constraints
            query = self.add_descriptor(
                command_str,
                setname,
                constraints=constraints,
                results=results,
            )
            response, response_array = self.run_vdms_query([query])
            if command_str in response[0] and response[0][command_str]["returned"] > 0:
                ids_of_interest = [
                    ent[LANGCHAIN_ID_PROPERTY]
                    for ent in response[0][command_str]["entities"]
                ]
            else:
                return [], []

            # (2) Find top fetch_k results
            response, response_array, max_dist = self.get_k_candidates(
                setname, fetch_k, results, all_blobs, normalize=normalize_distance
            )
            if command_str not in response[0] or (
                command_str in response[0] and response[0][command_str]["returned"] == 0
            ):
                return [], []

            # (3) Intersection of (1) & (2) using ids
            new_entities: List[Dict] = []
            for ent in response[0][command_str]["entities"]:
                if ent[LANGCHAIN_ID_PROPERTY] in ids_of_interest:
                    new_entities.append(ent)
                if len(new_entities) == k_neighbors:
                    break
            response[0][command_str]["entities"] = new_entities
            response[0][command_str]["returned"] = len(new_entities)
            if len(new_entities) < k_neighbors:
                p_str = "Returned items < k_neighbors; Try increasing fetch_k"
                self.logger.warning(p_str)

        if normalize_distance:
            max_dist = 1.0 if max_dist in [0, np.inf] else max_dist
            for ent_idx, ent in enumerate(response[0][command_str]["entities"]):
                ent["_distance"] = ent["_distance"] / max_dist
                response[0][command_str]["entities"][ent_idx]["_distance"] = ent[
                    "_distance"
                ]

        return response, response_array

    def get_k_candidates(
        self,
        setname: str,
        fetch_k: Optional[int],
        results: Optional[Dict[str, Any]] = None,
        all_blobs: Optional[List] = None,
        normalize: Optional[bool] = False,
        k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List, float]:
        max_dist = 1
        k = fetch_k if k is None else k
        command_str = "FindDescriptor"
        query = self.add_descriptor(
            command_str,
            setname,
            k_neighbors=fetch_k,
            results=results,
        )
        response, response_array = self.run_vdms_query([query], all_blobs)

        if "FailedCommand" in response[0]:
            return [], [], max_dist

        if k != fetch_k and "entities" in response[0][command_str]:
            response[0][command_str]["entities"] = response[0][command_str]["entities"][
                :k
            ]
            if len(response_array) != 0:
                response_array = response_array[:k]

        if normalize and command_str in response[0]:
            max_dist = response[0][command_str]["entities"][-1]["_distance"]

        return response, response_array, max_dist

    def get_properties(
        self,
        collection_name: str,
        unique_entity: Optional[bool] = False,
        deletion: Optional[bool] = False,
    ) -> List[str]:
        find_query = self.find_property_entity(
            collection_name, unique_entity=unique_entity, deletion=deletion
        )
        response, response_blob = self.run_vdms_query([find_query])
        if len(response_blob) > 0:
            collection_properties = self.bytes2str(response_blob[0]).split(",")
        else:
            collection_properties = deepcopy(DEFAULT_PROPERTIES)
        return collection_properties

    def run_vdms_query(
        self,
        all_queries: List[Dict],
        all_blobs: Optional[List] = [],
        print_last_response: Optional[bool] = False,
    ) -> Tuple[Any, Any]:
        response, response_array = self.client.query(all_queries, all_blobs)

        response, _ = self.check_valid_response(all_queries, response)
        if print_last_response:
            self.client.print_last_response()
        return response, response_array

    def str2bytes(self, in_str: str) -> bytes:
        return str.encode(in_str)

    def validate_vdms_properties(self, metadata: Dict[str, Any]) -> Dict:
        new_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(value, list):
                new_metadata[str(key)] = value
        return new_metadata
