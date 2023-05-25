"""Wrapper around Activeloop Deep Lake."""
from __future__ import annotations

import logging
import uuid
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)

distance_metric_map = {
    "l2": lambda a, b: np.linalg.norm(a - b, axis=1, ord=2),
    "l1": lambda a, b: np.linalg.norm(a - b, axis=1, ord=1),
    "max": lambda a, b: np.linalg.norm(a - b, axis=1, ord=np.inf),
    "cos": lambda a, b: np.dot(a, b.T)
    / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)),
    "dot": lambda a, b: np.dot(a, b.T),
}


def vector_search(
    query_embedding: np.ndarray,
    data_vectors: np.ndarray,
    distance_metric: str = "L2",
    k: Optional[int] = 4,
) -> Tuple[List, List]:
    """Naive search for nearest neighbors

    args:
        query_embedding: np.ndarray
        data_vectors: np.ndarray
        k (int): number of nearest neighbors
        distance_metric: distance function 'L2' for Euclidean, 'L1' for Nuclear, 'Max'
            l-infinity distnace, 'cos' for cosine similarity, 'dot' for dot product

    returns:
        nearest_indices: List, indices of nearest neighbors
    """
    if data_vectors.shape[0] == 0:
        return [], []

    # Calculate the distance between the query_vector and all data_vectors
    distances = distance_metric_map[distance_metric](query_embedding, data_vectors)
    nearest_indices = np.argsort(distances)

    nearest_indices = (
        nearest_indices[::-1][:k] if distance_metric in ["cos"] else nearest_indices[:k]
    )

    return nearest_indices.tolist(), distances[nearest_indices].tolist()


def dp_filter(x: dict, filter: Dict[str, str]) -> bool:
    """Filter helper function for Deep Lake"""
    metadata = x["metadata"].data()["value"]
    return all(k in metadata and v == metadata[k] for k, v in filter.items())


class DeepLake(VectorStore):
    """Wrapper around Deep Lake, a data lake for deep learning applications.

    We implement naive similarity search and filtering for fast prototyping,
    but it can be extended with Tensor Query Language (TQL) for production use cases
    over billion rows.

    Why Deep Lake?

    - Not only stores embeddings, but also the original data with version control.
    - Serverless, doesn't require another service and can be used with major
        cloud providers (S3, GCS, etc.)
    - More than just a multi-modal vector store. You can use the dataset
        to fine-tune your own LLM models.

    To use, you should have the ``deeplake`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import DeepLake
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = DeepLake("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_DEEPLAKE_PATH = "./deeplake/"

    def __init__(
        self,
        dataset_path: str = _LANGCHAIN_DEFAULT_DEEPLAKE_PATH,
        token: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 0,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize with Deep Lake client."""
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        self.verbose = verbose

        try:
            import deeplake
            from deeplake.constants import MB
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        self._deeplake = deeplake
        self.dataset_path = dataset_path
        creds_args = {"creds": kwargs["creds"]} if "creds" in kwargs else {}

        if deeplake.exists(dataset_path, token=token, **creds_args) and not kwargs.get(
            "overwrite", False
        ):
            if "overwrite" in kwargs:
                del kwargs["overwrite"]

            self.ds = deeplake.load(
                dataset_path,
                token=token,
                read_only=read_only,
                verbose=self.verbose,
                **kwargs,
            )
            logger.info(f"Loading deeplake {dataset_path} from storage.")
            if self.verbose:
                print(
                    f"Deep Lake Dataset in {dataset_path} already exists, "
                    f"loading from the storage"
                )
                self.ds.summary()
        else:
            if "overwrite" in kwargs:
                del kwargs["overwrite"]

            self.ds = deeplake.empty(
                dataset_path,
                token=token,
                overwrite=True,
                verbose=self.verbose,
                **kwargs,
            )

            with self.ds:
                self.ds.create_tensor(
                    "text",
                    htype="text",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )
                self.ds.create_tensor(
                    "metadata",
                    htype="json",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )
                self.ds.create_tensor(
                    "embedding",
                    htype="generic",
                    dtype=np.float32,
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    max_chunk_size=64 * MB,
                    create_shape_tensor=True,
                )
                self.ds.create_tensor(
                    "ids",
                    htype="text",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )

        self._embedding_function = embedding_function

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        text_list = list(texts)

        if metadatas is None:
            metadatas = [{}] * len(text_list)

        elements = list(zip(text_list, metadatas, ids))

        @self._deeplake.compute
        def ingest(sample_in: list, sample_out: list) -> None:
            text_list = [s[0] for s in sample_in]

            embeds: Sequence[Optional[np.ndarray]] = []

            if self._embedding_function is not None:
                embeddings = self._embedding_function.embed_documents(text_list)
                embeds = [np.array(e, dtype=np.float32) for e in embeddings]
            else:
                embeds = [None] * len(text_list)

            for s, e in zip(sample_in, embeds):
                sample_out.append(
                    {
                        "text": s[0],
                        "metadata": s[1],
                        "ids": s[2],
                        "embedding": e,
                    }
                )

        batch_size = min(self.ingestion_batch_size, len(elements))
        if batch_size == 0:
            return []

        batched = [
            elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
        ]

        ingest().eval(
            batched,
            self.ds,
            num_workers=min(self.num_workers, len(batched) // max(self.num_workers, 1)),
            **kwargs,
        )
        self.ds.commit(allow_empty=True)
        if self.verbose:
            self.ds.summary()
        return ids

    def _search_helper(
        self,
        query: Any[str, None] = None,
        embedding: Any[float, None] = None,
        k: int = 4,
        distance_metric: str = "L2",
        use_maximal_marginal_relevance: Optional[bool] = False,
        fetch_k: Optional[int] = 20,
        filter: Optional[Any[Dict[str, str], Callable, str]] = None,
        return_score: Optional[bool] = False,
        **kwargs: Any,
    ) -> Any[List[Document], List[Tuple[Document, float]]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            embedding: Embedding function to use. Defaults to None.
            k: Number of Documents to return. Defaults to 4.
            distance_metric: `L2` for Euclidean, `L1` for Nuclear,
                `max` L-infinity distance, `cos` for cosine similarity,
                'dot' for dot product. Defaults to `L2`.
            filter: Attribute filter by metadata example {'key': 'value'}. It can also
            take [Deep Lake filter]
            (https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter)
                Defaults to None.
            maximal_marginal_relevance: Whether to use maximal marginal relevance.
                Defaults to False.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            return_score: Whether to return the score. Defaults to False.

        Returns:
            List of Documents selected by the specified distance metric,
            if return_score True, return a tuple of (Document, score)
        """
        view = self.ds

        # attribute based filtering
        if filter is not None:
            if isinstance(filter, dict):
                filter = partial(dp_filter, filter=filter)

            view = view.filter(filter)
            if len(view) == 0:
                return []

        if self._embedding_function is None:
            view = view.filter(lambda x: query in x["text"].data()["value"])
            scores = [1.0] * len(view)

            if use_maximal_marginal_relevance:
                raise ValueError(
                    "For MMR search, you must specify an embedding function on"
                    "creation."
                )

        else:
            emb = embedding or self._embedding_function.embed_query(
                query
            )  # type: ignore
            query_emb = np.array(emb, dtype=np.float32)
            embeddings = view.embedding.numpy(fetch_chunks=True)
            k_search = fetch_k if use_maximal_marginal_relevance else k
            indices, scores = vector_search(
                query_emb,
                embeddings,
                k=k_search,
                distance_metric=distance_metric.lower(),
            )

            view = view[indices]
            if use_maximal_marginal_relevance:
                lambda_mult = kwargs.get("lambda_mult", 0.5)
                indices = maximal_marginal_relevance(
                    query_emb,
                    embeddings[indices],
                    k=min(k, len(indices)),
                    lambda_mult=lambda_mult,
                )
                view = view[indices]
                scores = [scores[i] for i in indices]

        docs = [
            Document(
                page_content=el["text"].data()["value"],
                metadata=el["metadata"].data()["value"],
            )
            for el in view
        ]

        if return_score:
            return [(doc, score) for doc, score in zip(docs, scores)]

        return docs

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: text to embed and run the query on.
            k: Number of Documents to return.
                Defaults to 4.
            query: Text to look up documents similar to.
            embedding: Embedding function to use.
                Defaults to None.
            k: Number of Documents to return.
                Defaults to 4.
            distance_metric: `L2` for Euclidean, `L1` for Nuclear, `max`
                L-infinity distance, `cos` for cosine similarity, 'dot' for dot product
                Defaults to `L2`.
            filter: Attribute filter by metadata example {'key': 'value'}.
                Defaults to None.
            maximal_marginal_relevance: Whether to use maximal marginal relevance.
                Defaults to False.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            return_score: Whether to return the score. Defaults to False.

        Returns:
            List of Documents most similar to the query vector.
        """
        return self._search_helper(query=query, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query vector.
        """
        return self._search_helper(embedding=embedding, k=k, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        distance_metric: str = "L2",
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Deep Lake with distance returned.

        Args:
            query (str): Query text to search for.
            distance_metric: `L2` for Euclidean, `L1` for Nuclear, `max` L-infinity
                distance, `cos` for cosine similarity, 'dot' for dot product.
                Defaults to `L2`.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        return self._search_helper(
            query=query,
            k=k,
            filter=filter,
            return_score=True,
            distance_metric=distance_metric,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return self._search_helper(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            use_maximal_marginal_relevance=True,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )
        return self._search_helper(
            query=query,
            k=k,
            fetch_k=fetch_k,
            use_maximal_marginal_relevance=True,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        dataset_path: str = _LANGCHAIN_DEFAULT_DEEPLAKE_PATH,
        **kwargs: Any,
    ) -> DeepLake:
        """Create a Deep Lake dataset from a raw documents.

        If a dataset_path is specified, the dataset will be persisted in that location,
        otherwise by default at `./deeplake`

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - Deep Lake cloud path of the form ``hub://username/dataset_name``.
                    To write to Deep Lake cloud datasets,
                    ensure that you are logged in to Deep Lake
                    (use 'activeloop login' from command line)
                - AWS S3 path of the form ``s3://bucketname/path/to/dataset``.
                    Credentials are required in either the environment
                - Google Cloud Storage path of the form
                    ``gcs://bucketname/path/to/dataset``Credentials are required
                    in either the environment
                - Local file system path of the form ``./path/to/dataset`` or
                    ``~/path/to/dataset`` or ``path/to/dataset``.
                - In-memory path of the form ``mem://path/to/dataset`` which doesn't
                    save the dataset, but keeps it in memory instead.
                    Should be used only for testing as it does not persist.
            documents (List[Document]): List of documents to add.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

        Returns:
            DeepLake: Deep Lake dataset.
        """
        deeplake_dataset = cls(
            dataset_path=dataset_path, embedding_function=embedding, **kwargs
        )
        deeplake_dataset.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return deeplake_dataset

    def delete(
        self,
        ids: Any[List[str], None] = None,
        filter: Any[Dict[str, str], None] = None,
        delete_all: Any[bool, None] = None,
    ) -> bool:
        """Delete the entities in the dataset

        Args:
            ids (Optional[List[str]], optional): The document_ids to delete.
                Defaults to None.
            filter (Optional[Dict[str, str]], optional): The filter to delete by.
                Defaults to None.
            delete_all (Optional[bool], optional): Whether to drop the dataset.
                Defaults to None.
        """
        if delete_all:
            self.ds.delete(large_ok=True)
            return True

        view = None
        if ids:
            view = self.ds.filter(lambda x: x["ids"].data()["value"] in ids)
            ids = list(view.sample_indices)

        if filter:
            if view is None:
                view = self.ds
            view = view.filter(partial(dp_filter, filter=filter))
            ids = list(view.sample_indices)

        with self.ds:
            for id in sorted(ids)[::-1]:
                self.ds.pop(id)

            self.ds.commit(f"deleted {len(ids)} samples", allow_empty=True)

        return True

    @classmethod
    def force_delete_by_path(cls, path: str) -> None:
        """Force delete dataset by path"""
        try:
            import deeplake
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        deeplake.delete(path, large_ok=True, force=True)

    def delete_dataset(self) -> None:
        """Delete the collection."""
        self.delete(delete_all=True)

    def persist(self) -> None:
        """Persist the collection."""
        self.ds.flush()
