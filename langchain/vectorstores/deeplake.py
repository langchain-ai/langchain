"""Wrapper around Activeloop Deep Lake."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


def L2_search(
    query_embedding: np.ndarray, data_vectors: np.ndarray, k: int = 4
) -> list:
    """naive L2 search for nearest neighbors"""
    # Calculate the L2 distance between the query_vector and all data_vectors
    distances = np.linalg.norm(data_vectors - query_embedding, axis=1)

    # Sort the distances and return the indices of the k nearest vectors
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices.tolist()


class DeepLake(VectorStore):
    """Wrapper around Deep Lake, a data lake for deep learning applications.

    It not only stores embeddings, but also the original data and queries with
    version control automatically enabled.

    It is more than just a vector store. You can use the dataset to fine-tune
    your own LLM models or use it for other downstream tasks.

    We implement naive similiarity search, but it can be extended with Tensor
    Query Language (TQL for production use cases) over billion rows.

    To use, you should have the ``deeplake`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import DeepLake
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = DeepLake("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_DEEPLAKE_PATH = "mem://langchain"

    def __init__(
        self,
        dataset_path: str = _LANGCHAIN_DEFAULT_DEEPLAKE_PATH,
        token: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
    ) -> None:
        """Initialize with Deep Lake client."""

        try:
            import deeplake
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        self._deeplake = deeplake

        if deeplake.exists(dataset_path, token=token):
            self.ds = deeplake.load(dataset_path, token=token)
            logger.warning(
                f"Deep Lake Dataset in {dataset_path} already exists, "
                f"loading from the storage"
            )
            self.ds.summary()
        else:
            self.ds = deeplake.empty(dataset_path, token=token, overwrite=True)
            with self.ds:
                self.ds.create_tensor("text", htype="text")
                self.ds.create_tensor("metadata", htype="json")
                self.ds.create_tensor("embedding", htype="generic")
                self.ds.create_tensor("ids", htype="text")

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

        if self._embedding_function is None:
            embeddings: Sequence[Optional[List[float]]] = [None] * len(text_list)
        else:
            embeddings = self._embedding_function.embed_documents(text_list)

        if metadatas is None:
            metadatas_to_use: Sequence[Optional[dict]] = [None] * len(text_list)
        else:
            metadatas_to_use = metadatas

        elements = zip(text_list, embeddings, metadatas_to_use, ids)

        @self._deeplake.compute
        def ingest(sample_in: list, sample_out: list) -> None:
            s = {
                "text": sample_in[0],
                "embedding": sample_in[1],
                "metadata": sample_in[2],
                "ids": sample_in[3],
            }
            sample_out.append(s)

        ingest().eval(list(elements), self.ds)
        self.ds.commit()

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        if self._embedding_function is None:
            self.ds.summary()
            ds_view = self.ds.filter(lambda x: query in x["text"].data()["value"])
        else:
            query_emb = np.array(self._embedding_function.embed_query(query))
            embeddings = self.ds.embedding.numpy()
            indices = L2_search(query_emb, embeddings, k=k)
            ds_view = self.ds[indices]

        docs = [
            Document(
                page_content=el["text"].data()["value"],
                metadata=el["metadata"].data()["value"],
            )
            for el in ds_view
        ]
        return docs

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

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://username/datasetname``.
                    To write to Deep Lake cloud datasets,
                    ensure that you are logged in to Deep Lake
                    (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``.
                    Credentials are required in either the environment or
                    passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or
                    ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't
                    save the dataset but keeps it in memory instead.
                    Should be used only for testing as it does not persist.
            documents (List[Document]): List of documents to add.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

        Returns:
            DeepLake: Deep Lake dataset.
        """
        deeplake_dataset = cls(
            dataset_path=dataset_path,
            embedding_function=embedding,
        )
        deeplake_dataset.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return deeplake_dataset

    def delete_dataset(self) -> None:
        """Delete the collection."""
        self.ds.delete()

    def persist(self) -> None:
        """Persist the collection."""
        self.ds.flush()
