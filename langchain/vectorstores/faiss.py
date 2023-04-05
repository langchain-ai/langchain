"""Wrapper around FAISS vector database."""
from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class FAISS(VectorStore):
    """Wrapper around FAISS vector database.

    To use, you should have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore, index_to_docstore_id)

    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        self.index.add(np.array(embeddings, dtype=np.float32))
        # Get list of index, id, and docs.
        full_info = [
            (starting_len + i, str(uuid.uuid4()), doc)
            for i, doc in enumerate(documents)
        ]
        # Add information to docstore and index.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(texts, embeddings, metadatas, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.

        texts = [te[0] for te in text_embeddings]
        embeddings = [te[1] for te in text_embeddings]
        return self.__add(texts, embeddings, metadatas, **kwargs)

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        _, indices = self.index.search(np.array([embedding], dtype=np.float32), fetch_k)
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32), embeddings, k=k
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        docs = []
        for i in selected_indices:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        return docs

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(embedding, k, fetch_k)
        return docs

    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Create new id for docs from target FAISS object
        full_info = []
        for i in target.index_to_docstore_id:
            doc = target.docstore.search(target.index_to_docstore_id[i])
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, str(uuid.uuid4()), doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(documents)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(texts, embeddings, embedding, metadatas, **kwargs)

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(texts, embeddings, embedding, metadatas, **kwargs)

    def save_local(self, folder_path: str) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / "index.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / "index.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(cls, folder_path: str, embeddings: Embeddings) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / "index.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / "index.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)
