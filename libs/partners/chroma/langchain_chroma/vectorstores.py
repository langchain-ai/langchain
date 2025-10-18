"""This is the langchain_chroma.vectorstores module.

It contains the Chroma class which is a vector store for handling various tasks.
"""

from __future__ import annotations

import base64
import logging
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import chromadb
import chromadb.config
import numpy as np
from chromadb import Settings
from chromadb.api import CreateCollectionConfiguration
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from chromadb.api.types import Where, WhereDocument

logger = logging.getLogger()
DEFAULT_K = 4  # Number of Documents to return.


def _results_to_docs(results: Any) -> list[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2]),
            result[3],
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results["distances"][0],
            strict=False,
        )
    ]


def _results_to_docs_and_vectors(results: Any) -> list[tuple[Document, np.ndarray]]:
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["embeddings"][0],
            strict=False,
        )
    ]


Matrix = list[list[float]], list[np.ndarray] | np.ndarray


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:  # type: ignore[valid-type]
    """Row-wise cosine similarity between two equal-width matrices.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        msg = (
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )
        raise ValueError(
            msg,
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between `0` and `1` that determines the degree
            of diversity among the results with `0` corresponding
            to maximum diversity and `1` to minimum diversity.
        k: Number of Documents to return.

    Returns:
        List of indices of embeddings selected by maximal marginal relevance.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class Chroma(VectorStore):
    """Chroma vector store integration.

    Setup:
        Install `chromadb`, `langchain-chroma` packages:

        ```bash
        pip install -qU chromadb langchain-chroma
        ```

    Key init args — indexing params:
        collection_name:
            Name of the collection.
        embedding_function:
            Embedding function to use.

    Key init args — client params:
        client:
            Chroma client to use.
        client_settings:
            Chroma client settings.
        persist_directory:
            Directory to persist the collection.
        host:
            Hostname of a deployed Chroma server.
        port:
            Connection port for a deployed Chroma server. Default is 8000.
        ssl:
            Whether to establish an SSL connection with a deployed Chroma server. Default is False.
        headers:
            HTTP headers to send to a deployed Chroma server.
        chroma_cloud_api_key:
            Chroma Cloud API key.
        tenant:
            Tenant ID. Required for Chroma Cloud connections. Default is 'default_tenant' for local Chroma servers.
        database:
            Database name. Required for Chroma Cloud connections. Default is 'default_database'.

    Instantiate:
        ```python
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        vector_store = Chroma(
            collection_name="foo",
            embedding_function=OpenAIEmbeddings(),
            # other params...
        )
        ```

    Add Documents:
        ```python
        from langchain_core.documents import Document

        document_1 = Document(page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(page_content="thud", metadata={"bar": "baz"})
        document_3 = Document(page_content="i will be deleted :(")

        documents = [document_1, document_2, document_3]
        ids = ["1", "2", "3"]
        vector_store.add_documents(documents=documents, ids=ids)
        ```

    Update Documents:
        ```python
        updated_document = Document(
            page_content="qux",
            metadata={"bar": "baz"},
        )

        vector_store.update_documents(ids=["1"], documents=[updated_document])
        ```

    Delete Documents:
        ```python
        vector_store.delete(ids=["3"])
        ```

    Search:
        ```python
        results = vector_store.similarity_search(query="thud", k=1)
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        *thud[{"baz": "bar"}]
        ```

    Search with filter:
        ```python
        results = vector_store.similarity_search(
            query="thud", k=1, filter={"baz": "bar"}
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        *foo[{"baz": "bar"}]
        ```

    Search with score:
        ```python
        results = vector_store.similarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        * [SIM=0.000000] qux [{'bar': 'baz', 'baz': 'bar'}]
        ```

    Async:
        ```python
        # add documents
        # await vector_store.aadd_documents(documents=documents, ids=ids)

        # delete documents
        # await vector_store.adelete(ids=["3"])

        # search
        # results = vector_store.asimilarity_search(query="thud",k=1)

        # search with score
        results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.335463] foo [{'baz': 'bar'}]
        ```

    Use as Retriever:
        ```python
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        retriever.invoke("thud")
        ```

        ```python
        [Document(metadata={"baz": "bar"}, page_content="thud")]
        ```
    """  # noqa: E501

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Embeddings | None = None,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        client: chromadb.ClientAPI | None = None,
        relevance_score_fn: Callable[[float], float] | None = None,
        create_collection_if_not_exists: bool | None = True,  # noqa: FBT001, FBT002
        *,
        ssl: bool = False,
    ) -> None:
        """Initialize with a Chroma client.

        Args:
            collection_name: Name of the collection to create.
            embedding_function: Embedding class object. Used to embed texts.
            persist_directory: Directory to persist the collection.
            host: Hostname of a deployed Chroma server.
            port: Connection port for a deployed Chroma server. Default is 8000.
            ssl: Whether to establish an SSL connection with a deployed Chroma server.
                    Default is False.
            headers: HTTP headers to send to a deployed Chroma server.
            chroma_cloud_api_key: Chroma Cloud API key.
            tenant: Tenant ID. Required for Chroma Cloud connections.
                    Default is 'default_tenant' for local Chroma servers.
            database: Database name. Required for Chroma Cloud connections.
                    Default is 'default_database'.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            collection_configuration: Index configuration for the collection.

            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/python/client
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to `True`.
        """
        _tenant = tenant or chromadb.DEFAULT_TENANT
        _database = database or chromadb.DEFAULT_DATABASE
        _settings = client_settings or Settings()

        client_args = {
            "persist_directory": persist_directory,
            "host": host,
            "chroma_cloud_api_key": chroma_cloud_api_key,
        }

        if sum(arg is not None for arg in client_args.values()) > 1:
            provided = [
                name for name, value in client_args.items() if value is not None
            ]
            msg = (
                f"Only one of 'persist_directory', 'host' and 'chroma_cloud_api_key' "
                f"is allowed, but got {','.join(provided)}"
            )
            raise ValueError(msg)

        if client is not None:
            self._client = client

        # PersistentClient
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=_settings,
                tenant=_tenant,
                database=_database,
            )

        # HttpClient
        elif host is not None:
            _port = port or 8000
            self._client = chromadb.HttpClient(
                host=host,
                port=_port,
                ssl=ssl,
                headers=headers,
                settings=_settings,
                tenant=_tenant,
                database=_database,
            )

        # CloudClient
        elif chroma_cloud_api_key is not None:
            if not tenant or not database:
                msg = (
                    "Must provide tenant and database values to connect to Chroma Cloud"
                )
                raise ValueError(msg)
            self._client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=chroma_cloud_api_key,
                settings=_settings,
            )

        else:
            self._client = chromadb.Client(settings=_settings)

        self._embedding_function = embedding_function
        self._chroma_collection: chromadb.Collection | None = None
        self._collection_name = collection_name
        self._collection_metadata = collection_metadata
        self._collection_configuration = collection_configuration
        if create_collection_if_not_exists:
            self.__ensure_collection()
        else:
            self._chroma_collection = self._client.get_collection(name=collection_name)
        self.override_relevance_score_fn = relevance_score_fn

    def __ensure_collection(self) -> None:
        """Ensure that the collection exists or create it."""
        self._chroma_collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata=self._collection_metadata,
            configuration=self._collection_configuration,
        )

    @property
    def _collection(self) -> chromadb.Collection:
        """Returns the underlying Chroma collection or throws an exception."""
        if self._chroma_collection is None:
            msg = (
                "Chroma collection not initialized. "
                "Use `reset_collection` to re-create and initialize the collection. "
            )
            raise ValueError(
                msg,
            )
        return self._chroma_collection

    @property
    def embeddings(self) -> Embeddings | None:
        """Access the query embedding object."""
        return self._embedding_function

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document] | chromadb.QueryResult:
        """Query the chroma collection.

        Args:
            query_texts: List of query texts.
            query_embeddings: List of query embeddings.
            n_results: Number of results to return.
            where: dict used to filter results by metadata.
                    E.g. {"color" : "red"}.
            where_document: dict used to filter by the document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `n_results` nearest neighbor embeddings for provided
            query_embeddings or query_texts.

        See more: https://docs.trychroma.com/reference/py-collection#query
        """
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,  # type: ignore[arg-type]
            n_results=n_results,
            where=where,  # type: ignore[arg-type]
            where_document=where_document,  # type: ignore[arg-type]
            **kwargs,
        )

    @staticmethod
    def encode_image(uri: str) -> str:
        """Get base64 string from image URI."""
        with Path(uri).open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def fork(self, new_name: str) -> Chroma:
        """Fork this vector store.

        Args:
            new_name: New name for the forked store.

        Returns:
            A new Chroma store forked from this vector store.

        """
        forked_collection = self._collection.fork(new_name=new_name)
        return Chroma(
            client=self._client,
            embedding_function=self._embedding_function,
            collection_name=forked_collection.name,
        )

    def add_images(
        self,
        uris: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Run more images through the embeddings and add to the `VectorStore`.

        Args:
            uris: File path to the image.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs. (Items without IDs will be assigned UUIDs)

        Returns:
            List of IDs of the added images.

        Raises:
            ValueError: When metadata is incorrect.
        """
        # Map from uris to b64 encoded strings
        b64_texts = [self.encode_image(uri=uri) for uri in uris]
        # Populate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in uris]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]
        embeddings = None
        # Set embeddings
        if self._embedding_function is not None and hasattr(
            self._embedding_function,
            "embed_image",
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
                images_with_metadatas = [b64_texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,  # type: ignore[arg-type]
                        documents=images_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg) from e
                    raise e
            if empty_ids:
                images_without_metadatas = [b64_texts[j] for j in empty_ids]
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
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Texts to add to the `VectorStore`.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs. (Items without IDs will be assigned UUIDs)
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: When metadata is incorrect.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

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
                    if embeddings is not None and len(embeddings) > 0
                    else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,  # type: ignore[arg-type]
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg) from e
                    raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,  # type: ignore[arg-type]
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                ids=ids,
            )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search with Chroma.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            filter: Filter by metadata.
            where_document: dict used to filter by the document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `Document` objects most similar to the query vector.
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return _results_to_docs(results)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector and similarity score.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k: Number of Documents to return.
            filter: Filter by metadata.
            where_document: dict used to filter by the documents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text and relevance score
            in float for each. Lower score represents more similarity.
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return _results_to_docs_and_scores(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            where_document: dict used to filter by document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text and
            distance in float for each. Lower score represents more similarity.
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )

        return _results_to_docs_and_scores(results)

    def similarity_search_with_vectors(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, np.ndarray]]:
        """Run similarity search with Chroma with vectors.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            where_document: dict used to filter by the document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text and
            embedding vectors for each.
        """
        include = ["documents", "metadatas", "embeddings"]
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=include,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=include,
                **kwargs,
            )

        return _results_to_docs_and_vectors(results)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on collections distance metric.

        The most similar documents will have the lowest relevance score. Default
        relevance score function is Euclidean distance. Distance metric must be
        provided in `collection_configuration` during initialization of Chroma object.
        Example: collection_configuration={"hnsw": {"space": "cosine"}}.
        Available distance metrics are: 'cosine', 'l2' and 'ip'.

        Returns:
            The relevance score function.

        Raises:
            ValueError: If the distance metric is not supported.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        hnsw_config = self._collection.configuration.get("hnsw")
        hnsw_distance: str | None = hnsw_config.get("space") if hnsw_config else None

        spann_config = self._collection.configuration.get("spann")
        spann_distance: str | None = spann_config.get("space") if spann_config else None

        distance = hnsw_distance or spann_distance

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        if distance == "l2":
            return self._euclidean_relevance_score_fn
        if distance == "ip":
            return self._max_inner_product_relevance_score_fn
        msg = (
            "No supported normalization function"
            f" for distance metric of type: {distance}."
            "Consider providing relevance_score_fn to Chroma constructor."
        )
        raise ValueError(
            msg,
        )

    def similarity_search_by_image(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar images based on the given image URI.

        Args:
            uri: URI of the image to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments to pass to function.


        Returns:
            List of Images most similar to the provided image. Each element in list is a
            LangChain Document Object. The page content is b64 encoded image, metadata
            is default or as defined by user.

        Raises:
            ValueError: If the embedding function does not support image embeddings.
        """
        if self._embedding_function is not None and hasattr(
            self._embedding_function, "embed_image"
        ):
            # Obtain image embedding
            # Assuming embed_image returns a single embedding
            image_embedding = self._embedding_function.embed_image(uris=[uri])

            # Perform similarity search based on the obtained embedding
            return self.similarity_search_by_vector(
                embedding=image_embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        msg = "The embedding function must support image embedding."
        raise ValueError(msg)

    def similarity_search_by_image_with_relevance_score(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar images based on the given image URI.

        Args:
            uri: URI of the image to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments to pass to function.

        Returns:
            List of tuples containing documents similar to the query image and their
            similarity scores. 0th element in each tuple is a LangChain Document Object.
            The page content is b64 encoded img, metadata is default or defined by user.

        Raises:
            ValueError: If the embedding function does not support image embeddings.
        """
        if self._embedding_function is not None and hasattr(
            self._embedding_function, "embed_image"
        ):
            # Obtain image embedding
            # Assuming embed_image returns a single embedding
            image_embedding = self._embedding_function.embed_image(uris=[uri])

            # Perform similarity search based on the obtained embedding
            return self.similarity_search_by_vector_with_relevance_scores(
                embedding=image_embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        msg = "The embedding function must support image embedding."
        raise ValueError(msg)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            where_document: dict used to filter by the document contents.
                e.g. `{"$contains": "hello"}`.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            where_document: dict used to filter by the document contents.
                e.g. `{"$contains": "hello"}`.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.

        Raises:
            ValueError: If the embedding function is not provided.
        """
        if self._embedding_function is None:
            msg = "For MMR search, you must specify an embedding function on creation."
            raise ValueError(
                msg,
            )

        embedding = self._embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            where_document=where_document,
        )

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)
        self._chroma_collection = None

    def reset_collection(self) -> None:
        """Resets the collection.

        Resets the collection by deleting the collection and recreating an empty one.
        """
        self.delete_collection()
        self.__ensure_collection()

    def get(
        self,
        ids: str | list[str] | None = None,
        where: Where | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: WhereDocument | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Gets the collection.

        Args:
            ids: The ids of the embeddings to get. Optional.
            where: A Where type dict used to filter results by.
                   E.g. `{"$and": [{"color": "red"}, {"price": 4.20}]}` Optional.
            limit: The number of documents to return. Optional.
            offset: The offset to start returning results from.
                    Useful for paging results with limit. Optional.
            where_document: A WhereDocument type dict used to filter by the documents.
                            E.g. `{"$contains": "hello"}`. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.

        Returns:
            A dict with the keys `"ids"`, `"embeddings"`, `"metadatas"`, `"documents"`.
        """
        kwargs = {
            "ids": ids,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
        }

        if include is not None:
            kwargs["include"] = include

        return self._collection.get(**kwargs)  # type: ignore[arg-type, return-value]

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.

        !!! version-added "Added in 0.2.1"
        """
        results = self.get(ids=list(ids))
        return [
            Document(page_content=doc, metadata=meta, id=doc_id)
            for doc, meta, doc_id in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
                strict=False,
            )
        ]

    def update_document(self, document_id: str, document: Document) -> None:
        """Update a document in the collection.

        Args:
            document_id: ID of the document to update.
            document: Document to update.
        """
        return self.update_documents([document_id], [document])

    def update_documents(self, ids: list[str], documents: list[Document]) -> None:
        """Update a document in the collection.

        Args:
            ids: List of ids of the document to update.
            documents: List of documents to update.

        Raises:
            ValueError: If the embedding function is not provided.
        """
        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        if self._embedding_function is None:
            msg = "For update, you must specify an embedding function on creation."
            raise ValueError(
                msg,
            )
        embeddings = self._embedding_function.embed_documents(text)

        if hasattr(
            self._client,
            "get_max_batch_size",
        ) or hasattr(  # for Chroma 0.5.1 and above
            self._client,
            "max_batch_size",
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._client,
                ids=ids,
                metadatas=metadata,  # type: ignore[arg-type]
                documents=text,
                embeddings=embeddings,  # type: ignore[arg-type]
            ):
                self._collection.update(
                    ids=batch[0],
                    embeddings=batch[1],
                    documents=batch[3],
                    metadatas=batch[2],
                )
        else:
            self._collection.update(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=text,
                metadatas=metadata,  # type: ignore[arg-type]
            )

    @classmethod
    def from_texts(
        cls: type[Chroma],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.ClientAPI | None = None,
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        *,
        ssl: bool = False,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts: List of texts to add to the collection.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            host: Hostname of a deployed Chroma server.
            port: Connection port for a deployed Chroma server.
                    Default is 8000.
            ssl: Whether to establish an SSL connection with a deployed Chroma server.
                    Default is False.
            headers: HTTP headers to send to a deployed Chroma server.
            chroma_cloud_api_key: Chroma Cloud API key.
            tenant: Tenant ID. Required for Chroma Cloud connections.
                    Default is 'default_tenant' for local Chroma servers.
            database: Database name. Required for Chroma Cloud connections.
                    Default is 'default_database'.
            embedding: Embedding function.
            metadatas: List of metadatas.
            ids: List of document IDs.
            client_settings: Chroma client settings.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/python/client
            collection_metadata: Collection configurations.
            collection_configuration: Index configuration for the collection.

            kwargs: Additional keyword arguments to initialize a Chroma client.

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            chroma_cloud_api_key=chroma_cloud_api_key,
            tenant=tenant,
            database=database,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            collection_configuration=collection_configuration,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]
        if hasattr(
            chroma_collection._client,
            "get_max_batch_size",
        ) or hasattr(  # for Chroma 0.5.1 and above
            chroma_collection._client,
            "max_batch_size",
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=chroma_collection._client,
                ids=ids,
                metadatas=metadatas,  # type: ignore[arg-type]
                documents=texts,
            ):
                chroma_collection.add_texts(
                    texts=batch[3] if batch[3] else [],
                    metadatas=batch[2] if batch[2] else None,  # type: ignore[arg-type]
                    ids=batch[0],
                )
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: type[Chroma],
        documents: list[Document],
        embedding: Embeddings | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.ClientAPI | None = None,  # Add this line
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        *,
        ssl: bool = False,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            host: Hostname of a deployed Chroma server.
            port: Connection port for a deployed Chroma server. Default is 8000.
            ssl: Whether to establish an SSL connection with a deployed Chroma server.
            headers: HTTP headers to send to a deployed Chroma server.
            chroma_cloud_api_key: Chroma Cloud API key.
            tenant: Tenant ID. Required for Chroma Cloud connections.
                    Default is 'default_tenant' for local Chroma servers.
            database: Database name. Required for Chroma Cloud connections.
                    Default is 'default_database'.
            ids: List of document IDs.
            documents: List of documents to add to the `VectorStore`.
            embedding: Embedding function.
            client_settings: Chroma client settings.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/python/client
            collection_metadata: Collection configurations.
            collection_configuration: Index configuration for the collection.

            kwargs: Additional keyword arguments to initialize a Chroma client.

        Returns:
            Chroma: Chroma vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            chroma_cloud_api_key=chroma_cloud_api_key,
            tenant=tenant,
            database=database,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            collection_configuration=collection_configuration,
            **kwargs,
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments.
        """
        self._collection.delete(ids=ids, **kwargs)
