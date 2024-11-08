"""This is the langchain_chroma.vectorstores module.

It contains the Chroma class which is a vector store for handling various tasks.
"""

from __future__ import annotations

import base64
import logging
import uuid
from typing import (
    TYPE_CHECKING,
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

import chromadb
import chromadb.config
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from chromadb.api.types import ID, OneOrMany, Where, WhereDocument

logger = logging.getLogger()
DEFAULT_K = 4  # Number of Documents to return.


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
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
        )
    ]


Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
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
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
        k: Number of Documents to return. Defaults to 4.

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
        Install ``chromadb``, ``langchain-chroma`` packages:

        .. code-block:: bash

            pip install -qU chromadb langchain-chroma

    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        embedding_function: Embeddings
            Embedding function to use.

    Key init args — client params:
        client: Optional[Client]
            Chroma client to use.
        client_settings: Optional[chromadb.config.Settings]
            Chroma client settings.
        persist_directory: Optional[str]
            Directory to persist the collection.

    Instantiate:
        .. code-block:: python

            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings

            vector_store = Chroma(
                collection_name="foo",
                embedding_function=OpenAIEmbeddings(),
                # other params...
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Update Documents:
        .. code-block:: python

            updated_document = Document(
                page_content="qux",
                metadata={"bar": "baz"}
            )

            vector_store.update_documents(ids=["1"],documents=[updated_document])

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'baz': 'bar'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"baz": "bar"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * foo [{'baz': 'bar'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.000000] qux [{'bar': 'baz', 'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.335463] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'baz': 'bar'}, page_content='thud')]

    """  # noqa: E501

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
    ) -> None:
        """Initialize with a Chroma client.

        Args:
            collection_name: Name of the collection to create.
            embedding_function: Embedding class object. Used to embed texts.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
        """
        if client is not None:
            self._client_settings = client_settings
            self._client = client
            self._persist_directory = persist_directory
        else:
            if client_settings:
                # If client_settings is provided with persist_directory specified,
                # then it is "in-memory and persisting to disk" mode.
                client_settings.persist_directory = (
                    persist_directory or client_settings.persist_directory
                )

                _client_settings = client_settings
            elif persist_directory:
                _client_settings = chromadb.config.Settings(is_persistent=True)
                _client_settings.persist_directory = persist_directory
            else:
                _client_settings = chromadb.config.Settings()
            self._client_settings = _client_settings
            self._client = chromadb.Client(_client_settings)
            self._persist_directory = (
                _client_settings.persist_directory or persist_directory
            )

        self._embedding_function = embedding_function
        self._chroma_collection: Optional[chromadb.Collection] = None
        self._collection_name = collection_name
        self._collection_metadata = collection_metadata
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
        )

    @property
    def _collection(self) -> chromadb.Collection:
        """Returns the underlying Chroma collection or throws an exception."""
        if self._chroma_collection is None:
            raise ValueError(
                "Chroma collection not initialized. "
                "Use `reset_collection` to re-create and initialize the collection. "
            )
        return self._chroma_collection

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object."""
        return self._embedding_function

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[List[Document], chromadb.QueryResult]:
        """Query the chroma collection.

        Args:
            query_texts: List of query texts.
            query_embeddings: List of query embeddings.
            n_results: Number of results to return. Defaults to 4.
            where: dict used to filter results by
                    e.g. {"color" : "red", "price": 4.20}.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `n_results` nearest neighbor embeddings for provided
            query_embeddings or query_texts.

        See more: https://docs.trychroma.com/reference/py-collection#query
        """
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,  # type: ignore
            n_results=n_results,
            where=where,  # type: ignore
            where_document=where_document,  # type: ignore
            **kwargs,
        )

    def encode_image(self, uri: str) -> str:
        """Get base64 string from image URI."""
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more images through the embeddings and add to the vectorstore.

        Args:
            uris: File path to the image.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs.
            kwargs: Additional keyword arguments to pass.

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
        embeddings = None
        # Set embeddings
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
                images_with_metadatas = [b64_texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore
                        embeddings=embeddings_with_metadatas,  # type: ignore
                        documents=images_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
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
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Texts to add to the vectorstore.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs.
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: When metadata is incorrect.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
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
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore
                        embeddings=embeddings_with_metadatas,  # type: ignore
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
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
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,  # type: ignore
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore
                documents=texts,
                ids=ids,
            )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Chroma.

        Args:
            query: Query text to search for.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return _results_to_docs(results)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector and similarity score.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of documents most similar to the query text and relevance score
            in float for each. Lower score represents more similarity.
        """
        results = self.__query_collection(
            query_embeddings=embedding,
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
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query: Query text to search for.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
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

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on collections distance metric.

        The most similar documents will have the lowest relevance score. Default
        relevance score function is euclidean distance. Distance metric must be
        provided in `collection_metadata` during initizalition of Chroma object.
        Example: collection_metadata={"hnsw:space": "cosine"}. Available distance
        metrics are: 'cosine', 'l2' and 'ip'.

        Returns:
            The relevance score function.

        Raises:
            ValueError: If the distance metric is not supported.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "l2"
        distance_key = "hnsw:space"
        metadata = self._collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        elif distance == "l2":
            return self._euclidean_relevance_score_fn
        elif distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )

    def similarity_search_by_image(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar images based on the given image URI.

        Args:
            uri (str): URI of the image to search for.
            k (int, optional): Number of results to return. Defaults to DEFAULT_K.
            filter (Optional[Dict[str, str]], optional): Filter by metadata.
            **kwargs (Any): Additional arguments to pass to function.


        Returns:
            List of Images most similar to the provided image.
            Each element in list is a Langchain Document Object.
            The page content is b64 encoded image, metadata is default or
            as defined by user.

        Raises:
            ValueError: If the embedding function does not support image embeddings.
        """
        if self._embedding_function is None or not hasattr(
            self._embedding_function, "embed_image"
        ):
            raise ValueError("The embedding function must support image embedding.")

        # Obtain image embedding
        # Assuming embed_image returns a single embedding
        image_embedding = self._embedding_function.embed_image(uris=[uri])

        # Perform similarity search based on the obtained embedding
        results = self.similarity_search_by_vector(
            embedding=image_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )

        return results

    def similarity_search_by_image_with_relevance_score(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar images based on the given image URI.

        Args:
            uri (str): URI of the image to search for.
            k (int, optional): Number of results to return.
            Defaults to DEFAULT_K.
            filter (Optional[Dict[str, str]], optional): Filter by metadata.
            **kwargs (Any): Additional arguments to pass to function.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing documents similar
            to the query image and their similarity scores.
            0th element in each tuple is a Langchain Document Object.
            The page content is b64 encoded img, metadata is default or defined by user.

        Raises:
            ValueError: If the embedding function does not support image embeddings.
        """
        if self._embedding_function is None or not hasattr(
            self._embedding_function, "embed_image"
        ):
            raise ValueError("The embedding function must support image embedding.")

        # Obtain image embedding
        # Assuming embed_image returns a single embedding
        image_embedding = self._embedding_function.embed_image(uris=[uri])

        # Perform similarity search based on the obtained embedding
        results = self.similarity_search_by_vector_with_relevance_scores(
            embedding=image_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )

        return results

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to
                20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self.__query_collection(
            query_embeddings=embedding,
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

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
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
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of Documents selected by maximal marginal relevance.

        Raises:
            ValueError: If the embedding function is not provided.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
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
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Gets the collection.

        Args:
            ids: The ids of the embeddings to get. Optional.
            where: A Where type dict used to filter results by.
                   E.g. `{"$and": [{"color": "red"}, {"price": 4.20}]}` Optional.
            limit: The number of documents to return. Optional.
            offset: The offset to start returning results from.
                    Useful for paging results with limit. Optional.
            where_document: A WhereDocument type dict used to filter by the documents.
                            E.g. `{$contains: "hello"}`. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.

        Return:
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

        return self._collection.get(**kwargs)  # type: ignore

    def update_document(self, document_id: str, document: Document) -> None:
        """Update a document in the collection.

        Args:
            document_id: ID of the document to update.
            document: Document to update.
        """
        return self.update_documents([document_id], [document])

    # type: ignore
    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
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
            raise ValueError(
                "For update, you must specify an embedding function on creation."
            )
        embeddings = self._embedding_function.embed_documents(text)

        if hasattr(
            self._collection._client, "get_max_batch_size"
        ) or hasattr(  # for Chroma 0.5.1 and above
            self._collection._client, "max_batch_size"
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._collection._client,
                ids=ids,
                metadatas=metadata,  # type: ignore
                documents=text,
                embeddings=embeddings,  # type: ignore
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
                embeddings=embeddings,  # type: ignore
                documents=text,
                metadatas=metadata,  # type: ignore
            )

    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.ClientAPI] = None,
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts: List of texts to add to the collection.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            embedding: Embedding function. Defaults to None.
            metadatas: List of metadatas. Defaults to None.
            ids: List of document IDs. Defaults to None.
            client_settings: Chroma client settings.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            collection_metadata: Collection configurations.
                                                  Defaults to None.
            kwargs: Additional keyword arguments to initialize a Chroma client.

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if hasattr(
            chroma_collection._client, "get_max_batch_size"
        ) or hasattr(  # for Chroma 0.5.1 and above
            chroma_collection._client, "max_batch_size"
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=chroma_collection._client,
                ids=ids,
                metadatas=metadatas,  # type: ignore
                documents=texts,
            ):
                chroma_collection.add_texts(
                    texts=batch[3] if batch[3] else [],
                    metadatas=batch[2] if batch[2] else None,  # type: ignore
                    ids=batch[0],
                )
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.ClientAPI] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            ids : List of document IDs. Defaults to None.
            documents: List of documents to add to the vectorstore.
            embedding: Embedding function. Defaults to None.
            client_settings: Chroma client settings.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            collection_metadata: Collection configurations.
                                                  Defaults to None.
            kwargs: Additional keyword arguments to initialize a Chroma client.

        Returns:
            Chroma: Chroma vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.id if doc.id else "" for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments.
        """
        self._collection.delete(ids=ids, **kwargs)
