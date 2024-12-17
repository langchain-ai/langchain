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
)

import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import chromadb
    import chromadb.config
    from chromadb.api.types import ID, OneOrMany, Where, WhereDocument

logger = logging.getLogger()
DEFAULT_K = 4  # Number of Documents to return.


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


@deprecated(since="0.2.9", removal="1.0", alternative_import="langchain_chroma.Chroma")
class Chroma(VectorStore):
    """`ChromaDB` vector store.

    To use, you should have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma("langchain_store", embeddings)
    """

    _LANGCHAIN_DEFAULT_COLLECTION_NAME: str = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.Client] = None,  # type: ignore[valid-type]
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize with a Chroma client."""
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

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
                if client_settings.persist_directory is not None:
                    # Maintain backwards compatibility with chromadb < 0.4.0
                    major, minor, _ = chromadb.__version__.split(".")
                    if int(major) == 0 and int(minor) < 4:
                        client_settings.chroma_db_impl = "duckdb+parquet"

                _client_settings = client_settings
            elif persist_directory:
                # Maintain backwards compatibility with chromadb < 0.4.0
                major, minor, _ = chromadb.__version__.split(".")
                if int(major) == 0 and int(minor) < 4:
                    _client_settings = chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                    )
                else:
                    _client_settings = chromadb.config.Settings(is_persistent=True)
                _client_settings.persist_directory = persist_directory
            else:
                _client_settings = chromadb.config.Settings()
            self._client_settings = _client_settings  # type: ignore[has-type]
            self._client = chromadb.Client(_client_settings)  # type: ignore[has-type]
            self._persist_directory = (  # type: ignore[has-type]
                _client_settings.persist_directory or persist_directory
            )

        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(  # type: ignore[has-type]
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Optional[Embeddings]:
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
    ) -> List[Document]:
        """Query the chroma collection."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        return self._collection.query(  # type: ignore[return-value]
            query_texts=query_texts,
            query_embeddings=query_embeddings,  # type: ignore[arg-type]
            n_results=n_results,
            where=where,  # type: ignore[arg-type]
            where_document=where_document,  # type: ignore[arg-type]
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
            uris List[str]: File path to the image.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added images.
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
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,
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
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
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
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
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
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
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
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
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
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            where_document=where_document,
        )
        return docs

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)  # type: ignore[has-type]

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
        kwargs = {
            "ids": ids,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
        }

        if include is not None:
            kwargs["include"] = include

        return self._collection.get(**kwargs)  # type: ignore[return-value, arg-type, arg-type, arg-type, arg-type, arg-type]

    @deprecated(
        since="0.1.17",
        message=(
            "Since Chroma 0.4.x the manual persistence method is no longer "
            "supported as docs are automatically persisted."
        ),
        removal="1.0",
    )
    def persist(self) -> None:
        """Persist the collection.

        This can be used to explicitly persist the data to disk.
        It will also be called automatically when the object is destroyed.

        Since Chroma 0.4.x the manual persistence method is no longer
        supported as docs are automatically persisted.
        """
        if self._persist_directory is None:  # type: ignore[has-type]
            raise ValueError(
                "You must specify a persist_directory on"
                "creation to persist the collection."
            )
        import chromadb

        # Maintain backwards compatibility with chromadb < 0.4.0
        major, minor, _ = chromadb.__version__.split(".")
        if int(major) == 0 and int(minor) < 4:
            self._client.persist()  # type: ignore[has-type]

    def update_document(self, document_id: str, document: Document) -> None:
        """Update a document in the collection.

        Args:
            document_id (str): ID of the document to update.
            document (Document): Document to update.
        """
        return self.update_documents([document_id], [document])

    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
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

        if hasattr(
            self._collection._client,
            "get_max_batch_size",  # for Chroma 0.5.1 and above
        ) or hasattr(
            self._collection._client, "max_batch_size"
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._collection._client,
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
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # type: ignore[valid-type]
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

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
            chroma_collection._client,  # type: ignore[has-type]
            "get_max_batch_size",  # for Chroma 0.5.1 and above
        ) or hasattr(
            chroma_collection._client,  # type: ignore[has-type]
            "max_batch_size",
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=chroma_collection._client,  # type: ignore[has-type]
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
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[  # type: ignore[valid-type]
            chromadb.Client
        ] = None,  # Add this line  # type: ignore[valid-type]
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
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
        """
        self._collection.delete(ids=ids, **kwargs)

    def __len__(self) -> int:
        """Count the number of documents in the collection."""
        return self._collection.count()
