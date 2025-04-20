import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

VST = TypeVar("VST", bound="VectorStore")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from momento import PreviewVectorIndexClient


class MomentoVectorIndex(VectorStore):
    """`Momento Vector Index` (MVI) vector store.

    Momento Vector Index is a serverless vector index that can be used to store and
    search vectors. To use you should have the ``momento`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenAIEmbeddings
            from langchain_community.vectorstores import MomentoVectorIndex
            from momento import (
                CredentialProvider,
                PreviewVectorIndexClient,
                VectorIndexConfigurations,
            )

            vectorstore = MomentoVectorIndex(
                embedding=OpenAIEmbeddings(),
                client=PreviewVectorIndexClient(
                    VectorIndexConfigurations.Default.latest(),
                    credential_provider=CredentialProvider.from_environment_variable(
                        "MOMENTO_API_KEY"
                    ),
                ),
                index_name="my-index",
            )
    """

    def __init__(
        self,
        embedding: Embeddings,
        client: "PreviewVectorIndexClient",
        index_name: str = "default",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        text_field: str = "text",
        ensure_index_exists: bool = True,
        **kwargs: Any,
    ):
        """Initialize a Vector Store backed by Momento Vector Index.

        Args:
            embedding (Embeddings): The embedding function to use.
            configuration (VectorIndexConfiguration): The configuration to initialize
                the Vector Index with.
            credential_provider (CredentialProvider): The credential provider to
                authenticate the Vector Index with.
            index_name (str, optional): The name of the index to store the documents in.
                Defaults to "default".
            distance_strategy (DistanceStrategy, optional): The distance strategy to
                use. If you select DistanceStrategy.EUCLIDEAN_DISTANCE, Momento uses
                the squared Euclidean distance. Defaults to DistanceStrategy.COSINE.
            text_field (str, optional): The name of the metadata field to store the
                original text in. Defaults to "text".
            ensure_index_exists (bool, optional): Whether to ensure that the index
                exists before adding documents to it. Defaults to True.
        """
        try:
            from momento import PreviewVectorIndexClient
        except ImportError:
            raise ImportError(
                "Could not import momento python package. "
                "Please install it with `pip install momento`."
            )

        self._client: PreviewVectorIndexClient = client
        self._embedding = embedding
        self.index_name = index_name
        self.__validate_distance_strategy(distance_strategy)
        self.distance_strategy = distance_strategy
        self.text_field = text_field
        self._ensure_index_exists = ensure_index_exists

    @staticmethod
    def __validate_distance_strategy(distance_strategy: DistanceStrategy) -> None:
        if distance_strategy not in [
            DistanceStrategy.COSINE,
            DistanceStrategy.MAX_INNER_PRODUCT,
            DistanceStrategy.MAX_INNER_PRODUCT,
        ]:
            raise ValueError(f"Distance strategy {distance_strategy} not implemented.")

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _create_index_if_not_exists(self, num_dimensions: int) -> bool:
        """Create index if it does not exist."""
        from momento.requests.vector_index import SimilarityMetric
        from momento.responses.vector_index import CreateIndex

        similarity_metric = None
        if self.distance_strategy == DistanceStrategy.COSINE:
            similarity_metric = SimilarityMetric.COSINE_SIMILARITY
        elif self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            similarity_metric = SimilarityMetric.INNER_PRODUCT
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarity_metric = SimilarityMetric.EUCLIDEAN_SIMILARITY
        else:
            logger.error(f"Distance strategy {self.distance_strategy} not implemented.")
            raise ValueError(
                f"Distance strategy {self.distance_strategy} not implemented."
            )

        response = self._client.create_index(
            self.index_name, num_dimensions, similarity_metric
        )
        if isinstance(response, CreateIndex.Success):
            return True
        elif isinstance(response, CreateIndex.IndexAlreadyExists):
            return False
        elif isinstance(response, CreateIndex.Error):
            logger.error(f"Error creating index: {response.inner_exception}")
            raise response.inner_exception
        else:
            logger.error(f"Unexpected response: {response}")
            raise Exception(f"Unexpected response: {response}")

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings to add to the vectorstore.
            metadatas (Optional[List[dict]]): Optional list of metadatas associated with
                the texts.
            kwargs (Any): Other optional parameters. Specifically:
            - ids (List[str], optional): List of ids to use for the texts.
                Defaults to None, in which case uuids are generated.

        Returns:
            List[str]: List of ids from adding the texts into the vectorstore.
        """
        from momento.requests.vector_index import Item
        from momento.responses.vector_index import UpsertItemBatch

        texts = list(texts)

        if len(texts) == 0:
            return []

        if metadatas is not None:
            for metadata, text in zip(metadatas, texts):
                metadata[self.text_field] = text
        else:
            metadatas = [{self.text_field: text} for text in texts]

        try:
            embeddings = self._embedding.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self._embedding.embed_query(x) for x in texts]

        # Create index if it does not exist.
        # We assume that if it does exist, then it was created with the desired number
        # of dimensions and similarity metric.
        if self._ensure_index_exists:
            self._create_index_if_not_exists(len(embeddings[0]))

        if "ids" in kwargs:
            ids = kwargs["ids"]
            if len(ids) != len(embeddings):
                raise ValueError("Number of ids must match number of texts")
        else:
            ids = [str(uuid4()) for _ in range(len(embeddings))]

        batch_size = 128
        for i in range(0, len(embeddings), batch_size):
            start = i
            end = min(i + batch_size, len(embeddings))
            items = [
                Item(id=id, vector=vector, metadata=metadata)
                for id, vector, metadata in zip(
                    ids[start:end],
                    embeddings[start:end],
                    metadatas[start:end],
                )
            ]

            response = self._client.upsert_item_batch(self.index_name, items)
            if isinstance(response, UpsertItemBatch.Success):
                pass
            elif isinstance(response, UpsertItemBatch.Error):
                raise response.inner_exception
            else:
                raise Exception(f"Unexpected response: {response}")

        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID.

        Args:
            ids (List[str]): List of ids to delete.
            kwargs (Any): Other optional parameters (unused)

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        from momento.responses.vector_index import DeleteItemBatch

        if ids is None:
            return True
        response = self._client.delete_item_batch(self.index_name, ids)
        return isinstance(response, DeleteItemBatch.Success)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents to the query string.

        Args:
            query (str): The query string to search for.
            k (int, optional): The number of results to return. Defaults to 4.

        Returns:
            List[Document]: A list of documents that are similar to the query.
        """
        res = self.similarity_search_with_score(query=query, k=k, **kwargs)
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query string.

        Args:
            query (str): The query string to search for.
            k (int, optional): The number of results to return. Defaults to 4.
            kwargs (Any): Vector Store specific search parameters. The following are
                forwarded to the Momento Vector Index:
            - top_k (int, optional): The number of results to return.

        Returns:
            List[Tuple[Document, float]]: A list of tuples of the form
                (Document, score).
        """
        embedding = self._embedding.embed_query(query)

        results = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return results

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query vector.

        Args:
            embedding (List[float]): The query vector to search for.
            k (int, optional): The number of results to return. Defaults to 4.
            kwargs (Any): Vector Store specific search parameters. The following are
                forwarded to the Momento Vector Index:
            - top_k (int, optional): The number of results to return.

        Returns:
            List[Tuple[Document, float]]: A list of tuples of the form
                (Document, score).
        """
        from momento.requests.vector_index import ALL_METADATA
        from momento.responses.vector_index import Search

        if "top_k" in kwargs:
            k = kwargs["k"]
        filter_expression = kwargs.get("filter_expression", None)
        response = self._client.search(
            self.index_name,
            embedding,
            top_k=k,
            metadata_fields=ALL_METADATA,
            filter_expression=filter_expression,
        )

        if not isinstance(response, Search.Success):
            return []

        results = []
        for hit in response.hits:
            text = cast(str, hit.metadata.pop(self.text_field))
            doc = Document(page_content=text, metadata=hit.metadata)
            pair = (doc, hit.score)
            results.append(pair)

        return results

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents to the query vector.

        Args:
            embedding (List[float]): The query vector to search for.
            k (int, optional): The number of results to return. Defaults to 4.

        Returns:
            List[Document]: A list of documents that are similar to the query.
        """
        results = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _ in results]

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
        from momento.requests.vector_index import ALL_METADATA
        from momento.responses.vector_index import SearchAndFetchVectors

        filter_expression = kwargs.get("filter_expression", None)
        response = self._client.search_and_fetch_vectors(
            self.index_name,
            embedding,
            top_k=fetch_k,
            metadata_fields=ALL_METADATA,
            filter_expression=filter_expression,
        )

        if isinstance(response, SearchAndFetchVectors.Success):
            pass
        elif isinstance(response, SearchAndFetchVectors.Error):
            logger.error(f"Error searching and fetching vectors: {response}")
            return []
        else:
            logger.error(f"Unexpected response: {response}")
            raise Exception(f"Unexpected response: {response}")

        mmr_selected = maximal_marginal_relevance(
            query_embedding=np.array([embedding], dtype=np.float32),
            embedding_list=[hit.vector for hit in response.hits],
            lambda_mult=lambda_mult,
            k=k,
        )
        selected = [response.hits[i].metadata for i in mmr_selected]
        return [
            Document(page_content=metadata.pop(self.text_field, ""), metadata=metadata)
            for metadata in selected
        ]

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
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return the Vector Store initialized from texts and embeddings.

        Args:
            cls (Type[VST]): The Vector Store class to use to initialize
                the Vector Store.
            texts (List[str]): The texts to initialize the Vector Store with.
            embedding (Embeddings): The embedding function to use.
            metadatas (Optional[List[dict]], optional): The metadata associated with
                the texts. Defaults to None.
            kwargs (Any): Vector Store specific parameters. The following are forwarded
                to the Vector Store constructor and required:
            - index_name (str, optional): The name of the index to store the documents
                in. Defaults to "default".
            - text_field (str, optional): The name of the metadata field to store the
                original text in. Defaults to "text".
            - distance_strategy (DistanceStrategy, optional): The distance strategy to
                use. Defaults to DistanceStrategy.COSINE. If you select
                DistanceStrategy.EUCLIDEAN_DISTANCE, Momento uses the squared
                Euclidean distance.
            - ensure_index_exists (bool, optional): Whether to ensure that the index
                exists before adding documents to it. Defaults to True.
            Additionally you can either pass in a client or an API key
            - client (PreviewVectorIndexClient): The Momento Vector Index client to use.
            - api_key (Optional[str]): The configuration to use to initialize
                the Vector Index with. Defaults to None. If None, the configuration
                is initialized from the environment variable `MOMENTO_API_KEY`.

        Returns:
            VST: Momento Vector Index vector store initialized from texts and
                embeddings.
        """
        from momento import (
            CredentialProvider,
            PreviewVectorIndexClient,
            VectorIndexConfigurations,
        )

        if "client" in kwargs:
            client = kwargs.pop("client")
        else:
            supplied_api_key = kwargs.pop("api_key", None)
            api_key = supplied_api_key or get_from_env("api_key", "MOMENTO_API_KEY")
            client = PreviewVectorIndexClient(
                configuration=VectorIndexConfigurations.Default.latest(),
                credential_provider=CredentialProvider.from_string(api_key),
            )
        vector_db = cls(embedding=embedding, client=client, **kwargs)  # type: ignore[call-arg]
        vector_db.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return vector_db
