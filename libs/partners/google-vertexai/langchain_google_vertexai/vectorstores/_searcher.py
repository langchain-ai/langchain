import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchNeighbor,
    Namespace,
)
from google.cloud.storage import Bucket


class Searcher(ABC):
    """Abstract implementation of a similarity searcher."""

    @abstractmethod
    def find_neighbors(
        self,
        embeddings: List[List[float]],
        k: int = 4,
        filter_: List[Namespace] | None = None,
    ) -> List[List[Tuple[str, float]]]:
        """Finds the k closes neighbors of each instance of embeddings.

        Args:
            embedding: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: List of filters to apply.

        Returns:
            List of lists of Tuples (id, distance) for each embedding vector.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict] | None = None,
        **kwargs: Any,
    ):
        """ """
        raise NotImplementedError()

    def _postprocess_response(
        self, response: List[List[MatchNeighbor]]
    ) -> List[List[Tuple[str, float]]]:
        """Posproceses an endpoint response and converts it to a list of list of
        tuples instead of using vertexai objects.

        Args:
            response: Endpoint response.

        Returns:
            List of list of tuples of (id, distance).
        """
        return [
            [(neighbor.id, neighbor.distance) for neighbor in matching_neighbor_list]
            for matching_neighbor_list in response
        ]


class VectorSearchSearcher(Searcher):
    """ """

    def __init__(
        self,
        endpoint: MatchingEngineIndexEndpoint,
        index: MatchingEngineIndex,
        staging_bucket: Bucket | None = None,
    ) -> None:
        """Constructor.

        Args:
            endpoint: Endpoint that will be used to make find_neighbors requests.
            index: Underlying index deployed in that endpoint.
            staging_bucket: Necessary only if updating the index. Bucket where the
                embeddings and metadata will be staged.

        Raises:
            ValueError: If the index provided is not deployed in the endpoint.
        """
        super().__init__()
        self._endpoint = endpoint
        self._index = index
        self._deployed_index_id = self._get_deployed_index_id()
        self._staging_bucket = staging_bucket

    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        """ """

        if self._staging_bucket is None:
            raise ValueError(
                "In order to update a Vector Search index a staging bucket must"
                " be defined."
            )

        record_list = []
        for i, (idx, embedding) in enumerate(zip(ids, embeddings)):
            record: Dict[str, Any] = {"id": idx, "embedding": embedding}
            if metadatas is not None:
                record["metadata"] = metadatas[i]
            record_list.append(record)
        file_content = "\n".join([json.dumps(x) for x in record_list])

        filename_prefix = f"indexes/{uuid.uuid4()}"
        filename = f"{filename_prefix}/{time.time()}.json"
        blob = self._staging_bucket.blob(filename)
        blob.upload_from_string(data=file_content)

        self.index = self._index.update_embeddings(
            contents_delta_uri=f"gs://{self._staging_bucket.name}/{filename_prefix}/"
        )

    def _get_deployed_index_id(self) -> str:
        """Gets the deployed index id that matches with the provided index.

        Raises:
            ValueError if the index provided is not found in the endpoint.
        """
        for index in self._endpoint.deployed_indexes:
            if index.index == self._index.resource_name:
                return index.id

        raise ValueError(
            f"No index with id {self._index.resource_name} "
            f"deployed on endpoint "
            f"{self._endpoint.display_name}."
        )


class PublicEndpointVectorSearchSearcher(VectorSearchSearcher):
    """ """

    def find_neighbors(
        self,
        embeddings: List[List[float]],
        k: int = 4,
        filter_: List[Namespace] | None = None,
    ) -> List[List[Tuple[str, float]]]:
        """Finds the k closes neighbors of each instance of embeddings.

        Args:
            embedding: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: List of filters to apply.

        Returns:
            List of lists of Tuples (id, distance) for each embedding vector.
        """

        response = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=embeddings,
            num_neighbors=k,
            filter=filter_,
        )

        return self._postprocess_response(response)


class VPCVertexVectorStore(VectorSearchSearcher):
    """ """

    def find_neighbors(
        self,
        embeddings: List[List[float]],
        k: int = 4,
        filter_: List[Namespace] | None = None,
    ) -> List[List[Tuple[str, float]]]:
        """Finds the k closes neighbors of each instance of embeddings.

        Args:
            embedding: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: List of filters to apply.

        Returns:
            List of lists of Tuples (id, distance) for each embedding vector.
        """

        response = self._endpoint.match(
            deployed_index_id=self._deployed_index_id,
            queries=embeddings,
            num_neighbors=k,
            filter=filter_,
        )

        return self._postprocess_response(response)
