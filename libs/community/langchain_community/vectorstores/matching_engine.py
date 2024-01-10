from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.utilities.vertexai import get_client_info

if TYPE_CHECKING:
    from google.cloud import storage
    from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
        Namespace,
    )
    from google.oauth2.service_account import Credentials

    from langchain_community.embeddings import TensorflowHubEmbeddings

logger = logging.getLogger(__name__)


class MatchingEngine(VectorStore):
    """`Google Vertex AI Vector Search` (previously Matching Engine) vector store.

    While the embeddings are stored in the Matching Engine, the embedded
    documents will be stored in GCS.

    An existing Index and corresponding Endpoint are preconditions for
    using this module.

    See usage in docs/integrations/vectorstores/google_vertex_ai_vector_search.ipynb

    Note that this implementation is mostly meant for reading if you are
    planning to do a real time implementation. While reading is a real time
    operation, updating the index takes close to one hour."""

    def __init__(
        self,
        project_id: str,
        index: MatchingEngineIndex,
        endpoint: MatchingEngineIndexEndpoint,
        embedding: Embeddings,
        gcs_client: storage.Client,
        gcs_bucket_name: str,
        credentials: Optional[Credentials] = None,
        *,
        document_id_key: Optional[str] = None,
    ):
        """Google Vertex AI Vector Search (previously Matching Engine)
         implementation of the vector store.

        While the embeddings are stored in the Matching Engine, the embedded
        documents will be stored in GCS.

        An existing Index and corresponding Endpoint are preconditions for
        using this module.

        See usage in
        docs/integrations/vectorstores/google_vertex_ai_vector_search.ipynb.

        Note that this implementation is mostly meant for reading if you are
        planning to do a real time implementation. While reading is a real time
        operation, updating the index takes close to one hour.

        Attributes:
            project_id: The GCS project id.
            index: The created index class. See
                ~:func:`MatchingEngine.from_components`.
            endpoint: The created endpoint class. See
                ~:func:`MatchingEngine.from_components`.
            embedding: A :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
            gcs_client: The GCS client.
            gcs_bucket_name: The GCS bucket name.
            credentials (Optional): Created GCP credentials.
            document_id_key (Optional): Key for storing document ID in document
                metadata. If None, document ID will not be returned in document
                metadata.
        """
        super().__init__()
        self._validate_google_libraries_installation()

        self.project_id = project_id
        self.index = index
        self.endpoint = endpoint
        self.embedding = embedding
        self.gcs_client = gcs_client
        self.credentials = credentials
        self.gcs_bucket_name = gcs_bucket_name
        self.document_id_key = document_id_key

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _validate_google_libraries_installation(self) -> None:
        """Validates that Google libraries that are needed are installed."""
        try:
            from google.cloud import aiplatform, storage  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run `pip install --upgrade "
                "google-cloud-aiplatform google-cloud-storage`"
                "to use the MatchingEngine Vectorstore."
            )

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
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError(
                "texts and metadatas do not have the same length. Received "
                f"{len(texts)} texts and {len(metadatas)} metadatas."
            )
        logger.debug("Embedding documents.")
        embeddings = self.embedding.embed_documents(texts)
        jsons = []
        ids = []
        # Could be improved with async.
        for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
            id = str(uuid.uuid4())
            ids.append(id)
            json_: dict = {"id": id, "embedding": embedding}
            if metadatas is not None:
                json_["metadata"] = metadatas[idx]
            jsons.append(json_)
            self._upload_to_gcs(text, f"documents/{id}")

        logger.debug(f"Uploaded {len(ids)} documents to GCS.")

        # Creating json lines from the embedded documents.
        result_str = "\n".join([json.dumps(x) for x in jsons])

        filename_prefix = f"indexes/{uuid.uuid4()}"
        filename = f"{filename_prefix}/{time.time()}.json"
        self._upload_to_gcs(result_str, filename)
        logger.debug(
            f"Uploaded updated json with embeddings to "
            f"{self.gcs_bucket_name}/{filename}."
        )

        self.index = self.index.update_embeddings(
            contents_delta_uri=f"gs://{self.gcs_bucket_name}/{filename_prefix}/"
        )

        logger.debug("Updated index with new configuration.")

        return ids

    def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
        """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        blob.upload_from_string(data)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query and their cosine distance from the query.

        Args:
            query: String query look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional. A list of Namespaces for filtering
                the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        logger.debug(f"Embedding query {query}.")
        embedding_query = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding_query, k=k, filter=filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to the embedding and their cosine distance.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional. A list of Namespaces for filtering
                the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.

        """
        filter = filter or []

        # If the endpoint is public we use the find_neighbors function.
        if hasattr(self.endpoint, "_public_match_client") and (
            self.endpoint._public_match_client
        ):
            response = self.endpoint.find_neighbors(
                deployed_index_id=self._get_index_id(),
                queries=[embedding],
                num_neighbors=k,
                filter=filter,
            )
        else:
            response = self.endpoint.match(
                deployed_index_id=self._get_index_id(),
                queries=[embedding],
                num_neighbors=k,
                filter=filter,
            )

        logger.debug(f"Found {len(response)} matches.")

        if len(response) == 0:
            return []

        docs: List[Tuple[Document, float]] = []

        # I'm only getting the first one because queries receives an array
        # and the similarity_search method only receives one query. This
        # means that the match method will always return an array with only
        # one element.
        for result in response[0]:
            page_content = self._download_from_gcs(f"documents/{result.id}")
            # TODO: return all metadata.
            metadata = {}
            if self.document_id_key is not None:
                metadata[self.document_id_key] = result.id
            document = Document(
                page_content=page_content,
                metadata=metadata,
            )
            docs.append((document, result.distance))

        logger.debug("Downloaded documents for query.")

        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.
            filter: Optional. A list of Namespaces for filtering the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                 for more detail.

        Returns:
            A list of k matching documents.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )

        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to the embedding.

        Args:
            embedding: Embedding to look up documents similar to.
            k: The amount of neighbors that will be retrieved.
            filter: Optional. A list of Namespaces for filtering the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                 for more detail.

        Returns:
            A list of k matching documents.
        """
        docs_and_scores = self.similarity_search_by_vector_with_score(
            embedding, k=k, filter=filter, **kwargs
        )

        return [doc for doc, _ in docs_and_scores]

    def _get_index_id(self) -> str:
        """Gets the correct index id for the endpoint.

        Returns:
            The index id if found (which should be found) or throws
            ValueError otherwise.
        """
        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.resource_name:
                return index.id

        raise ValueError(
            f"No index with id {self.index.resource_name} "
            f"deployed on endpoint "
            f"{self.endpoint.display_name}."
        )

    def _download_from_gcs(self, gcs_location: str) -> str:
        """Downloads from GCS in text format.

        Args:
            gcs_location: The location where the file is located.

        Returns:
            The string contents of the file.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        return blob.download_as_string()

    @classmethod
    def from_texts(
        cls: Type["MatchingEngine"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`add_texts`"
        )

    @classmethod
    def from_components(
        cls: Type["MatchingEngine"],
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        index_id: str,
        endpoint_id: str,
        credentials_path: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
            order for the index to be created.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.
            kwargs: Additional keyword arguments to pass to MatchingEngine.__init__().

        Returns:
            A configured MatchingEngine with the texts added to the index.
        """
        gcs_bucket_name = cls._validate_gcs_bucket(gcs_bucket_name)
        credentials = cls._create_credentials_from_file(credentials_path)
        index = cls._create_index_by_id(index_id, project_id, region, credentials)
        endpoint = cls._create_endpoint_by_id(
            endpoint_id, project_id, region, credentials
        )

        gcs_client = cls._get_gcs_client(credentials, project_id)
        cls._init_aiplatform(project_id, region, gcs_bucket_name, credentials)

        return cls(
            project_id=project_id,
            index=index,
            endpoint=endpoint,
            embedding=embedding or cls._get_default_embeddings(),
            gcs_client=gcs_client,
            credentials=credentials,
            gcs_bucket_name=gcs_bucket_name,
            **kwargs,
        )

    @classmethod
    def _validate_gcs_bucket(cls, gcs_bucket_name: str) -> str:
        """Validates the gcs_bucket_name as a bucket name.

        Args:
              gcs_bucket_name: The received bucket uri.

        Returns:
              A valid gcs_bucket_name or throws ValueError if full path is
              provided.
        """
        gcs_bucket_name = gcs_bucket_name.replace("gs://", "")
        if "/" in gcs_bucket_name:
            raise ValueError(
                f"The argument gcs_bucket_name should only be "
                f"the bucket name. Received {gcs_bucket_name}"
            )
        return gcs_bucket_name

    @classmethod
    def _create_credentials_from_file(
        cls, json_credentials_path: Optional[str]
    ) -> Optional[Credentials]:
        """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """

        from google.oauth2 import service_account

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(
                json_credentials_path
            )

        return credentials

    @classmethod
    def _create_index_by_id(
        cls, index_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndex:
        """Creates a MatchingEngineIndex object by id.

        Args:
            index_id: The created index id.
            project_id: The project to retrieve index from.
            region: Location to retrieve index from.
            credentials: GCS credentials.

        Returns:
            A configured MatchingEngineIndex.
        """

        from google.cloud import aiplatform

        logger.debug(f"Creating matching engine index with id {index_id}.")
        return aiplatform.MatchingEngineIndex(
            index_name=index_id,
            project=project_id,
            location=region,
            credentials=credentials,
        )

    @classmethod
    def _create_endpoint_by_id(
        cls, endpoint_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndexEndpoint:
        """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.
            project_id: The project to retrieve index from.
            region: Location to retrieve index from.
            credentials: GCS credentials.

        Returns:
            A configured MatchingEngineIndexEndpoint.
        """

        from google.cloud import aiplatform

        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=project_id,
            location=region,
            credentials=credentials,
        )

    @classmethod
    def _get_gcs_client(
        cls, credentials: "Credentials", project_id: str
    ) -> "storage.Client":
        """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """

        from google.cloud import storage

        return storage.Client(
            credentials=credentials,
            project=project_id,
            client_info=get_client_info(module="vertex-ai-matching-engine"),
        )

    @classmethod
    def _init_aiplatform(
        cls,
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        credentials: "Credentials",
    ) -> None:
        """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: GCS staging location.
            credentials: The GCS Credentials object.
        """

        from google.cloud import aiplatform

        logger.debug(
            f"Initializing AI Platform for project {project_id} on "
            f"{region} and for {gcs_bucket_name}."
        )
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=gcs_bucket_name,
            credentials=credentials,
        )

    @classmethod
    def _get_default_embeddings(cls) -> "TensorflowHubEmbeddings":
        """This function returns the default embedding.

        Returns:
            Default TensorflowHubEmbeddings to use.
        """

        from langchain_community.embeddings import TensorflowHubEmbeddings

        return TensorflowHubEmbeddings()
