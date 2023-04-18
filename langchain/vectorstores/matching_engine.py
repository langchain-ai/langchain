"""Vertex Matching Engine implementation of the vector store."""

import json
import logging
import time
import uuid

from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
from typing import Any, Iterable, List, Optional, Type, Union

from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


class MatchingEngine(VectorStore):
    """Vertex Matching Engine implementation of the vector store.

    While the embeddings are stored in the Matching Engine, the embedded
    documents will be stored in GCS.

    Note that this implementation is mostly meant for reading if you are
    planning to do a real time implementation. While reading is a real time
    operation, updating the index takes close to one hour."""

    def __init__(self,
        project_id: str,
        region: str,
        gcs_bucket_uri: str,
        index_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        endpoint_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        json_credentials_path: Union[str, None] = None,
        embedder: Embeddings = TensorflowHubEmbeddings(model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    ):
        """Vertex Matching Engine implementation of the vector store.

        While the embeddings are stored in the Matching Engine, the embedded
        documents will be stored in GCS.

        Note that this implementation is mostly meant for reading if you are
        planning to do a real time implementation. While reading is a real time
        operation, updating the index takes close to one hour.
            TODO: create docs for this module: https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/vectorstores.rst
            TODO: preconditio for this class the index and the endpoint must exist
            TODO: add aiplatform and storage dependencies to poetry
            Attributes:
                project_id: The GCS project id.
                region: The default location making the API calls. It must have
                the same location as the GCS bucket. TODO(scafati98) is this correct?
                gcs_bucket_uri: The location where the vectors will be stored in
                order for the index to be created.
                index_id: The id of the created index TODO(scafati98) link to notebook to where the index is created and the enpoint is printed.
                endpoint_id: The id of the created endpoint TODO(scafati98) idem arriba.
                json_credentials_path (Optional): Path where the GCS
                credentials are stored on the local file system. If none are
                provided, then the default login will be used.
                embedder: A :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        """
        super().__init__()
        logger.debug(f"Constructor.")
        self.project_id = project_id
        self.region = region
        self.gcs_client = None
        self.gcs_bucket_uri = self._validate_gcs_bucket(gcs_bucket_uri)
        self.embedder = embedder

        self.credentials = self._create_credentials_from_file(
            json_credentials_path)
        self._init_aiplatform(project_id, region, self.gcs_bucket_uri)
        self.index = self._create_index_by_id(index_id)
        self.endpoint = self._create_endpoint_by_id(endpoint_id)

    def _validate_gcs_bucket(self, gcs_bucket_uri: str) -> str:
        """Validates the gcs_bucket_uri as a bucket name.

        Args:
              gcs_bucket_uri: The received bucket uri.

        Returns:
              A valid gcs_bucket_uri or throws ValueError if full path is
              provided.
        """
        gcs_bucket_uri = gcs_bucket_uri.replace("gs://", "")
        if "/" in gcs_bucket_uri:
            raise ValueError(f"The argument gcs_bucket_uri should only be "
                             f"the bucket name. Received {gcs_bucket_uri}")
        return gcs_bucket_uri

    def _create_credentials_from_file(
        self,
        json_credentials_path: Optional[str]
    ) -> Optional[service_account.Credentials]:
        """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(
                json_credentials_path)

        return credentials
    
    def _init_aiplatform(
        self,
        project_id: str,
        region: str,
        gcs_bucket_uri: str
    ) -> None:
        """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket. TODO(scafati98) is this correct?
            gcs_bucket_uri: GCS staging location.
        """
        logger.debug(f"Initializing AI Platform for project {project_id} on "
                     f"{region} and for {gcs_bucket_uri}.")
        aiplatform.init(
            project=project_id, 
            location=region, 
            staging_bucket=gcs_bucket_uri,
            credentials=self.credentials
        )

    def _create_index_by_id(
        self,
        index_id: str
    ) -> "aiplatform.MatchingEngineIndex":
        """Creates a MatchingEngineIndex object by id.

        Args:
            index_id: The created index id.

        Returns:
            A configured MatchingEngineIndex. TODO(scafati98) link notebook
        """
        logger.debug(f"Creating matching engine index with id {index_id}.")
        return aiplatform.MatchingEngineIndex(
            index_name=index_id,
            project=self.project_id,
            location=self.region,
            credentials=self.credentials
        )
    
    def _create_endpoint_by_id(
        self,
        endpoint_id: str
    ) -> "aiplatform.MatchingEngineIndexEndpoint":
        """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.

        Returns:
            A configured MatchingEngineIndexEndpoint. TODO(scafati98) link notebook
        """
        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=self.project_id,
            location=self.region,
            credentials=self.credentials,
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
        logger.debug(f"Embedding documents.")
        embeddings = self.embedder.embed_documents(list(texts))
        jsons = []
        ids = []
        # Could be improved with async.
        for embedding, text in zip(embeddings, texts):
            id = str(uuid.uuid4())
            ids.append(id)
            jsons.append({
                "id": id,
                "embedding": embedding
            })
            self._upload_to_gcs(text, f"documents/{id}")

        logger.debug(f"Uploaded {len(ids)} documents to GCS.")

        # Creating json lines from the embedded documents.
        result_str = "\n".join([json.dumps(x) for x in jsons])

        filename_prefix = f"indexes/{uuid.uuid4()}"
        filename = f"{filename_prefix}/{time.time()}.json"
        self._upload_to_gcs(result_str, filename)
        logger.debug(f"Uploaded updated json with embeddings to "
                     f"{self.gcs_bucket_uri}/{filename}.")

        self.index = self.index.update_embeddings(
            contents_delta_uri=f"gs://{self.gcs_bucket_uri}/{filename_prefix}/"
        )

        logger.debug(f"Updated index with new configuration.")

        return ids
    
    def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
        """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        blob = bucket.blob(gcs_location)
        blob.upload_from_string(data)

    def _get_gcs_client(self) -> storage.Client:
        """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """
        if self.gcs_client is None:            
            self.gcs_client = storage.Client(
                credentials=self.credentials, 
                project=self.project_id
            )

        return self.gcs_client

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.

        Returns:
            A list of k matching documents.
        """

        logger.debug(f"Embedding query {query}.")
        embedding_query = self.embedder.embed_documents([query])

        index_id = None

        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.resource_name:
                index_id = index.id
                break
        
        if index_id == None:
            raise ValueError(f"No index with id {self.index.resource_name} "
                             f"deployed on enpoint "
                             f"{self.endpoint.display_name}.")

        response = self.endpoint.match(
            deployed_index_id=index_id,
            queries=embedding_query,
            num_neighbors=k,
        )

        if len(response) == 0:
            return []

        logger.debug(f"Found {len(response)} matches for the query {query}.")

        results = []

        # I'm only getting the first one because queries receives an array
        # and the similarity_search method only recevies one query. This
        # means that the match method will always return an array with only
        # one element.
        for doc in response[0]:
            print(doc)
            page_content = self._download_from_gcs(f"documents/{doc.id}")
            results.append(Document(page_content=page_content))

        logger.debug(f"Downloaded documents for query.")
            
        return results
    
    def _download_from_gcs(self, gcs_location: str) -> str:
        """Downloads from GCS in text format.

        Args:
            gcs_location: The location where the file is located.

        Returns:
            The string contents of the file.
        """
        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        blob = bucket.blob(gcs_location)
        return blob.download_as_string()
    
    @classmethod
    def from_texts(
        cls: Type["MatchingEngine"], 
        texts: List[str],
        embedding: Embeddings = None,
        metadatas: Optional[List[dict]] = None, 
        project_id: str = None,
        region: str = None,
        gcs_bucket_uri: str = None,
        index_id: str = None,
        endpoint_id: str = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Return VectorStore initialized from texts and embeddings.

        Note that this function shouldn't be run more than once. Otherwise,
        the texts will be added multiple times to the index and GCS.

        Args:
            texts: The texts that will get .
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.
            metadatas: List of metadatas. Defaults to None..
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket. TODO(scafati98) is this correct?.
            gcs_bucket_uri: The location where the vectors will be stored in
            order for the index to be created.
            index_id: The id of the created index TODO(scafati98) link to notebook to where the index is created and the enpoint is printed.
            endpoint_id: The id of the created endpoint TODO(scafati98) idem arriba.

        Returns:
            A configured MatchingEngine with the texts added to the index.
        """
        matching_engine = cls(
            project_id=project_id,
            region=region,
            gcs_bucket_uri=gcs_bucket_uri,
            index_id=index_id,
            endpoint_id=endpoint_id,
            embedder=embedding
        )

        matching_engine.add_texts(texts=texts, metadatas=metadatas)
        return matching_engine
