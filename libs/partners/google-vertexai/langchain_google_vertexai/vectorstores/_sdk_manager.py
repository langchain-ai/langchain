from typing import TYPE_CHECKING, Any

from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.oauth2.service_account import Credentials

if TYPE_CHECKING:
    from google.cloud import datastore


class VectorSearchSDKManager:
    """Class in charge of building all Google Cloud SDK Objects needed to build
    VectorStores from project_id, credentials or other specifications. Abstracts 
    away the authentication layer.
    """

    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        credentials: Credentials | None = None,
        credentials_path: str | None = None,
    ) -> None:
        """Constructor.

        If `credentials` is provided, those credentials are used. If not provided
        `credentials_path` is used to retrieve credentials from a file. If also not
        provided, falls back to default credentials.

        Args:
            project_id: Id of the project.
            region: Region of the project. E.j. 'us-central1'
            credentials: Google cloud Credentials object.
            credentials_path: Google Cloud Credentials json file path.
        """
        self._project_id = project_id
        self._region = region

        if credentials is not None:
            self._credentials = credentials
        elif credentials_path is not None:
            self._credentials = Credentials.from_service_account_file(credentials_path)
        else:
            self._credentials = None

        self.initialize_aiplatform()

    def initialize_aiplatform(self) -> None:
        """Initializes aiplatform."""
        aiplatform.init(
            project=self._project_id,
            location=self._region,
            credentials=self._credentials,
        )

    def get_gcs_client(self) -> storage.Client:
        """Retrieves a Google Cloud Storage client.

        Returns:
            Google Cloud Storage Agent.
        """
        return storage.Client(project=self._project_id, credentials=self._credentials)

    def get_gcs_bucket(self, bucket_name: str) -> storage.Bucket:
        """Retrieves a Google Cloud Bucket by bucket name.

        Args:
            bucket_name: Name of the bucket to be retrieved.

        Returns:
            Google Cloud Bucket.
        """
        client = self.get_gcs_client()
        return client.get_bucket(bucket_name)

    def get_index(self, index_id: str) -> MatchingEngineIndex:
        """Retrieves a MatchingEngineIndex (VectorSearchIndex) by id.

        Args:
            index_id: Id of the index to be retrieved.

        Returns:
            MatchingEngineIndex instance.
        """
        return MatchingEngineIndex(
            index_name=index_id,
            project=self._project_id,
            location=self._region,
            credentials=self._credentials,
        )

    def get_endpoint(self, endpoint_id: str) -> MatchingEngineIndexEndpoint:
        """Retrieves a MatchingEngineIndexEndpoint (VectorSearchIndexEndpoint) by id.

        Args:
            endpoint_id: Id of the endpoint to be retrieved.

        Returns:
            MatchingEngineIndexEndpoint instance.
        """
        return MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=self._project_id,
            location=self._region,
            credentials=self._credentials,
        )

    def get_datastore_client(self, **kwargs: Any) -> "datastore.Client":
        """ Get's a datastore Client.
        
        Args:
            **kwargs: Keyword arguments to pass to datatastore.Client constructor.
        
        Returns:
            datastore Client.
        """
        from google.cloud import datastore

        ds_client = datastore.Client(
            project=self._project_id, credentials=self._credentials, **kwargs
        )

        return ds_client
