# coding:utf-8

import os
import tempfile
from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class OBSFileLoader(BaseLoader):
    """Load from the `Huawei OBS file`."""

    def __init__(
        self,
        bucket: str,
        key: str,
        client: Any = None,
        endpoint: str = "",
        config: Optional[dict] = None,
    ) -> None:
        """Initialize the OBSFileLoader with the specified settings.

        Args:
            bucket (str): The name of the OBS bucket to be used.
            key (str): The name of the object in the OBS bucket.
            client (ObsClient, optional): An instance of the ObsClient to connect to OBS.
            endpoint (str, optional): The endpoint URL of your OBS bucket. This parameter is mandatory if `client` is not provided.
            config (dict, optional): The parameters for connecting to OBS, provided as a dictionary. This parameter is ignored if `client` is provided. The dictionary could have the following keys:
                - "ak" (str, optional): Your OBS access key (required if `get_token_from_ecs` is False and bucket policy is not public read).
                - "sk" (str, optional): Your OBS secret key (required if `get_token_from_ecs` is False and bucket policy is not public read).
                - "token" (str, optional): Your security token (required if using temporary credentials).
                - "get_token_from_ecs" (bool, optional): Whether to retrieve the security token from ECS. Defaults to False if not provided. If set to True, `ak`, `sk`, and `token` will be ignored.

        Raises:
            ValueError: If the `esdk-obs-python` package is not installed.
            TypeError: If the provided `client` is not an instance of ObsClient.
            ValueError: If `client` is not provided, but `endpoint` is missing.

        Note:
            Before using this class, make sure you have registered with OBS and have the necessary credentials. The `ak`, `sk`, and `endpoint` values are mandatory unless `get_token_from_ecs` is True or the bucket policy is public read. `token` is required when using temporary credentials.

        Example:
            To create a new OBSFileLoader with a new client:
            ```
            config = {
                "ak": "your-access-key",
                "sk": "your-secret-key"
            }
            obs_loader = OBSFileLoader("your-bucket-name", "your-object-key", config=config)
            ```

            To create a new OBSFileLoader with an existing client:
            ```
            from obs import ObsClient

            # Assuming you have an existing ObsClient object 'obs_client'
            obs_loader = OBSFileLoader("your-bucket-name", "your-object-key", client=obs_client)
            ```

            To create a new OBSFileLoader without an existing client:
            ```
            obs_loader = OBSFileLoader("your-bucket-name", "your-object-key", endpoint="your-endpoint-url")
            ```
        """  # noqa: E501
        try:
            from obs import ObsClient
        except ImportError:
            raise ImportError(
                "Could not import esdk-obs-python python package. "
                "Please install it with `pip install esdk-obs-python`."
            )
        if not client:
            if not endpoint:
                raise ValueError("Either OBSClient or endpoint must be provided.")
            if not config:
                config = dict()
            if config.get("get_token_from_ecs"):
                client = ObsClient(server=endpoint, security_provider_policy="ECS")
            else:
                client = ObsClient(
                    access_key_id=config.get("ak"),
                    secret_access_key=config.get("sk"),
                    security_token=config.get("token"),
                    server=endpoint,
                )
        if not isinstance(client, ObsClient):
            raise TypeError("Client must be ObsClient type")
        self.client = client
        self.bucket = bucket
        self.key = key

    def load(self) -> List[Document]:
        """Load documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.bucket}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Download the file to a destination
            self.client.downloadFile(
                bucketName=self.bucket, objectKey=self.key, downloadFile=file_path
            )
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
