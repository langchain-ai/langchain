# coding:utf-8

import os
import tempfile
from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class AlibabaOSSFileLoader(BaseLoader):
    """Load from the `Alibaba Cloud OSS file`."""

    def __init__(
        self,
        bucket: str,
        key: str,
        endpoint: str,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        auth: Optional[Any] = None,
    ) -> None:
        """Initialize the OSSFileLoader with the specified settings.

        Args:
            bucket (str): The name of the OSS bucket to be used.
            key (str): The name of the object in the OSS bucket.
            endpoint_url (str): The endpoint URL of your OSS bucket.
            access_key_id (str, optional): The access key ID for authentication. Defaults to None.
            access_key_secret (str, optional): The access key secret for authentication. Defaults to None.
            auth (oss2.auth.Auth or oss2.auth.ProviderAuth, optional): An instance of the oss2.auth class.

        Raises:
            ImportError: If the `oss2` package is not installed.
            TypeError: If the provided `auth` is not an instance of oss2.auth.Auth or oss2.auth.ProviderAuth.
        Note:
            Before using this class, make sure you have registered with OSS and have the necessary credentials.
            If none of the above authentication methods is provided, the loader will attempt to access oss file anonymously.

        Example:
            To create a new OSSFileLoader with explicit access key and secret:
            ```
            oss_loader = OSSFileLoader(
                "your-bucket-name",
                "your-object-key",
                "your-endpoint-url",
                "your-access-key",
                "you-access-key-secret"
            )
            ```

            To create a new OSSFileLoader with an existing auth from environment variables:
            ```
            from oss2.credentials import EnvironmentVariableCredentialsProvider
            auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

            oss_loader = OSSFileLoader("
                "your-bucket-name",
                "your-object-key",
                "your-endpoint-url",
                auth=auth
            )
            ```
        """  # noqa: E501
        try:
            import oss2
        except ImportError:
            raise ImportError(
                "Could not import oss2 python package. "
                "Please install it with `pip install oss2`."
            )

        if access_key_id and access_key_secret:
            self.auth = oss2.Auth(access_key_id, access_key_secret)
        elif auth and isinstance(auth, (oss2.Auth, oss2.ProviderAuth)):
            self.auth = auth
        else:
            self.auth = oss2.AnonymousAuth()

        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket)
        self.key = key

    def load(self) -> List[Document]:
        """Load documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.bucket}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Download the file to a destination
            self.bucket.get_object_to_file(self.key, file_path)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
