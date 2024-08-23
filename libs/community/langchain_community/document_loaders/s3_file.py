from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader

if TYPE_CHECKING:
    import botocore


class S3FileLoader(UnstructuredBaseLoader):
    """Load from `Amazon AWS S3` file."""

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        region_name: Optional[str] = None,
        api_version: Optional[str] = None,
        use_ssl: Optional[bool] = True,
        verify: Union[str, bool, None] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        boto_config: Optional[botocore.client.Config] = None,
        mode: str = "single",
        post_processors: Optional[List[Callable]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with bucket and key name.

        :param bucket: The name of the S3 bucket.
        :param key: The key of the S3 object.

        :param region_name: The name of the region associated with the client.
            A client is associated with a single region.

        :param api_version: The API version to use.  By default, botocore will
            use the latest API version when creating a client.  You only need
            to specify this parameter if you want to use a previous API version
            of the client.

        :param use_ssl: Whether or not to use SSL.  By default, SSL is used.
            Note that not all services support non-ssl connections.

        :param verify: Whether or not to verify SSL certificates.
            By default SSL certificates are verified.  You can provide the
            following values:

            * False - do not validate SSL certificates.  SSL will still be
              used (unless use_ssl is False), but SSL certificates
              will not be verified.
            * path/to/cert/bundle.pem - A filename of the CA cert bundle to
              uses.  You can specify this argument if you want to use a
              different CA cert bundle than the one used by botocore.

        :param endpoint_url: The complete URL to use for the constructed
            client.  Normally, botocore will automatically construct the
            appropriate URL to use when communicating with a service.  You can
            specify a complete URL (including the "http/https" scheme) to
            override this behavior.  If this value is provided, then
            ``use_ssl`` is ignored.

        :param aws_access_key_id: The access key to use when creating
            the client.  This is entirely optional, and if not provided,
            the credentials configured for the session will automatically
            be used.  You only need to provide this argument if you want
            to override the credentials used for this specific client.

        :param aws_secret_access_key: The secret key to use when creating
            the client.  Same semantics as aws_access_key_id above.

        :param aws_session_token: The session token to use when creating
            the client.  Same semantics as aws_access_key_id above.

        :type boto_config: botocore.client.Config
        :param boto_config: Advanced boto3 client configuration options. If a value
            is specified in the client config, its value will take precedence
            over environment variables and configuration values, but not over
            a value passed explicitly to the method. If a default config
            object is set on the session, the config object used when creating
            the client will be the result of calling ``merge()`` on the
            default config with the config provided to this call.
        :param mode: Mode in which to read the file. Valid options are: single,
            paged and elements.
        :param post_processors: Post processing functions to be applied to
            extracted elements.
        :param **unstructured_kwargs: Arbitrary additional kwargs to pass in when
            calling `partition`
        """
        super().__init__(mode, post_processors, **unstructured_kwargs)
        self.bucket = bucket
        self.key = key
        self.region_name = region_name
        self.api_version = api_version
        self.use_ssl = use_ssl
        self.verify = verify
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.boto_config = boto_config

    def _get_elements(self) -> List:
        """Get elements."""
        from unstructured.partition.auto import partition

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config=self.boto_config,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(self.bucket, self.key, file_path)
            return partition(filename=file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": f"s3://{self.bucket}/{self.key}"}
