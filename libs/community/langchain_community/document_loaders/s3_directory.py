from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.s3_file import S3FileLoader

if TYPE_CHECKING:
    import botocore


class S3DirectoryLoader(BaseLoader):
    """Load from `Amazon AWS S3` directory."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
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
    ):
        """Initialize with bucket and key name.

        :param bucket: The name of the S3 bucket.
        :param prefix: The prefix of the S3 key. Defaults to "".

        :param region_name: The name of the region associated with the client.
            A client is associated with a single region.

        :param api_version: The API version to use.  By default, botocore will
            use the latest API version when creating a client.  You only need
            to specify this parameter if you want to use a previous API version
            of the client.

        :param use_ssl: Whether to use SSL.  By default, SSL is used.
            Note that not all services support non-ssl connections.

        :param verify: Whether to verify SSL certificates.
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
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region_name = region_name
        self.api_version = api_version
        self.use_ssl = use_ssl
        self.verify = verify
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.boto_config = boto_config

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.resource(
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
        bucket = s3.Bucket(self.bucket)
        docs = []
        for obj in bucket.objects.filter(Prefix=self.prefix):
            loader = S3FileLoader(
                self.bucket,
                obj.key,
                region_name=self.region_name,
                api_version=self.api_version,
                use_ssl=self.use_ssl,
                verify=self.verify,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                boto_config=self.boto_config,
            )
            docs.extend(loader.load())
        return docs
