from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.s3_file import S3FileLoader


class S3DirectoryLoader(BaseLoader):
    """Load from `Amazon AWS S3` directory."""

    def __init__(self, bucket: str, prefix: str = ""):
        """Initialize with bucket and key name.

        Args:
            bucket: The name of the S3 bucket.
            prefix: The prefix of the S3 key. Defaults to "".
        """
        self.bucket = bucket
        self.prefix = prefix

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
            loader = S3FileLoader(self.bucket, obj.key)
            docs.extend(loader.load())
        return docs
