import os
import tempfile
from typing import Any, Iterator

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class TencentCOSFileLoader(BaseLoader):
    """Load from `Tencent Cloud COS` file."""

    def __init__(self, conf: Any, bucket: str, key: str):
        """Initialize with COS config, bucket and key name.
        :param conf(CosConfig): COS config.
        :param bucket(str): COS bucket.
        :param key(str): COS file key.
        """
        self.conf = conf
        self.bucket = bucket
        self.key = key

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        try:
            from qcloud_cos import CosS3Client
        except ImportError:
            raise ImportError(
                "Could not import cos-python-sdk-v5 python package. "
                "Please install it with `pip install cos-python-sdk-v5`."
            )

        # initialize a client
        client = CosS3Client(self.conf)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.bucket}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Download the file to a destination
            client.download_file(
                Bucket=self.bucket, Key=self.key, DestFilePath=file_path
            )
            loader = UnstructuredFileLoader(file_path)
            # UnstructuredFileLoader not implement lazy_load yet
            return iter(loader.load())
