from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.tencent_cos_file import TencentCOSFileLoader


class TencentCOSDirectoryLoader(BaseLoader):
    """Load from `Tencent Cloud COS` directory."""

    def __init__(self, conf: Any, bucket: str, prefix: str = ""):
        """Initialize with COS config, bucket and prefix.
        :param conf(CosConfig): COS config.
        :param bucket(str): COS bucket.
        :param prefix(str): prefix.
        """
        self.conf = conf
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        try:
            from qcloud_cos import CosS3Client
        except ImportError:
            raise ImportError(
                "Could not import cos-python-sdk-v5 python package. "
                "Please install it with `pip install cos-python-sdk-v5`."
            )
        client = CosS3Client(self.conf)
        contents = []
        marker = ""
        while True:
            response = client.list_objects(
                Bucket=self.bucket, Prefix=self.prefix, Marker=marker, MaxKeys=1000
            )
            if "Contents" in response:
                contents.extend(response["Contents"])
            if response["IsTruncated"] == "false":
                break
            marker = response["NextMarker"]
        for content in contents:
            if content["Key"].endswith("/"):
                continue
            loader = TencentCOSFileLoader(self.conf, self.bucket, content["Key"])
            yield loader.load()[0]
