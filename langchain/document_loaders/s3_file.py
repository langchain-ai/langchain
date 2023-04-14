"""Loading logic for loading documents from an s3 file."""
from io import BytesIO
from typing import List, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.schema import Blob
from langchain.document_loaders.unstructured import UnstructuredFileIOLoader


def _get_bytes(s3, bucket, key) -> BytesIO:
    raise NotImplementedError()


class S3FileLoader(BaseLoader):
    """Loading logic for loading documents from s3."""

    def __init__(self, bucket: str, key: str) -> None:
        """Initialize with bucket and key name."""
        self.bucket = bucket
        self.key = key

    def load(self) -> List[Document]:
        """Load documents."""
        docs = []
        for blob in self.lazy_load():
            docs.extend(UnstructuredFileIOLoader(blob.data).load())
        return docs

    def lazy_load(
        self,
    ) -> Union[Generator[Blob, None, None], Generator[Document, None, None]]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ValueError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.client("s3")
        yield Blob(
            data=_get_bytes(s3, self.bucket, self.key),
        )
