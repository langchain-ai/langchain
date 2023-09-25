import os
import tempfile
from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class BaiduBOSFileLoader(BaseLoader):
    """Load from `Baidu Cloud BOS` file."""

    def __init__(self, conf: Any, bucket: str, key: str):
        """Initialize with COS config, bucket and key name.
        :param conf(BceClientConfiguration): BOS config.
        :param bucket(str): BOS bucket.
        :param key(str): BOS file key.
        """
        self.conf = conf
        self.bucket = bucket
        self.key = key

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        try:
            from baidubce import exception
            from baidubce.services import bos
            from baidubce.services.bos import canned_acl
            from baidubce.services.bos.bos_client import BosClient
        except ImportError:
            raise ImportError(
                "Could not import cos-python-sdk-v5 python package. "
                "Please install it with `pip install cos-python-sdk-v5`."
            )

        # Initialise a client
        client = BosClient(self.conf)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.bucket}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Download the file to a destination
            print(f"get object key {self.key} to file {file_path}")
            client.get_object_to_file(
                self.bucket, self.key, file_path
            )
            try:
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
                return iter(documents)
            except Exception as ex:
                print(f"load doucment error = {ex}")
                return iter([Document(page_content="")])
