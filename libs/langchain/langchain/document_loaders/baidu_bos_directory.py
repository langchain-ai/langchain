from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.tencent_cos_file import TencentCOSFileLoader


class BaiduBOSDirectoryLoader(BaseLoader):
    """Load from `Baidu BOS directory`."""

    def __init__(self, conf: Any, bucket: str, prefix: str = ""):
        """Initialize with BOS config, bucket and prefix.
        :param conf(BosConfig): BOS config.
        :param bucket(str): BOS bucket.
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
            from baidubce import exception
            from baidubce.services import bos
            from baidubce.services.bos import canned_acl
            from baidubce.services.bos.bos_client import BosClient
        except ImportError:
            raise ImportError(
                "Could not import bos-python-sdk-v5 python package. "
                "Please install it with `pip install -python-sdk-v5`."
            )
        client = BosClient(self.conf)
        contents = []
        marker = ""
        while True:
            response = client.list_objects(
                bucket_name=self.bucket, prefix=self.prefix, marker=marker, max_keys=1000
            )
            print(f"response={response}")
            contents_len = len(response.contents)
            contents.extend(response.contents)
            if response.is_truncated or contents_len < int(str(response.max_keys)) :
                break
            marker = response.next_marker
        from baidu_bos_file import BaiduBOSFileLoader   
        for content in contents:
            if str(content.key).endswith("/"):
                continue
            loader = BaiduBOSFileLoader(self.conf, self.bucket, str(content.key))
            yield loader.load()[0]
