from typing import Any, Iterator

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


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

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        try:
            from baidubce.services.bos.bos_client import BosClient
        except ImportError:
            raise ImportError(
                "Please install bce-python-sdk with `pip install bce-python-sdk`."
            )
        client = BosClient(self.conf)
        contents = []
        marker = ""
        while True:
            response = client.list_objects(
                bucket_name=self.bucket,
                prefix=self.prefix,
                marker=marker,
                max_keys=1000,
            )
            contents_len = len(response.contents)
            contents.extend(response.contents)
            if response.is_truncated or contents_len < int(str(response.max_keys)):
                break
            marker = response.next_marker
        from langchain_community.document_loaders.baiducloud_bos_file import (
            BaiduBOSFileLoader,
        )

        for content in contents:
            if str(content.key).endswith("/"):
                continue
            loader = BaiduBOSFileLoader(self.conf, self.bucket, str(content.key))
            yield loader.load()[0]
