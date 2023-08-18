# coding:utf-8
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.obs_file import OBSFileLoader


class OBSDirectoryLoader(BaseLoader):
    """Load from `Huawei OBS directory`."""

    def __init__(
        self,
        bucket: str,
        endpoint: str,
        config: Optional[dict] = None,
        prefix: str = "",
    ):
        """Initialize the OBSDirectoryLoader with the specified settings.

        Args:
            bucket (str): The name of the OBS bucket to be used.
            endpoint (str): The endpoint URL of your OBS bucket.
            config (dict): The parameters for connecting to OBS, provided as a dictionary. The dictionary could have the following keys:
                - "ak" (str, optional): Your OBS access key (required if `get_token_from_ecs` is False and bucket policy is not public read).
                - "sk" (str, optional): Your OBS secret key (required if `get_token_from_ecs` is False and bucket policy is not public read).
                - "token" (str, optional): Your security token (required if using temporary credentials).
                - "get_token_from_ecs" (bool, optional): Whether to retrieve the security token from ECS. Defaults to False if not provided. If set to True, `ak`, `sk`, and `token` will be ignored.
            prefix (str, optional): The prefix to be added to the OBS key. Defaults to "".

        Note:
            Before using this class, make sure you have registered with OBS and have the necessary credentials. The `ak`, `sk`, and `endpoint` values are mandatory unless `get_token_from_ecs` is True or the bucket policy is public read. `token` is required when using temporary credentials.
        Example:
            To create a new OBSDirectoryLoader:
            ```
            config = {
                "ak": "your-access-key",
                "sk": "your-secret-key"
            }
            ```
            directory_loader = OBSDirectoryLoader("your-bucket-name", "your-end-endpoint", config, "your-prefix")
        """  # noqa: E501
        try:
            from obs import ObsClient
        except ImportError:
            raise ImportError(
                "Could not import esdk-obs-python python package. "
                "Please install it with `pip install esdk-obs-python`."
            )
        if not config:
            config = dict()
        if config.get("get_token_from_ecs"):
            self.client = ObsClient(server=endpoint, security_provider_policy="ECS")
        else:
            self.client = ObsClient(
                access_key_id=config.get("ak"),
                secret_access_key=config.get("sk"),
                security_token=config.get("token"),
                server=endpoint,
            )

        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        """Load documents."""
        max_num = 1000
        mark = None
        docs = []
        while True:
            resp = self.client.listObjects(
                self.bucket, prefix=self.prefix, marker=mark, max_keys=max_num
            )
            if resp.status < 300:
                for content in resp.body.contents:
                    loader = OBSFileLoader(self.bucket, content.key, client=self.client)
                    docs.extend(loader.load())
                if resp.body.is_truncated is True:
                    mark = resp.body.next_marker
                else:
                    break
        return docs
