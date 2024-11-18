from __future__ import annotations

import base64
import hashlib
import logging
from functools import cached_property
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.stores import ByteStore

try:
    from opensearchpy import OpenSearch
    from opensearchpy.helpers import bulk
except ImportError:
    raise ImportError("Could not import OpenSearch. Please install it with `pip install opensearch-py`.")

logger = logging.getLogger(__name__)


class LindormSearchByteStore(ByteStore):
    """An lindorm search byte store."""

    def __init__(self,
                 lindorm_search_url: str,
                 index_name: str,
                 **kwargs: Any
                 ):

        """
        Initialize the Lindorm Search ByteStore by specifying the index to use and
        determining whether the input key should be stored in it.

        Args:
            lindorm_search_url (str): The endpoint of lindorm search engine with format
                like "{instanceid}-proxy-search-vpc.lindorm.aliyuncs.com:30070".
            index_name (str): The name of the index.
                If they do not exist an index is created,
                according to the default mapping defined by the `mapping` property.
            kwargs: Parameter to initialize an OpenSearch client

        """
        self._index_name = index_name
        self.client = OpenSearch(lindorm_search_url, **kwargs)
        if not self.client.indices.exists(index=index_name):
            logger.info(f"Creating new index: {index_name}")
            self.client.indices.create(index=index_name, body=self.mapping)

    @cached_property
    def mapping(self) -> Dict[str, Any]:
        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 4
                }
            },
            "mappings": {
                "properties": {
                    "key": {
                        "type": "text"
                    },
                    "value": {
                        "type": "text"
                    }
                }
            }
        }
        return mapping

    @staticmethod
    def transform_bytes_to_str(data: bytes) -> str:
        """Encode the data as bytes to as a base64 string."""
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def transform_str_to_bytes(data: str) -> bytes:
        """Decode the base64 string to data as bytes."""
        return base64.b64decode(data)

    def _key(self, input_text: str) -> str:
        """Generate a key for the store."""
        return hashlib.md5(input_text.encode()).hexdigest()

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        if not any(keys):
            return []

        cache_keys = [self._key(k) for k in keys]

        records = self.client.mget(
            body={
                "docs": [{"_id": doc_id} for doc_id in cache_keys]
            },
            index=self._index_name
        )

        return [
            self.transform_str_to_bytes(r["_source"]["value"]) if r["found"] else None
            for r in records["docs"]
        ]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, bytes]]): A sequence of key-value pairs.
        """
        if not self.client.indices.exists(index=self._index_name):
            logger.info(f"Creating new index: {self._index_name}")
            self.client.indices.create(index=self._index_name, body=self.mapping)

        requests = []
        for key, value in key_value_pairs:
            request = {
                "_op_type": "index",
                "_index": self._index_name,
                "_id": self._key(key),
                "key": key,
                "value": self.transform_bytes_to_str(value)
            }
            requests.append(request)

        bulk(self.client, requests)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        requests = []
        for key in keys:
            request = {
                "_op_type": "delete",
                "_index": self._index_name,
                "_id": self._key(key)
            }
            requests.append(request)

        bulk(self.client, requests)
        self.client.indices.refresh(index=self._index_name)

    def yield_keys(self, prefix: Optional[str] = None, scroll='5m', size=100) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Returns:
            Iterator[dict]: An iterator over keys that match the given prefix.

            This method is allowed to return an iterator over dict which stores

            all the property as mapping describes.
        """

        """Get an iterator over keys that match the given prefix."""

        if prefix == None or prefix == "":
            prefix = "" if prefix == None else prefix
            logger.warning("It might be expensive to yield keys when prefix is None or \"\"")

        query = {
            "query": {
                "prefix": {
                    "key": prefix
                }
            }
        }

        results = self.client.search(index=self._index_name,
                                     body=query,
                                     scroll=scroll,
                                     size=size)

        # yield results from the initial search
        for hit in results["hits"]["hits"]:
            yield hit["_source"]["key"]

        scroll_id = results["_scroll_id"]

        # save the scroll_id for scrolling
        while True:
            # scroll through the results
            results = self.client.scroll(scroll_id=scroll_id, scroll=scroll)

            # exit when no more results
            if len(results['hits']['hits']) == 0:
                break

            # yield results from the scroll request.
            for hit in results['hits']['hits']:
                yield hit['_source']["key"]

            # update the scroll_id for the next iteration.
            scroll_id = results['_scroll_id']

    def delete_index(self, index_name: Optional[str] = None) -> Optional[bool]:
        """Deletes a given index from vectorstore."""
        if index_name is None:
            if self._index_name is None:
                raise ValueError("index_name must be provided.")
            index_name = self._index_name
        try:
            self.client.indices.delete(index=index_name)
            return True
        except Exception as e:
            raise e
