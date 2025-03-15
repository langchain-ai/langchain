from typing import Iterator, List, Optional, Sequence, Tuple
from langchain_core.stores import BaseStore
from langchain_core.documents import Document


class OpenSearchStore(BaseStore[str, str]):
    """BaseStore implementation using OpenSearch as the underlying store"""

    def __init__(
        self,
        host: str,
        index_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> None:
        """Initialize the OpenSearch client and set up the index.

        Args:
            host (str): The OpenSearch host URL.
            index_name (str): The OpenSearch index name where data will be stored.
            username (Optional[str]): Optional username for authentication.
            password (Optional[str]): Optional password for authentication.
        """

        try:
            from opensearchpy import OpenSearch
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import the opensearchpy package. "
                "Please install it with `pip install opensearch-py`."
            )

        self.client = OpenSearch(
            [host],
            http_auth=(username, password) if username and password else None,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        self.index_name = index_name

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name)

    def _get_document(self, key: str) -> dict:
        """Helper function to retrieve the document from OpenSearch by key."""
        try:
            response = self.client.get(index=self.index_name, id=key)
            return response['_source']
        except Exception as e:
            return None

    def mget(self, keys: Sequence[str]) -> List[Optional[str]]:
        """Get the values associated with the given keys from OpenSearch.

        Args:
            keys (Sequence[str]): A sequence of keys to retrieve.

        Returns:
            A list of values (str) associated with the keys, or None if not found.
        """
        values: List[Optional[str]] = []
        for key in keys:
            doc = self._get_document(key)
            if doc:
                values.append(
                    Document(
                        page_content=doc["page_content"],
                        metadata=doc["metadata"]
                    )
                )
            else:
                values.append(None)
        return values

    def mset(self, key_value_pairs: Sequence[Tuple[str, str]]) -> None:
        """Set the values for the given keys in OpenSearch.

        Args:
            key_value_pairs (Sequence[Tuple[str, str]]):
                Key-value pairs to store in OpenSearch.
        """
        for key, value in key_value_pairs:
            document = value.dict()
            self.client.index(index=self.index_name, id=key, body=document)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys from OpenSearch.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        for key in keys:
            self.client.delete(index=self.index_name, id=key)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix from OpenSearch.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """

        query = {
            "query": {
                "prefix": {
                    "_id": prefix if prefix else ""
                }
            }
        }
        response = self.client.search(index=self.index_name, body=query)

        for hit in response['hits']['hits']:
            yield hit['_id']