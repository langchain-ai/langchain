"""Module providing Infinispan as a VectorStore"""

from __future__ import annotations

import json
import logging
import uuid
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class InfinispanVS(VectorStore):
    """`Infinispan` VectorStore interface.

    This class exposes the method to present Infinispan as a
    VectorStore. It relies on the Infinispan class (below) which takes care
    of the REST interface with the server.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import InfinispanVS
            from mymodels import RGBEmbeddings

            vectorDb = InfinispanVS.from_documents(docs,
                            embedding=RGBEmbeddings(),
                            output_fields=["texture", "color"],
                            lambda_key=lambda text,meta: str(meta["_key"]),
                            lambda_content=lambda item: item["color"])

    """

    def __init__(
        self,
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        clear_old: Optional[bool] = True,
        **kwargs: Any,
    ):
        self.ispn = Infinispan(**kwargs)
        self._configuration = kwargs
        self._cache_name = str(self._configuration.get("cache_name", "vector"))
        self._entity_name = str(self._configuration.get("entity_name", "vector"))
        self._embedding = embedding
        self._textfield = self._configuration.get("textfield", "text")
        self._vectorfield = self._configuration.get("vectorfield", "vector")
        self._to_content = self._configuration.get(
            "lambda_content", lambda item: self._default_content(item)
        )
        self._to_metadata = self._configuration.get(
            "lambda_metadata", lambda item: self._default_metadata(item)
        )
        self._output_fields = self._configuration.get("output_fields")
        self._ids = ids
        if clear_old:
            self.ispn.cache_clear(self._cache_name)

    def _default_metadata(self, item: dict) -> dict:
        meta = dict(item)
        meta.pop(self._vectorfield, None)
        meta.pop(self._textfield, None)
        meta.pop("_type", None)
        return meta

    def _default_content(self, item: dict[str, Any]) -> Any:
        return item.get(self._textfield)

    def schema_create(self, proto: str) -> requests.Response:
        """Deploy the schema for the vector db
        Args:
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.schema_post(self._entity_name + ".proto", proto)

    def schema_delete(self) -> requests.Response:
        """Delete the schema for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.schema_delete(self._entity_name + ".proto")

    def cache_create(self, config: str = "") -> requests.Response:
        """Create the cache for the vector db
        Args:
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        if config == "":
            config = (
                '''
            {
  "distributed-cache": {
    "owners": "2",
    "mode": "SYNC",
    "statistics": true,
    "encoding": {
      "media-type": "application/x-protostream"
    },
    "indexing": {
      "enabled": true,
      "storage": "filesystem",
      "startup-mode": "AUTO",
      "indexing-mode": "AUTO",
      "indexed-entities": [
        "'''
                + self._entity_name
                + """"
      ]
    }
  }
}
"""
            )
        return self.ispn.cache_post(self._cache_name, config)

    def cache_delete(self) -> requests.Response:
        """Delete the cache for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.cache_delete(self._cache_name)

    def cache_clear(self) -> requests.Response:
        """Clear the cache for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.cache_clear(self._cache_name)

    def cache_index_clear(self) -> requests.Response:
        """Clear the index for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.index_clear(self._cache_name)

    def cache_index_reindex(self) -> requests.Response:
        """Rebuild the for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.index_reindex(self._cache_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        result = []
        embeds = self._embedding.embed_documents(list(texts))  # type: ignore
        if not metadatas:
            metadatas = [{} for _ in texts]
        ids = self._ids or [str(uuid.uuid4()) for _ in texts]
        data_input = list(zip(metadatas, embeds, ids))
        for metadata, embed, key in data_input:
            data = {"_type": self._entity_name, self._vectorfield: embed}
            data.update(metadata)
            data_str = json.dumps(data)
            self.ispn.put(key, data_str, self._cache_name)
            result.append(key)
        return result

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        documents = self.similarity_search_with_score(query=query, k=k)
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]
        """
        embed = self._embedding.embed_query(query)  # type: ignore
        documents = self.similarity_search_with_score_by_vector(embedding=embed, k=k)
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in res]

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of pair (Documents, score) most similar to the query vector.
        """
        if self._output_fields is None:
            query_str = (
                "select v, score(v) from "
                + self._entity_name
                + " v where v."
                + self._vectorfield
                + " <-> "
                + json.dumps(embedding)
                + "~"
                + str(k)
            )
        else:
            query_proj = "select "
            for field in self._output_fields[:-1]:
                query_proj = query_proj + "v." + field + ","
            query_proj = query_proj + "v." + self._output_fields[-1]
            query_str = (
                query_proj
                + ", score(v) from "
                + self._entity_name
                + " v where v."
                + self._vectorfield
                + " <-> "
                + json.dumps(embedding)
                + "~"
                + str(k)
            )
        query_res = self.ispn.req_query(query_str, self._cache_name)
        result = json.loads(query_res.text)
        return self._query_result_to_docs(result)

    def _query_result_to_docs(
        self, result: dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        documents = []
        for row in result["hits"]:
            hit = row["hit"] or {}
            if self._output_fields is None:
                entity = hit["*"]
            else:
                entity = {key: hit.get(key) for key in self._output_fields}
            doc = Document(
                page_content=self._to_content(entity),
                metadata=self._to_metadata(entity),
            )
            documents.append((doc, hit["score()"]))
        return documents

    @classmethod
    def from_texts(
        cls: Type[InfinispanVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        clear_old: Optional[bool] = None,
        **kwargs: Any,
    ) -> InfinispanVS:
        """Return VectorStore initialized from texts and embeddings."""
        infinispanvs = cls(embedding=embedding, ids=ids, clear_old=clear_old, **kwargs)
        if texts:
            infinispanvs.add_texts(texts, metadatas)
        return infinispanvs


REST_TIMEOUT = 10


class Infinispan:
    """Helper class for `Infinispan` REST interface.

    This class exposes the Infinispan operations needed to
    create and set up a vector db.

    You need a running Infinispan (15+) server without authentication.
    You can easily start one, see: https://github.com/rigazilla/infinispan-vector#run-infinispan
    """

    def __init__(self, **kwargs: Any):
        self._configuration = kwargs
        self._schema = str(self._configuration.get("schema", "http"))
        self._host = str(self._configuration.get("hosts", ["127.0.0.1:11222"])[0])
        self._default_node = self._schema + "://" + self._host
        self._cache_url = str(self._configuration.get("cache_url", "/rest/v2/caches"))
        self._schema_url = str(self._configuration.get("cache_url", "/rest/v2/schemas"))
        self._use_post_for_query = str(
            self._configuration.get("use_post_for_query", True)
        )

    def req_query(
        self, query: str, cache_name: str, local: bool = False
    ) -> requests.Response:
        """Request a query
        Args:
            query(str): query requested
            cache_name(str): name of the target cache
            local(boolean): whether the query is local to clustered
        Returns:
            An http Response containing the result set or errors
        """
        if self._use_post_for_query:
            return self._query_post(query, cache_name, local)
        return self._query_get(query, cache_name, local)

    def _query_post(
        self, query_str: str, cache_name: str, local: bool = False
    ) -> requests.Response:
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "?action=search&local="
            + str(local)
        )
        data = {"query": query_str}
        data_json = json.dumps(data)
        response = requests.post(
            api_url,
            data_json,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def _query_get(
        self, query_str: str, cache_name: str, local: bool = False
    ) -> requests.Response:
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "?action=search&query="
            + query_str
            + "&local="
            + str(local)
        )
        response = requests.get(api_url, timeout=REST_TIMEOUT)
        return response

    def post(self, key: str, data: str, cache_name: str) -> requests.Response:
        """Post an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.post(
            api_url,
            data,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def put(self, key: str, data: str, cache_name: str) -> requests.Response:
        """Put an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.put(
            api_url,
            data,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def get(self, key: str, cache_name: str) -> requests.Response:
        """Get an entry
        Args:
            key(str): key of the entry
            cache_name(str): target cache
        Returns:
            An http Response containing the entry or errors
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.get(
            api_url, headers={"Content-Type": "application/json"}, timeout=REST_TIMEOUT
        )
        return response

    def schema_post(self, name: str, proto: str) -> requests.Response:
        """Deploy a schema
        Args:
            name(str): name of the schema. Will be used as a key
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.post(api_url, proto, timeout=REST_TIMEOUT)
        return response

    def cache_post(self, name: str, config: str) -> requests.Response:
        """Create a cache
        Args:
            name(str): name of the cache.
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.post(
            api_url,
            config,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def schema_delete(self, name: str) -> requests.Response:
        """Delete a schema
        Args:
            name(str): name of the schema.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_delete(self, name: str) -> requests.Response:
        """Delete a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_clear(self, cache_name: str) -> requests.Response:
        """Clear a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node + self._cache_url + "/" + cache_name + "?action=clear"
        )
        response = requests.post(api_url, timeout=REST_TIMEOUT)
        return response

    def index_clear(self, cache_name: str) -> requests.Response:
        """Clear an index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "/search/indexes?action=clear"
        )
        return requests.post(api_url, timeout=REST_TIMEOUT)

    def index_reindex(self, cache_name: str) -> requests.Response:
        """Rebuild index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "/search/indexes?action=reindex"
        )
        return requests.post(api_url, timeout=REST_TIMEOUT)
